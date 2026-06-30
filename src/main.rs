mod audio;
mod config;
mod error;
mod history;
mod llm;
mod matrix;
mod metrics;
mod room_profiles;

use anyhow::Result;
use matrix_sdk::config::SyncSettings;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{info, warn};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::config::Config;
use crate::matrix::handler::{BotCtx, ReloadableState};

/// Initialize tracing: human-readable to stderr, JSON with daily rotation to a
/// log directory (`ROGER_LOG_DIR`, default `roger_session/logs`). The returned
/// guard must be kept alive for the lifetime of the process so the non-blocking
/// writer flushes on shutdown.
fn init_logging() -> WorkerGuard {
    let log_dir = std::env::var("ROGER_LOG_DIR").unwrap_or_else(|_| "roger_session/logs".into());
    let file_appender = tracing_appender::rolling::daily(&log_dir, "roger.log");
    let (file_writer, guard) = tracing_appender::non_blocking(file_appender);

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "roger=info,matrix_sdk=warn".into());

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(fmt::layer().json().with_ansi(false).with_writer(file_writer))
        .init();

    guard
}

/// Listen for SIGHUP and hot-reload the reloadable parts of the config
/// (LLM client, system prompt, per-room settings). Matrix credentials and the
/// room allowlist are fixed for the process lifetime and require a restart.
async fn reload_on_sighup(config_dir: PathBuf, state: Arc<RwLock<ReloadableState>>) {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sighup = match signal(SignalKind::hangup()) {
        Ok(s) => s,
        Err(e) => {
            warn!("failed to install SIGHUP handler, hot-reload disabled: {}", e);
            return;
        }
    };
    info!("SIGHUP hot-reload enabled (send `kill -HUP <pid>` or `systemctl --user reload roger`)");

    while sighup.recv().await.is_some() {
        info!("SIGHUP received — reloading config");
        let cfg = match Config::load(&config_dir) {
            Ok(c) => c,
            Err(e) => {
                warn!("config reload failed (keeping current config): {}", e);
                continue;
            }
        };
        let llms = match cfg.build_all_llms() {
            Ok(v) => v,
            Err(e) => {
                warn!("config reload failed building LLMs (keeping current config): {}", e);
                continue;
            }
        };
        let mut st = state.write().await;
        st.llms = llms.into_iter().map(|(k, v)| (k, Arc::new(v))).collect();
        st.system_prompt = cfg.system_prompt;
        st.room_configs = cfg.rooms;
        // Drop runtime /model overrides that point at a profile that no longer builds.
        let valid: HashSet<String> = st.llms.keys().cloned().collect();
        st.room_profiles.retain(|_, profile| valid.contains(profile));
        info!(
            "config reloaded — profiles: {}, rooms: {}",
            st.llms.len(),
            st.room_configs.len()
        );
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let _log_guard = init_logging();

    let config_dir = PathBuf::from("config");
    let cfg = Config::load(&config_dir)?;

    info!("roger starting — homeserver: {}", cfg.matrix_homeserver);
    info!("allowlist: {:?}", cfg.room_allowlist);

    // Build an LLM per profile (primary + fallbacks; chat required)
    let llms: HashMap<String, Arc<llm::ProfileLlm>> = cfg
        .build_all_llms()?
        .into_iter()
        .map(|(k, v)| (k, Arc::new(v)))
        .collect();
    for (name, client) in &llms {
        info!("profile '{}' → {}", name, client.model_chain().join(" → "));
    }

    // Build speaches client if configured
    let speaches = cfg.speaches_url.as_ref().map(|url| {
        info!("speaches: {}", url);
        Arc::new(audio::SpeachesClient::new(url.clone()))
    });

    let session_dir = PathBuf::from("roger_session");
    let client = matrix::client::build_client(&cfg.matrix_homeserver, &session_dir).await?;
    matrix::client::login(&client, &cfg.matrix_user, &cfg.matrix_password, &session_dir).await?;

    let bot_user_id = client
        .user_id()
        .ok_or_else(|| anyhow::anyhow!("not logged in"))?
        .to_string();

    let bot_localpart = bot_user_id
        .trim_start_matches('@')
        .split(':')
        .next()
        .unwrap_or(&bot_user_id)
        .to_string();

    for (room_id, room_cfg) in &cfg.rooms {
        info!(
            room = %room_id,
            name = %room_cfg.name,
            require_mention = room_cfg.require_mention,
            "room config"
        );
    }

    let history = Arc::new(history::HistoryStore::new(session_dir.join("history"))?);
    info!("history store initialized");

    // Load persisted /model overrides, dropping any that name a profile that
    // isn't built on this host.
    let room_profile_store =
        Arc::new(room_profiles::RoomProfileStore::new(session_dir.join("room_profiles.json")));
    let mut room_profiles_map = room_profile_store.load();
    room_profiles_map.retain(|_, profile| llms.contains_key(profile));
    if !room_profiles_map.is_empty() {
        info!("loaded {} persisted /model override(s)", room_profiles_map.len());
    }

    let state = Arc::new(RwLock::new(ReloadableState {
        llms,
        system_prompt: cfg.system_prompt,
        room_configs: cfg.rooms,
        room_profiles: room_profiles_map,
    }));

    // Spawn the SIGHUP hot-reload listener
    tokio::spawn(reload_on_sighup(config_dir, state.clone()));

    let bot_ctx = BotCtx {
        allowed_rooms: HashSet::from_iter(cfg.room_allowlist.iter().cloned()),
        bot_user_id,
        bot_localpart,
        speaches,
        history,
        started_at: Arc::new(Instant::now()),
        state,
        room_profiles: room_profile_store,
        metrics: Arc::new(metrics::Metrics::default()),
    };

    client.add_event_handler_context(bot_ctx);
    client.add_event_handler(matrix::handler::handle_invite);
    client.add_event_handler(matrix::handler::handle_message);

    info!("sync loop starting");
    client.sync(SyncSettings::default()).await?;

    Ok(())
}

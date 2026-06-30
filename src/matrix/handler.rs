use crate::{
    audio::SpeachesClient,
    config::RoomConfig,
    history::{ChatMessage, HistoryStore},
    llm::ProfileLlm,
    metrics::Metrics,
    room_profiles::RoomProfileStore,
};
use matrix_sdk::{
    event_handler::Ctx,
    media::{MediaFormat, MediaRequestParameters},
    room::Room,
    ruma::{
        events::room::{
            member::{MembershipState, StrippedRoomMemberEvent},
            message::{
                AudioMessageEventContent, MessageType, OriginalSyncRoomMessageEvent,
                ReplacementMetadata, RoomMessageEventContent,
            },
        },
        OwnedEventId,
    },
    Client,
};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, RwLock};
use tracing::{info, warn};

/// Streaming flush cadence. We post/update the response when a sentence completes
/// OR `MAX_FLUSH_WAIT_MS` has passed since the last update (whichever first), but
/// never more often than `MIN_FLUSH_GAP_MS`. The typing indicator covers the gap
/// before the first flush, so there's no placeholder message.
const MIN_FLUSH_GAP_MS: u64 = 250;
const MAX_FLUSH_WAIT_MS: u64 = 1000;

/// Byte index of the last sentence-ending boundary in `s` (`.`, `!`, `?`, or a
/// newline), if any.
fn last_sentence_end(s: &str) -> Option<usize> {
    s.rfind(|c: char| matches!(c, '.' | '!' | '?' | '\n'))
}

/// Decide whether to flush a streamed update. Flush when the text changed, the
/// rate floor has passed, and either a new sentence boundary appeared or the time
/// ceiling was reached.
fn should_flush(
    changed: bool,
    elapsed: Duration,
    sentence_ready: bool,
    min_gap: Duration,
    max_wait: Duration,
) -> bool {
    changed && elapsed >= min_gap && (sentence_ready || elapsed >= max_wait)
}

/// Edit a previously sent message in place via an `m.replace` relation.
async fn edit_message(room: &Room, id: OwnedEventId, text: &str) {
    let edited = RoomMessageEventContent::text_plain(text)
        .make_replacement(ReplacementMetadata::new(id, None), None);
    if let Err(e) = room.send(edited).await {
        warn!("failed to edit message: {}", e);
    }
}

/// Config that can be swapped at runtime on SIGHUP without restarting the bot.
/// Everything reachable behind `BotCtx::state` is reloadable; fields directly on
/// `BotCtx` are fixed for the process lifetime.
pub struct ReloadableState {
    /// LLM (primary + fallbacks) per profile name (e.g. "chat", "reason", "code").
    pub llms: HashMap<String, Arc<ProfileLlm>>,
    pub system_prompt: String,
    pub room_configs: HashMap<String, RoomConfig>,
    /// Runtime per-room profile overrides set via `/model`, keyed by room id.
    pub room_profiles: HashMap<String, String>,
}

impl ReloadableState {
    /// The configured profile name for a room: runtime `/model` override, else
    /// the room's `profile` config, else "chat".
    pub fn profile_name_for_room(&self, room_id: &str) -> String {
        if let Some(p) = self.room_profiles.get(room_id) {
            return p.clone();
        }
        self.room_configs
            .get(room_id)
            .and_then(|r| r.profile.clone())
            .unwrap_or_else(|| "chat".to_string())
    }

    /// Resolve the LLM for a room. Returns the profile LLM and the profile name
    /// actually used (falls back to "chat" if the requested profile has no built
    /// client).
    pub fn llm_for_room(&self, room_id: &str) -> (Arc<ProfileLlm>, String) {
        let requested = self.profile_name_for_room(room_id);
        if let Some(c) = self.llms.get(&requested) {
            return (c.clone(), requested);
        }
        let chat = self.llms.get("chat").expect("chat client always present");
        (chat.clone(), "chat".to_string())
    }
}

#[derive(Clone)]
pub struct BotCtx {
    pub allowed_rooms: std::collections::HashSet<String>,
    pub bot_user_id: String,
    pub bot_localpart: String,
    pub speaches: Option<Arc<SpeachesClient>>,
    pub history: Arc<HistoryStore>,
    pub started_at: Arc<Instant>,
    pub state: Arc<RwLock<ReloadableState>>,
    /// Persists `/model` runtime overrides across restarts.
    pub room_profiles: Arc<RoomProfileStore>,
    /// Process-lifetime response counters.
    pub metrics: Arc<Metrics>,
}

impl BotCtx {
    fn is_mentioned(&self, body: &str) -> bool {
        let needle_full = format!("@{}", self.bot_user_id);
        let needle_local = format!("@{}", self.bot_localpart);
        body.contains(&needle_full) || body.contains(&needle_local)
    }
}

pub async fn handle_invite(
    event: StrippedRoomMemberEvent,
    room: Room,
    client: Client,
    ctx: Ctx<BotCtx>,
) {
    let room_id = room.room_id().to_string();

    if event.state_key != client.user_id().map(|u| u.to_string()).unwrap_or_default() {
        return;
    }
    if event.content.membership != MembershipState::Invite {
        return;
    }

    if ctx.allowed_rooms.contains(&room_id) {
        info!("accepting invite to allowed room {}", room_id);
        if let Err(e) = room.join().await {
            warn!("failed to join room {}: {}", room_id, e);
        }
    } else {
        warn!("declining invite to non-allowlisted room {}", room_id);
    }
}

pub async fn handle_message(
    event: OriginalSyncRoomMessageEvent,
    room: Room,
    client: Client,
    ctx: Ctx<BotCtx>,
) {
    let room_id = room.room_id().to_string();

    if !ctx.allowed_rooms.contains(&room_id) {
        return;
    }
    if event.sender.to_string() == ctx.bot_user_id {
        return;
    }

    // Resolve message body — text directly, audio via transcription
    let body = match &event.content.msgtype {
        MessageType::Text(text) => text.body.clone(),

        MessageType::Audio(audio) => {
            match transcribe_audio(&client, audio, &ctx).await {
                Ok(transcript) => {
                    info!(room = %room_id, "transcribed audio: {}", transcript);
                    transcript
                }
                Err(e) => {
                    warn!("audio transcription failed: {}", e);
                    let _ = room
                        .send(RoomMessageEventContent::text_plain(
                            "Sorry, I couldn't transcribe that audio.",
                        ))
                        .await;
                    return;
                }
            }
        }

        _ => return,
    };

    // Mention gate — read the (reloadable) per-room config
    let require_mention = ctx
        .state
        .read()
        .await
        .room_configs
        .get(&room_id)
        .map(|r| r.require_mention)
        .unwrap_or(true);
    if require_mention && !ctx.is_mentioned(&body) {
        return;
    }

    info!(room = %room_id, sender = %event.sender, "processing: {}", body);

    // Handle slash commands without touching the LLM — reply directly, no placeholder.
    if let Some(cmd_reply) = handle_slash_command(&body, &room_id, &ctx).await {
        if let Err(e) = room.send(RoomMessageEventContent::text_plain(&cmd_reply)).await {
            warn!("failed to send command reply: {}", e);
        }
        return;
    }

    // Show the typing indicator while we wait for the model — no placeholder message.
    let _ = room.typing_notice(true).await;

    // Build context: system prompt + history + current message.
    // Snapshot the reloadable bits (LLM client + resolved system prompt) and drop
    // the read lock before the long-running LLM call so SIGHUP reloads aren't blocked.
    if let Err(e) = ctx.history.append(&room_id, ChatMessage::user(&body)) {
        warn!("failed to save user message to history: {}", e);
    }
    let (llm, system_prompt, profile, model) = {
        let st = ctx.state.read().await;
        let prompt = st
            .room_configs
            .get(&room_id)
            .and_then(|r| r.system_prompt.clone())
            .unwrap_or_else(|| st.system_prompt.clone());
        let (client, profile) = st.llm_for_room(&room_id);
        let model = client.model().to_string();
        (client, prompt, profile, model)
    };
    let budget = llm.history_token_budget(crate::history::estimate_tokens(&system_prompt));
    let mut messages = vec![ChatMessage::system(&system_prompt)];
    messages.extend(ctx.history.windowed_by_tokens(&room_id, budget));

    // Stream the LLM response. We flush (post the first message, then edit it in
    // place) on sentence boundaries or once MAX_FLUSH_WAIT_MS passes — whichever
    // comes first — but never more often than MIN_FLUSH_GAP_MS. The typing
    // indicator covers the wait, so there's no "Working on it…" placeholder.
    let req_start = Instant::now();
    let (tx, mut rx) = mpsc::channel::<String>(64);
    let stream_handle = {
        let llm = llm.clone();
        let messages = messages.clone();
        tokio::spawn(async move { llm.chat_stream(&messages, tx).await })
    };

    let min_gap = Duration::from_millis(MIN_FLUSH_GAP_MS);
    let max_wait = Duration::from_millis(MAX_FLUSH_WAIT_MS);
    let mut msg_id: Option<OwnedEventId> = None;
    let mut last_flush: Option<Instant> = None;
    let mut shown = String::new();
    while let Some(acc) = rx.recv().await {
        // Time since the last flush, or since the request started for the first one.
        let elapsed = last_flush.unwrap_or(req_start).elapsed();
        let sentence_ready = last_sentence_end(&acc).map_or(false, |i| i >= shown.len());
        if !should_flush(acc != shown, elapsed, sentence_ready, min_gap, max_wait) {
            continue;
        }
        match &msg_id {
            None => match room.send(RoomMessageEventContent::text_plain(&acc)).await {
                Ok(resp) => msg_id = Some(resp.event_id),
                Err(e) => {
                    warn!("failed to send first response message: {}", e);
                    continue;
                }
            },
            Some(id) => edit_message(&room, id.clone(), &acc).await,
        }
        shown = acc;
        last_flush = Some(Instant::now());
    }

    let streamed = stream_handle.await;
    let _ = room.typing_notice(false).await;

    // Resolve the final text: the streamed result, falling back to a non-streaming
    // request if the stream errored or produced nothing (backend without SSE support).
    let result: anyhow::Result<String> = match streamed {
        Ok(Ok(text)) if !text.trim().is_empty() => Ok(text),
        Ok(Ok(_)) => llm.chat(&messages).await,
        Ok(Err(e)) => {
            warn!("stream error, falling back to non-streaming: {}", e);
            llm.chat(&messages).await
        }
        Err(e) => Err(anyhow::anyhow!("stream task failed: {}", e)),
    };

    // Record metrics and emit a structured per-request log line.
    let latency_ms = req_start.elapsed().as_millis() as u64;
    let ok = result.is_ok();
    ctx.metrics.record(latency_ms, ok);
    info!(
        room = %room_id,
        profile = %profile,
        model = %model,
        latency_ms,
        ok,
        "responded"
    );

    // Render the final text: edit the message in place if we already posted one,
    // otherwise send it fresh (short responses that never hit the paint threshold).
    let final_text = match result {
        Ok(reply) => {
            if let Err(e) = ctx.history.append(&room_id, ChatMessage::assistant(&reply)) {
                warn!("failed to save assistant reply to history: {}", e);
            }
            reply
        }
        Err(e) => {
            warn!("LLM error: {}", e);
            format!("Sorry, I hit an error: {}", e)
        }
    };

    match msg_id {
        Some(id) => {
            if final_text != shown {
                edit_message(&room, id, &final_text).await;
            }
        }
        None => {
            let text = if final_text.trim().is_empty() {
                "(no response)".to_string()
            } else {
                final_text
            };
            if let Err(e) = room.send(RoomMessageEventContent::text_plain(&text)).await {
                warn!("failed to send response: {}", e);
            }
        }
    }
}

/// Returns Some(reply) if the message is a slash command, None otherwise.
async fn handle_slash_command(body: &str, room_id: &str, ctx: &BotCtx) -> Option<String> {
    let trimmed = body.trim();
    if !trimmed.starts_with('/') {
        return None;
    }
    let parts: Vec<&str> = trimmed.splitn(2, ' ').collect();
    let cmd = parts[0].to_lowercase();

    match cmd.as_str() {
        "/help" => Some(
            "**Roger commands**\n\
             `/help` — show this list\n\
             `/clear` — wipe conversation history for this room\n\
             `/status` — show uptime, model, and history stats\n\
             `/model [name]` — show/switch this room's LLM profile (`/model reset` to revert)"
                .to_string(),
        ),
        "/clear" => {
            if let Err(e) = ctx.history.clear(room_id) {
                Some(format!("Failed to clear history: {}", e))
            } else {
                Some("History cleared.".to_string())
            }
        }
        "/status" => {
            let uptime = ctx.started_at.elapsed();
            let secs = uptime.as_secs();
            let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
            let history_len = ctx.history.windowed(room_id, 100).len();
            let (client, profile) = ctx.state.read().await.llm_for_room(room_id);
            let m_snap = ctx.metrics.snapshot();
            let model_desc = if client.fallback_count() > 0 {
                format!("{} (+{} fallback)", client.model(), client.fallback_count())
            } else {
                client.model().to_string()
            };
            Some(format!(
                "**Roger status**\nUptime: {}h {}m {}s\nProfile: {} ({})\nHistory: {} messages (this room)\nRequests: {} ({} errors), avg {}ms",
                h, m, s, profile, model_desc, history_len,
                m_snap.requests, m_snap.errors, m_snap.avg_latency_ms
            ))
        }
        "/model" => Some(handle_model_command(parts.get(1).copied(), room_id, ctx).await),
        _ => Some(format!("Unknown command `{}`. Try `/help`.", parts[0])),
    }
}

/// `/model` — show or switch the LLM profile for this room. Persists overrides.
async fn handle_model_command(arg: Option<&str>, room_id: &str, ctx: &BotCtx) -> String {
    let arg = arg.map(str::trim).unwrap_or("");

    // No argument: report current profile + the available ones.
    if arg.is_empty() {
        let st = ctx.state.read().await;
        let current = st.profile_name_for_room(room_id);
        let mut names: Vec<String> = st.llms.keys().cloned().collect();
        names.sort();
        let list = names
            .iter()
            .map(|n| format!("`{}`", n))
            .collect::<Vec<_>>()
            .join(", ");
        return format!(
            "Current profile: `{}`\nAvailable: {}\n`/model <name>` to switch · `/model reset` to revert to default",
            current, list
        );
    }

    let reset = arg == "reset" || arg == "default";

    // Mutate the in-memory map under the write lock, then snapshot it for persistence.
    let snapshot = {
        let mut st = ctx.state.write().await;
        if reset {
            st.room_profiles.remove(room_id);
        } else if st.llms.contains_key(arg) {
            st.room_profiles.insert(room_id.to_string(), arg.to_string());
        } else {
            let mut names: Vec<String> = st.llms.keys().cloned().collect();
            names.sort();
            return format!(
                "Unknown profile `{}`. Available: {}",
                arg,
                names.join(", ")
            );
        }
        st.room_profiles.clone()
    };

    if let Err(e) = ctx.room_profiles.save(&snapshot) {
        warn!("failed to persist room profiles: {}", e);
    }

    let (client, profile) = ctx.state.read().await.llm_for_room(room_id);
    if reset {
        format!("Reset to default profile: `{}` ({})", profile, client.model())
    } else {
        format!("Switched to profile `{}` ({})", profile, client.model())
    }
}

async fn transcribe_audio(
    client: &Client,
    audio: &AudioMessageEventContent,
    ctx: &BotCtx,
) -> anyhow::Result<String> {
    let speaches = ctx
        .speaches
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("SPEACHES_URL not configured"))?;

    let bytes = client
        .media()
        .get_media_content(
            &MediaRequestParameters {
                source: audio.source.clone(),
                format: MediaFormat::File,
            },
            true,
        )
        .await?;

    speaches.transcribe(bytes, "audio.ogg").await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::HistoryStore;
    use crate::llm::LlmClient;
    use crate::room_profiles::RoomProfileStore;
    use std::collections::HashSet;
    use tempfile::TempDir;

    fn client(model: &str) -> Arc<ProfileLlm> {
        Arc::new(ProfileLlm::new(vec![Arc::new(LlmClient::new(
            "http://localhost/v1".into(),
            model.into(),
            None,
            128,
            0.0,
            8192,
        ))]))
    }

    fn state() -> ReloadableState {
        let mut llms = HashMap::new();
        llms.insert("chat".to_string(), client("chat-model"));
        llms.insert("reason".to_string(), client("reason-model"));
        let mut room_configs = HashMap::new();
        room_configs.insert(
            "!coding:srv".to_string(),
            RoomConfig {
                name: "Coding".into(),
                require_mention: true,
                system_prompt: None,
                profile: Some("reason".into()),
            },
        );
        ReloadableState {
            llms,
            system_prompt: "sys".into(),
            room_configs,
            room_profiles: HashMap::new(),
        }
    }

    #[test]
    fn unconfigured_room_uses_chat() {
        let st = state();
        let (client, profile) = st.llm_for_room("!unknown:srv");
        assert_eq!(profile, "chat");
        assert_eq!(client.model(), "chat-model");
    }

    #[test]
    fn room_config_profile_is_used() {
        let st = state();
        let (client, profile) = st.llm_for_room("!coding:srv");
        assert_eq!(profile, "reason");
        assert_eq!(client.model(), "reason-model");
    }

    #[test]
    fn runtime_override_beats_room_config() {
        let mut st = state();
        st.room_profiles
            .insert("!coding:srv".to_string(), "chat".to_string());
        let (client, profile) = st.llm_for_room("!coding:srv");
        assert_eq!(profile, "chat");
        assert_eq!(client.model(), "chat-model");
    }

    #[test]
    fn missing_profile_falls_back_to_chat() {
        let mut st = state();
        st.room_profiles
            .insert("!coding:srv".to_string(), "nonexistent".to_string());
        let (client, profile) = st.llm_for_room("!coding:srv");
        assert_eq!(profile, "chat");
        assert_eq!(client.model(), "chat-model");
    }

    fn test_ctx(dir: &TempDir) -> BotCtx {
        BotCtx {
            allowed_rooms: HashSet::new(),
            bot_user_id: "@roger:srv".into(),
            bot_localpart: "roger".into(),
            speaches: None,
            history: Arc::new(HistoryStore::new(dir.path().join("history")).unwrap()),
            started_at: Arc::new(Instant::now()),
            state: Arc::new(RwLock::new(state())),
            room_profiles: Arc::new(RoomProfileStore::new(dir.path().join("rp.json"))),
            metrics: Arc::new(Metrics::default()),
        }
    }

    #[tokio::test]
    async fn model_no_arg_lists_available() {
        let dir = TempDir::new().unwrap();
        let ctx = test_ctx(&dir);
        let out = handle_model_command(None, "!coding:srv", &ctx).await;
        assert!(out.contains("Current profile: `reason`"));
        assert!(out.contains("`chat`") && out.contains("`reason`"));
    }

    #[tokio::test]
    async fn model_switch_persists_and_applies() {
        let dir = TempDir::new().unwrap();
        let ctx = test_ctx(&dir);
        let out = handle_model_command(Some("chat"), "!coding:srv", &ctx).await;
        assert!(out.contains("Switched to profile `chat`"));
        // In-memory state updated
        assert_eq!(
            ctx.state.read().await.profile_name_for_room("!coding:srv"),
            "chat"
        );
        // Persisted to disk
        assert_eq!(
            ctx.room_profiles.load().get("!coding:srv").map(String::as_str),
            Some("chat")
        );
    }

    #[tokio::test]
    async fn model_unknown_is_rejected() {
        let dir = TempDir::new().unwrap();
        let ctx = test_ctx(&dir);
        let out = handle_model_command(Some("bogus"), "!coding:srv", &ctx).await;
        assert!(out.starts_with("Unknown profile `bogus`"));
        // Unchanged: still the room-config default
        assert_eq!(
            ctx.state.read().await.profile_name_for_room("!coding:srv"),
            "reason"
        );
        assert!(ctx.room_profiles.load().is_empty());
    }

    #[test]
    fn last_sentence_end_finds_boundaries() {
        assert_eq!(last_sentence_end("no boundary here"), None);
        assert_eq!(last_sentence_end("Hello there."), Some(11));
        // Returns the *last* boundary.
        assert_eq!(last_sentence_end("One. Two!"), Some(8));
        assert!(last_sentence_end("line one\nline two").is_some());
    }

    #[test]
    fn should_flush_respects_rate_floor() {
        let min = Duration::from_millis(250);
        let max = Duration::from_millis(1000);
        // Sentence ready but too soon since last flush → no.
        assert!(!should_flush(true, Duration::from_millis(100), true, min, max));
        // Sentence ready and past the floor → yes.
        assert!(should_flush(true, Duration::from_millis(300), true, min, max));
    }

    #[test]
    fn should_flush_time_ceiling_without_sentence() {
        let min = Duration::from_millis(250);
        let max = Duration::from_millis(1000);
        // No sentence, under the ceiling → no.
        assert!(!should_flush(true, Duration::from_millis(500), false, min, max));
        // No sentence, but ceiling reached → yes.
        assert!(should_flush(true, Duration::from_millis(1000), false, min, max));
    }

    #[test]
    fn should_flush_requires_change() {
        let min = Duration::from_millis(250);
        let max = Duration::from_millis(1000);
        // Unchanged text never flushes, even past the ceiling.
        assert!(!should_flush(false, Duration::from_secs(5), true, min, max));
    }

    #[tokio::test]
    async fn model_reset_clears_override() {
        let dir = TempDir::new().unwrap();
        let ctx = test_ctx(&dir);
        handle_model_command(Some("chat"), "!coding:srv", &ctx).await;
        let out = handle_model_command(Some("reset"), "!coding:srv", &ctx).await;
        assert!(out.contains("Reset to default profile: `reason`"));
        assert!(ctx.room_profiles.load().is_empty());
    }
}

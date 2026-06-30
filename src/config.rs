use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use chrono::Local;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CommsMode {
    Sync,
    Async,
    Auto,
}

impl Default for CommsMode {
    fn default() -> Self {
        CommsMode::Auto
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LatencyClass {
    Fast,
    Slow,
}

impl Default for LatencyClass {
    fn default() -> Self {
        LatencyClass::Fast
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum BackendKind {
    OpenAi,
    ClaudeCode,
    OpenCode,
}

impl Default for BackendKind {
    fn default() -> Self {
        BackendKind::OpenAi
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProfileConfig {
    pub backend: String,
    /// Ordered fallback backend names, tried (in order) when the primary
    /// `backend` is unreachable or errors. Same profile params apply to each.
    #[serde(default)]
    pub fallback: Vec<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Approximate context window (tokens) for this profile's model. Used to size
    /// the conversation-history budget. Defaults to 8192.
    #[serde(default)]
    pub context_tokens: Option<u32>,
    #[serde(default)]
    pub comms: CommsMode,
    #[serde(default)]
    pub latency_class: LatencyClass,
    #[serde(default)]
    pub idle_timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BackendConfig {
    #[serde(default)]
    pub kind: BackendKind,
    pub base_url: String,
    pub model: String,
    #[serde(default)]
    pub api_key_env: String,
}

impl BackendConfig {
    pub fn api_key(&self) -> Option<String> {
        if self.api_key_env.is_empty() {
            None
        } else {
            env::var(&self.api_key_env).ok()
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct CommsConfig {
    #[serde(default = "default_sync_budget")]
    pub sync_budget_ms: u64,
    #[serde(default = "default_idle_timeout")]
    pub idle_timeout_ms: u64,
    #[serde(default = "default_absolute_ceiling")]
    pub absolute_ceiling_ms: u64,
    #[serde(default = "default_soft_worker_cap")]
    pub soft_worker_cap: usize,
    #[serde(default = "default_max_concurrent_children")]
    pub max_concurrent_children: usize,
    #[serde(default = "default_edit_debounce")]
    pub edit_debounce_ms: u64,
}

fn default_sync_budget() -> u64 { 7000 }
fn default_idle_timeout() -> u64 { 60000 }
fn default_absolute_ceiling() -> u64 { 1800000 }
fn default_soft_worker_cap() -> usize { 4 }
fn default_max_concurrent_children() -> usize { 3 }
fn default_edit_debounce() -> u64 { 600 }

impl Default for CommsConfig {
    fn default() -> Self {
        CommsConfig {
            sync_budget_ms: default_sync_budget(),
            idle_timeout_ms: default_idle_timeout(),
            absolute_ceiling_ms: default_absolute_ceiling(),
            soft_worker_cap: default_soft_worker_cap(),
            max_concurrent_children: default_max_concurrent_children(),
            edit_debounce_ms: default_edit_debounce(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct RoutingConfig {
    #[serde(default)]
    pub task_profiles: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RoomConfig {
    #[serde(default)]
    pub name: String,
    /// If true (default), bot only responds when @-mentioned.
    /// Set to false for rooms where the bot should respond to every message.
    #[serde(default = "default_require_mention")]
    pub require_mention: bool,
    /// Optional per-room system prompt override. When set, replaces the global
    /// system prompt for this room. Supports the `{date}` placeholder.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Optional LLM profile name for this room (e.g. "chat", "reason", "code").
    /// Defaults to "chat" when unset. Overridable at runtime via `/model`.
    #[serde(default)]
    pub profile: Option<String>,
}

fn default_require_mention() -> bool { true }

impl Default for RoomConfig {
    fn default() -> Self {
        RoomConfig {
            name: String::new(),
            require_mention: true,
            system_prompt: None,
            profile: None,
        }
    }
}

/// Replace the `{date}` placeholder with today's date (YYYY-MM-DD).
fn inject_date(s: &str) -> String {
    s.replace("{date}", &Local::now().format("%Y-%m-%d").to_string())
}

#[derive(Debug, Deserialize)]
struct ProfilesFile {
    profiles: HashMap<String, ProfileConfig>,
    #[serde(default)]
    routing: Option<RoutingConfig>,
    #[serde(default)]
    comms: Option<CommsConfig>,
    #[serde(default)]
    rooms: HashMap<String, RoomConfig>,
}

#[derive(Debug, Deserialize)]
struct BackendsFile {
    backends: HashMap<String, BackendConfig>,
}

#[derive(Debug)]
pub struct Config {
    pub profiles: HashMap<String, ProfileConfig>,
    pub backends: HashMap<String, BackendConfig>,
    pub routing: RoutingConfig,
    pub comms: CommsConfig,
    pub rooms: HashMap<String, RoomConfig>,
    pub matrix_homeserver: String,
    pub matrix_user: String,
    pub matrix_password: String,
    pub room_allowlist: Vec<String>,
    /// URL for the Whisper-compatible transcription service (e.g. speaches)
    pub speaches_url: Option<String>,
    /// System prompt injected as the first message in every LLM call
    pub system_prompt: String,
    /// Base URL for the SearXNG instance used by the web_search tool
    pub searxng_url: Option<String>,
}

impl Config {
    pub fn load(config_dir: &Path) -> anyhow::Result<Self> {
        // Load env
        let _ = dotenvy::dotenv();

        // Load profiles
        let profiles_path = config_dir.join("profiles.toml");
        let profiles_str = fs::read_to_string(&profiles_path)
            .map_err(|e| anyhow::anyhow!("failed to read {}: {}", profiles_path.display(), e))?;
        let profiles_file: ProfilesFile = toml::from_str(&profiles_str)?;

        // Load per-host backends
        let host_role = env::var("HOST_ROLE").unwrap_or_else(|_| "local".to_string());
        let backends_path = config_dir.join(format!("backends.{}.toml", host_role));
        let backends_str = fs::read_to_string(&backends_path)
            .map_err(|e| anyhow::anyhow!("failed to read {}: {}\nCopy backends.example.toml to backends.{}.toml", backends_path.display(), e, host_role))?;
        let backends_file: BackendsFile = toml::from_str(&backends_str)?;

        let matrix_homeserver = env::var("MATRIX_HOMESERVER")
            .map_err(|_| anyhow::anyhow!("MATRIX_HOMESERVER not set"))?;
        let matrix_user = env::var("MATRIX_USER")
            .map_err(|_| anyhow::anyhow!("MATRIX_USER not set"))?;
        let matrix_password = env::var("MATRIX_PASSWORD")
            .map_err(|_| anyhow::anyhow!("MATRIX_PASSWORD not set"))?;
        let room_allowlist = env::var("ROOM_ALLOWLIST")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let speaches_url = env::var("SPEACHES_URL").ok();
        let searxng_url = env::var("SEARXNG_URL").ok();

        // Load system prompt from file, inject current date
        let prompt_path = config_dir.join("system_prompt.txt");
        let raw_prompt = fs::read_to_string(&prompt_path).unwrap_or_else(|_| {
            "You are Roger, a helpful Matrix-native AI assistant.".to_string()
        });
        let system_prompt = inject_date(&raw_prompt);

        // Inject {date} into any per-room system prompt overrides.
        let mut rooms = profiles_file.rooms;
        for room in rooms.values_mut() {
            room.system_prompt = room.system_prompt.as_deref().map(inject_date);
        }

        Ok(Config {
            profiles: profiles_file.profiles,
            backends: backends_file.backends,
            routing: profiles_file.routing.unwrap_or_else(|| RoutingConfig {
                task_profiles: HashMap::new(),
            }),
            comms: profiles_file.comms.unwrap_or_default(),
            rooms,
            matrix_homeserver,
            matrix_user,
            matrix_password,
            room_allowlist,
            speaches_url,
            system_prompt,
            searxng_url,
        })
    }

    /// Build one LLM client for a named backend, applying the given profile's
    /// generation params (temperature, max_tokens, context_tokens).
    fn build_client(
        &self,
        backend_name: &str,
        profile: &ProfileConfig,
    ) -> anyhow::Result<crate::llm::LlmClient> {
        let backend = self.backends.get(backend_name)
            .ok_or_else(|| anyhow::anyhow!("unknown backend '{}'", backend_name))?;
        let base_url = format!(
            "{}/v1",
            backend.base_url.trim_end_matches('/').trim_end_matches("/v1")
        );
        Ok(crate::llm::LlmClient::new(
            base_url,
            backend.model.clone(),
            backend.api_key(),
            profile.max_tokens.unwrap_or(1024),
            profile.temperature.unwrap_or(0.7),
            profile.context_tokens.unwrap_or(8192),
        ))
    }

    /// Build the `ProfileLlm` for a named profile: its primary backend followed by
    /// any `fallback` backends, in order. Backends that don't exist on this host
    /// are skipped with a warning; the profile fails only if none are usable.
    pub fn build_profile_llm(&self, profile_name: &str) -> anyhow::Result<crate::llm::ProfileLlm> {
        let profile = self.profiles.get(profile_name)
            .ok_or_else(|| anyhow::anyhow!("unknown profile: {}", profile_name))?;
        let mut names = vec![profile.backend.clone()];
        names.extend(profile.fallback.iter().cloned());

        let mut clients = Vec::new();
        for name in &names {
            match self.build_client(name, profile) {
                Ok(c) => clients.push(std::sync::Arc::new(crate::llm::Backend::Http(c))),
                Err(e) => {
                    tracing::warn!("profile '{}': skipping backend '{}': {}", profile_name, name, e)
                }
            }
        }
        if clients.is_empty() {
            anyhow::bail!("profile '{}' has no usable backend", profile_name);
        }
        Ok(crate::llm::ProfileLlm::new(clients))
    }

    /// Build a `ProfileLlm` for every defined profile. Profiles with no usable
    /// backend are skipped with a warning rather than aborting startup. The "chat"
    /// profile is required. Shared by startup and config hot-reload.
    pub fn build_all_llms(&self) -> anyhow::Result<HashMap<String, crate::llm::ProfileLlm>> {
        let mut profiles = HashMap::new();
        for name in self.profiles.keys() {
            match self.build_profile_llm(name) {
                Ok(p) => {
                    profiles.insert(name.clone(), p);
                }
                Err(e) => tracing::warn!("skipping profile '{}': {}", name, e),
            }
        }
        if !profiles.contains_key("chat") {
            anyhow::bail!("the 'chat' profile is required but failed to build");
        }
        Ok(profiles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed profiles.toml must parse, including per-room overrides.
    #[test]
    fn committed_profiles_toml_parses() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/config/profiles.toml");
        let s = fs::read_to_string(path).expect("read profiles.toml");
        let parsed: ProfilesFile = toml::from_str(&s).expect("parse profiles.toml");
        assert!(parsed.profiles.contains_key("chat"), "chat profile required");
        // At least one room defines a system_prompt override.
        assert!(
            parsed.rooms.values().any(|r| r.system_prompt.is_some()),
            "expected a per-room system_prompt example"
        );
    }

    #[test]
    fn room_without_override_defaults_to_none() {
        let toml = r#"
            [profiles.chat]
            backend = "x"
            [rooms."!a:b"]
            name = "Plain"
            require_mention = true
        "#;
        let parsed: ProfilesFile = toml::from_str(toml).unwrap();
        let room = parsed.rooms.get("!a:b").unwrap();
        assert!(room.system_prompt.is_none());
        assert!(room.require_mention);
    }

    #[test]
    fn inject_date_replaces_placeholder() {
        let out = inject_date("today is {date}.");
        assert!(!out.contains("{date}"));
        assert!(out.starts_with("today is "));
    }
}

use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

/// Persists the per-room agentic working directory (set via the `set_workdir`
/// tool) to a single JSON file so it survives restarts. Keyed by room id →
/// absolute path. Self-contained (interior `Mutex`) so it can be shared between
/// the tool executor (writes) and the handler (reads) behind an `Arc`.
pub struct RoomWorkdirStore {
    path: PathBuf,
    map: Mutex<HashMap<String, String>>,
}

impl RoomWorkdirStore {
    /// Load from disk (empty if missing/invalid).
    pub fn load(path: PathBuf) -> Self {
        let map = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        RoomWorkdirStore {
            path,
            map: Mutex::new(map),
        }
    }

    pub fn get(&self, room: &str) -> Option<String> {
        self.map.lock().unwrap().get(room).cloned()
    }

    /// Set the workdir for a room and persist the whole map.
    pub fn set(&self, room: &str, workdir: &str) -> Result<()> {
        let snapshot = {
            let mut map = self.map.lock().unwrap();
            map.insert(room.to_string(), workdir.to_string());
            map.clone()
        };
        std::fs::write(&self.path, serde_json::to_string_pretty(&snapshot)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn missing_file_loads_empty() {
        let dir = TempDir::new().unwrap();
        let store = RoomWorkdirStore::load(dir.path().join("rw.json"));
        assert!(store.get("!a:srv").is_none());
    }

    #[test]
    fn set_then_get_and_persist() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("rw.json");
        let store = RoomWorkdirStore::load(path.clone());
        store.set("!a:srv", "/home/me/proj").unwrap();
        assert_eq!(store.get("!a:srv").as_deref(), Some("/home/me/proj"));
        // Reload from disk sees the persisted value.
        let reloaded = RoomWorkdirStore::load(path);
        assert_eq!(reloaded.get("!a:srv").as_deref(), Some("/home/me/proj"));
    }
}

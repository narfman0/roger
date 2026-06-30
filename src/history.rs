use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Rough token estimate for a message body. Real tokenization is model-specific;
/// ~4 characters per token is a decent cross-model heuristic, plus a small
/// per-message overhead for the role wrapper.
pub fn estimate_tokens(content: &str) -> usize {
    content.len() / 4 + 4
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        ChatMessage { role: "user".into(), content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        ChatMessage { role: "assistant".into(), content: content.into() }
    }

    pub fn system(content: impl Into<String>) -> Self {
        ChatMessage { role: "system".into(), content: content.into() }
    }
}

pub struct HistoryStore {
    data_dir: PathBuf,
}

impl HistoryStore {
    pub fn new(data_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&data_dir)?;
        Ok(HistoryStore { data_dir })
    }

    fn path(&self, room_id: &str) -> PathBuf {
        let safe = room_id.replace(['!', ':', '/'], "_");
        self.data_dir.join(format!("{}.json", safe))
    }

    pub fn load(&self, room_id: &str) -> Vec<ChatMessage> {
        let path = self.path(room_id);
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn append(&self, room_id: &str, msg: ChatMessage) -> Result<()> {
        let mut msgs = self.load(room_id);
        msgs.push(msg);
        std::fs::write(self.path(room_id), serde_json::to_string_pretty(&msgs)?)?;
        Ok(())
    }

    /// Returns up to `max` most recent messages.
    pub fn windowed(&self, room_id: &str, max: usize) -> Vec<ChatMessage> {
        let msgs = self.load(room_id);
        if msgs.len() <= max {
            msgs
        } else {
            msgs[msgs.len() - max..].to_vec()
        }
    }

    /// Returns the most recent messages whose combined estimated token count fits
    /// within `token_budget`, in chronological order. The single most recent
    /// message is always included even if it alone exceeds the budget, so the
    /// user's latest turn is never dropped.
    pub fn windowed_by_tokens(&self, room_id: &str, token_budget: usize) -> Vec<ChatMessage> {
        let msgs = self.load(room_id);
        let mut kept: Vec<ChatMessage> = Vec::new();
        let mut used = 0usize;
        for msg in msgs.into_iter().rev() {
            let cost = estimate_tokens(&msg.content);
            if !kept.is_empty() && used + cost > token_budget {
                break;
            }
            used += cost;
            kept.push(msg);
        }
        kept.reverse();
        kept
    }

    pub fn clear(&self, room_id: &str) -> Result<()> {
        let path = self.path(room_id);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn temp_store() -> (TempDir, HistoryStore) {
        let dir = TempDir::new().unwrap();
        let store = HistoryStore::new(dir.path().to_path_buf()).unwrap();
        (dir, store)
    }

    #[test]
    fn test_empty_room_returns_empty() {
        let (_dir, store) = temp_store();
        assert!(store.load("!room:server").is_empty());
    }

    #[test]
    fn test_append_and_load() {
        let (_dir, store) = temp_store();
        store.append("!room:server", ChatMessage::user("hello")).unwrap();
        store.append("!room:server", ChatMessage::assistant("hi")).unwrap();
        let msgs = store.load("!room:server");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[1].role, "assistant");
    }

    #[test]
    fn test_windowed_truncates() {
        let (_dir, store) = temp_store();
        for i in 0..25u32 {
            store.append("!room:server", ChatMessage::user(i.to_string())).unwrap();
        }
        let windowed = store.windowed("!room:server", 20);
        assert_eq!(windowed.len(), 20);
        assert_eq!(windowed[0].content, "5");
    }

    #[test]
    fn test_windowed_under_max_returns_all() {
        let (_dir, store) = temp_store();
        store.append("!room:server", ChatMessage::user("a")).unwrap();
        store.append("!room:server", ChatMessage::user("b")).unwrap();
        assert_eq!(store.windowed("!room:server", 20).len(), 2);
    }

    #[test]
    fn test_clear() {
        let (_dir, store) = temp_store();
        store.append("!room:server", ChatMessage::user("x")).unwrap();
        store.clear("!room:server").unwrap();
        assert!(store.load("!room:server").is_empty());
    }

    #[test]
    fn test_room_ids_are_isolated() {
        let (_dir, store) = temp_store();
        store.append("!room1:server", ChatMessage::user("room1")).unwrap();
        store.append("!room2:server", ChatMessage::user("room2")).unwrap();
        assert_eq!(store.load("!room1:server").len(), 1);
        assert_eq!(store.load("!room2:server").len(), 1);
    }

    #[test]
    fn test_special_chars_in_room_id_dont_crash() {
        let (_dir, store) = temp_store();
        store.append("!abc/def:host.example.com", ChatMessage::user("hi")).unwrap();
        assert_eq!(store.load("!abc/def:host.example.com").len(), 1);
    }

    #[test]
    fn test_windowed_by_tokens_keeps_recent_within_budget() {
        let (_dir, store) = temp_store();
        // Each body is 35 chars → estimate 35/4 + 4 = 12 tokens.
        for i in 0..10u32 {
            store
                .append("!room:server", ChatMessage::user(format!("message number {:020}", i)))
                .unwrap();
        }
        // Budget 40 tokens fits 3 messages (36), the 4th (48) is dropped.
        let kept = store.windowed_by_tokens("!room:server", 40);
        assert_eq!(kept.len(), 3);
        // Chronological order, newest retained at the end.
        assert!(kept.last().unwrap().content.ends_with("0000000009"));
        assert!(kept.first().unwrap().content.ends_with("0000000007"));
    }

    #[test]
    fn test_windowed_by_tokens_always_keeps_latest() {
        let (_dir, store) = temp_store();
        store
            .append("!room:server", ChatMessage::user("x".repeat(10_000)))
            .unwrap();
        // Budget far too small, but the latest message must still be returned.
        let kept = store.windowed_by_tokens("!room:server", 1);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn test_windowed_by_tokens_empty_room() {
        let (_dir, store) = temp_store();
        assert!(store.windowed_by_tokens("!room:server", 1000).is_empty());
    }
}

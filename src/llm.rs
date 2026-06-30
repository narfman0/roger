use crate::history::ChatMessage;
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    max_tokens: u32,
    temperature: f32,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: AssistantMessage,
}

#[derive(Debug, Deserialize)]
struct AssistantMessage {
    content: String,
}

// --- Streaming (Server-Sent Events) response shapes ---

#[derive(Debug, Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: Delta,
}

#[derive(Debug, Default, Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
}

/// Parse one SSE line. Returns the content delta for `data:` lines carrying a
/// chunk, `None` for comments, blanks, `[DONE]`, or chunks without content.
fn parse_sse_line(line: &str) -> Option<String> {
    let payload = line.strip_prefix("data:")?.trim();
    if payload.is_empty() || payload == "[DONE]" {
        return None;
    }
    let chunk: StreamChunk = serde_json::from_str(payload).ok()?;
    chunk.choices.into_iter().next().and_then(|c| c.delta.content)
}

pub struct LlmClient {
    http: Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
    max_tokens: u32,
    temperature: f32,
    context_tokens: u32,
}

impl LlmClient {
    pub fn new(
        base_url: String,
        model: String,
        api_key: Option<String>,
        max_tokens: u32,
        temperature: f32,
        context_tokens: u32,
    ) -> Self {
        LlmClient {
            http: Client::new(),
            base_url,
            model,
            api_key,
            max_tokens,
            temperature,
            context_tokens,
        }
    }

    /// The model name this client sends requests for.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Max tokens reserved for the model's response.
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    /// Approximate total context window (tokens) for this model.
    pub fn context_tokens(&self) -> u32 {
        self.context_tokens
    }

    /// Token budget available for conversation history, after reserving space for
    /// the system prompt and the response. Never returns less than 256.
    pub fn history_token_budget(&self, system_prompt_tokens: usize) -> usize {
        let reserved = self.max_tokens as usize + system_prompt_tokens + 256; // 256 = safety margin
        (self.context_tokens as usize).saturating_sub(reserved).max(256)
    }

    pub async fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let body = ChatRequest {
            model: &self.model,
            messages,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: false,
        };

        let mut req = self.http.post(&url).json(&body);

        if let Some(key) = &self.api_key {
            if !key.is_empty() {
                req = req.bearer_auth(key);
            }
        }

        let resp = req.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("LLM request failed {}: {}", status, text);
        }

        let chat_resp: ChatResponse = resp.json().await?;
        let content = chat_resp
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_else(|| "(no response)".to_string());

        Ok(content)
    }

    /// Stream a chat completion. The accumulated response text is sent on `tx`
    /// after each content delta; the complete text is also returned. Errors before
    /// the stream starts (HTTP failure) are returned without sending on `tx`.
    pub async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        tx: mpsc::Sender<String>,
    ) -> Result<String> {
        use futures_util::StreamExt;

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let body = ChatRequest {
            model: &self.model,
            messages,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: true,
        };

        let mut req = self.http.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            if !key.is_empty() {
                req = req.bearer_auth(key);
            }
        }

        let resp = req.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("LLM request failed {}: {}", status, text);
        }

        let mut stream = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut full = String::new();

        while let Some(chunk) = stream.next().await {
            buf.extend_from_slice(&chunk?);
            // Process all complete (newline-terminated) lines in the buffer.
            while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = buf.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line_bytes);
                if let Some(delta) = parse_sse_line(line.trim()) {
                    if !delta.is_empty() {
                        full.push_str(&delta);
                        // Receiver gone (e.g. handler bailed) — stop streaming.
                        if tx.send(full.clone()).await.is_err() {
                            break;
                        }
                    }
                }
            }
        }

        Ok(full)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client(max_tokens: u32, context_tokens: u32) -> LlmClient {
        LlmClient::new("u".into(), "m".into(), None, max_tokens, 0.0, context_tokens)
    }

    #[test]
    fn budget_reserves_response_prompt_and_margin() {
        // 8192 - (1024 + 100 + 256) = 6812
        let c = client(1024, 8192);
        assert_eq!(c.history_token_budget(100), 6812);
    }

    #[test]
    fn budget_floors_at_256_when_context_is_tiny() {
        let c = client(1024, 512);
        assert_eq!(c.history_token_budget(100), 256);
    }

    #[test]
    fn sse_extracts_content_delta() {
        let line = r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#;
        assert_eq!(parse_sse_line(line), Some("Hello".to_string()));
    }

    #[test]
    fn sse_ignores_done_and_blanks_and_roles() {
        assert_eq!(parse_sse_line("data: [DONE]"), None);
        assert_eq!(parse_sse_line(""), None);
        assert_eq!(parse_sse_line(": comment"), None);
        // Role-only opening delta has no content.
        assert_eq!(
            parse_sse_line(r#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#),
            None
        );
    }

    #[test]
    fn sse_handles_empty_choices() {
        assert_eq!(parse_sse_line(r#"data: {"choices":[]}"#), None);
    }
}

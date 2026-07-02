use crate::history::ChatMessage;
use crate::subprocess::SubprocessBackend;
use crate::tools::{tool_definitions, ToolCall, ToolExecutor};
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, warn};

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<Value>,
    max_tokens: u32,
    temperature: f32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: AssistantMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AssistantMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
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

    /// Token budget available for conversation history, after reserving space for
    /// the system prompt and the response. Never returns less than 256.
    pub fn history_token_budget(&self, system_prompt_tokens: usize) -> usize {
        let reserved = self.max_tokens as usize + system_prompt_tokens + 256; // 256 = safety margin
        (self.context_tokens as usize).saturating_sub(reserved).max(256)
    }

    fn messages_to_json(messages: &[ChatMessage]) -> Vec<Value> {
        messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect()
    }

    pub async fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        self.chat_raw(Self::messages_to_json(messages), false).await
    }

    async fn chat_raw(&self, messages: Vec<Value>, with_tools: bool) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let body = ChatRequest {
            model: &self.model,
            messages,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: false,
            tools: if with_tools { Some(tool_definitions()) } else { None },
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
            .and_then(|c| c.message.content)
            .unwrap_or_else(|| "(no response)".to_string());

        Ok(content)
    }

    /// Run the tool-use loop: call the LLM with tools enabled, execute any tool
    /// calls, append results, repeat — then stream the final text answer.
    /// Falls back to a plain streaming call when no executor is provided.
    pub async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        executor: Option<&ToolExecutor>,
        tx: mpsc::Sender<String>,
    ) -> Result<String> {
        let Some(exec) = executor else {
            return self.chat_stream(messages, tx).await;
        };

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let mut msg_values: Vec<Value> = Self::messages_to_json(messages);
        const MAX_TOOL_ROUNDS: usize = 5;

        for round in 0..MAX_TOOL_ROUNDS {
            let body = ChatRequest {
                model: &self.model,
                messages: msg_values.clone(),
                max_tokens: self.max_tokens,
                temperature: self.temperature,
                stream: false,
                tools: Some(tool_definitions()),
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
            let choice = chat_resp.choices.into_iter().next()
                .ok_or_else(|| anyhow::anyhow!("empty choices"))?;

            let finish = choice.finish_reason.as_deref().unwrap_or("");

            if finish != "tool_calls" || choice.message.tool_calls.is_none() {
                // Final answer — stream it
                let final_text = choice.message.content.unwrap_or_default();
                if !final_text.is_empty() {
                    let _ = tx.send(final_text.clone()).await;
                    return Ok(final_text);
                }
                // Empty content and no tool calls — fall through to streaming
                break;
            }

            // Execute tool calls and append results
            let tool_calls = choice.message.tool_calls.unwrap();
            info!("tool round {}: {} call(s)", round + 1, tool_calls.len());

            // Append assistant message with tool_calls
            msg_values.push(serde_json::json!({
                "role": "assistant",
                "content": null,
                "tool_calls": tool_calls
            }));

            // Execute each tool and append its result
            for call in &tool_calls {
                let result = exec.execute(call).await;
                info!(tool = %call.function.name, "result length: {}", result.len());
                msg_values.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result
                }));
            }
        }

        // Fallback: non-streaming call with the accumulated tool context.
        // We use non-streaming here so the handler receives exactly one complete
        // message (no intermediate streaming edits), which prevents Matrix clients
        // from showing multiple near-identical messages during a long tool-call response.
        let result = self.chat_raw(msg_values, false).await?;
        let _ = tx.send(result.clone()).await;
        Ok(result)
    }

    /// Stream a chat completion. The accumulated response text is sent on `tx`
    /// after each content delta; the complete text is also returned. Errors before
    /// the stream starts (HTTP failure) are returned without sending on `tx`.
    pub async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        tx: mpsc::Sender<String>,
    ) -> Result<String> {
        self.chat_stream_raw(Self::messages_to_json(messages), tx).await
    }

    async fn chat_stream_raw(
        &self,
        messages: Vec<Value>,
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
            tools: None,
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

/// A single backend behind a profile. Today every backend is an OpenAI-compatible
/// HTTP client; subprocess kinds (`claude-code`, `opencode`) become additional
/// arms here. The fallback chain in `ProfileLlm` is kind-agnostic — it only sees
/// this enum's methods, so a chain can freely mix HTTP and subprocess backends.
pub enum Backend {
    Http(LlmClient),
    Subprocess(SubprocessBackend),
}

impl Backend {
    pub fn model(&self) -> &str {
        match self {
            Backend::Http(c) => c.model(),
            Backend::Subprocess(s) => s.model(),
        }
    }

    pub fn history_token_budget(&self, system_prompt_tokens: usize) -> usize {
        match self {
            Backend::Http(c) => c.history_token_budget(system_prompt_tokens),
            // The subprocess CLI manages its own context; size the transcript we
            // hand it generously and let it compact as needed.
            Backend::Subprocess(_) => 64_000,
        }
    }

    pub async fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        match self {
            Backend::Http(c) => c.chat(messages).await,
            Backend::Subprocess(s) => s.chat(messages).await,
        }
    }

    pub async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        executor: Option<&ToolExecutor>,
        tx: mpsc::Sender<String>,
    ) -> Result<String> {
        match self {
            Backend::Http(c) => c.chat_with_tools(messages, executor, tx).await,
            // The subprocess runs its own tool loop; roger's executor is ignored.
            Backend::Subprocess(s) => s.chat_stream(messages, tx).await,
        }
    }

    /// True for agentic subprocess backends (claude-code / opencode).
    pub fn is_subprocess(&self) -> bool {
        matches!(self, Backend::Subprocess(_))
    }
}

/// An LLM profile backed by an ordered chain of backends: a primary and zero or
/// more fallbacks. Each call tries backends in order, advancing to the next only
/// when one fails to produce a response (transport error or non-2xx). A backend
/// that returns successfully — even with empty text — ends the chain.
pub struct ProfileLlm {
    clients: Vec<Arc<Backend>>,
}

impl ProfileLlm {
    /// `clients` must be non-empty, primary first.
    pub fn new(clients: Vec<Arc<Backend>>) -> Self {
        debug_assert!(!clients.is_empty(), "ProfileLlm requires at least one backend");
        ProfileLlm { clients }
    }

    /// The primary model name (shown in `/status` and logs).
    pub fn model(&self) -> &str {
        self.clients[0].model()
    }

    /// Number of fallback clients behind the primary.
    pub fn fallback_count(&self) -> usize {
        self.clients.len().saturating_sub(1)
    }

    /// True when the primary backend is an agentic subprocess (claude-code).
    pub fn is_subprocess(&self) -> bool {
        self.clients.first().map_or(false, |c| c.is_subprocess())
    }

    /// Model names in priority order (primary first).
    pub fn model_chain(&self) -> Vec<String> {
        self.clients.iter().map(|c| c.model().to_string()).collect()
    }

    /// History token budget, sized from the primary client.
    pub fn history_token_budget(&self, system_prompt_tokens: usize) -> usize {
        self.clients[0].history_token_budget(system_prompt_tokens)
    }

    pub async fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        let mut last_err = None;
        for (i, c) in self.clients.iter().enumerate() {
            match c.chat(messages).await {
                Ok(text) => return Ok(text),
                Err(e) => {
                    warn!(model = %c.model(), "backend {} failed: {}", i, e);
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no backends configured")))
    }

    pub async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        executor: Option<&ToolExecutor>,
        tx: mpsc::Sender<String>,
    ) -> Result<String> {
        let mut last_err = None;
        for (i, c) in self.clients.iter().enumerate() {
            match c.chat_with_tools(messages, executor, tx.clone()).await {
                Ok(text) => return Ok(text),
                Err(e) => {
                    warn!(model = %c.model(), "tool-call backend {} failed: {}", i, e);
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no backends configured")))
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

    fn dead_client(model: &str) -> Arc<Backend> {
        // Port 9 (discard) refuses connections — a reliable transport failure.
        Arc::new(Backend::Http(LlmClient::new(
            "http://127.0.0.1:9/v1".into(),
            model.into(),
            None,
            64,
            0.0,
            4096,
        )))
    }

    #[test]
    fn profile_llm_reports_chain_and_fallback_count() {
        let p = ProfileLlm::new(vec![dead_client("primary"), dead_client("backup")]);
        assert_eq!(p.model(), "primary");
        assert_eq!(p.fallback_count(), 1);
        assert_eq!(p.model_chain(), vec!["primary".to_string(), "backup".to_string()]);
    }

    #[tokio::test]
    async fn profile_llm_errors_when_all_backends_fail() {
        let p = ProfileLlm::new(vec![dead_client("primary"), dead_client("backup")]);
        let (tx, _rx) = mpsc::channel(8);
        assert!(p.chat_with_tools(&[ChatMessage::user("hi")], None, tx).await.is_err());
        assert!(p.chat(&[ChatMessage::user("hi")]).await.is_err());
    }

    // Live failover check: dead primary, real Ollama fallback. Run with
    //   cargo test --release falls_over_to_live_backend -- --ignored
    #[tokio::test]
    #[ignore = "requires a local Ollama at 127.0.0.1:11434 with model gemma4:31b"]
    async fn falls_over_to_live_backend() {
        let primary = dead_client("dead");
        let fallback = Arc::new(Backend::Http(LlmClient::new(
            "http://127.0.0.1:11434/v1".into(),
            "gemma4:31b".into(),
            None,
            1024,
            0.0,
            8192,
        )));
        let p = ProfileLlm::new(vec![primary, fallback]);
        let out = p.chat(&[ChatMessage::user("Reply with exactly: ok")]).await.unwrap();
        assert!(!out.trim().is_empty(), "fallback should produce a response");
    }
}

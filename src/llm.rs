use crate::history::ChatMessage;
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    max_tokens: u32,
    temperature: f32,
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
}

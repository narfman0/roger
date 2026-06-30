# Architecture

## Overview

Roger is a single-process Rust async application using `matrix-sdk` (tokio-based). It syncs with a Matrix homeserver, listens for messages in allowlisted rooms, and responds via LLM.

## Config system

Two-file design keeps secrets off GitHub:

- `config/profiles.toml` — committed. Defines named LLM profiles (`chat`, `fast`, `code`, etc.) and routing rules. References backends by logical name, not URL.
- `config/backends.<HOST_ROLE>.toml` — **gitignored**. Maps logical backend names to real URLs, models, and `api_key_env` variable names. `HOST_ROLE` env var selects which file to load (default: `local`).

API keys are never in config files — only the *name* of the env var holding them.

## Session persistence

On first login, roger saves `roger_session/session.json` (access_token, device_id, user_id). On restart, it restores from this file instead of re-logging in. This prevents a new device ID on every restart, which would conflict with the SQLite E2EE crypto store.

## Conversation history

`HistoryStore` writes one JSON file per room to `roger_session/history/`. Room IDs are sanitized for filesystem safety. Context window is the last 20 messages, passed as the full messages array to the LLM.

History is room-scoped by default. No cross-room sharing (each room has independent context).

### Token budgeting

Context is selected by token budget rather than a fixed message count.
`windowed_by_tokens` walks history newest-first, keeping messages until an
estimated token budget is hit (~4 chars/token heuristic via `estimate_tokens`);
the latest turn is always kept. The budget is
`context_tokens − max_tokens − system_prompt − 256 margin` (floored at 256),
where `context_tokens` is a per-profile config value (default 8192).

## Response UX

For every response:
1. `room.typing_notice(true)` — shows the typing indicator in Matrix clients
2. Send "Working on it…" immediately — user sees activity before LLM responds
3. Stream the LLM response (`LlmClient::chat_stream`): SSE deltas are accumulated
   and pushed over an mpsc channel; the handler edits the ack in place via
   `m.replace` as text grows, debounced to one edit per `STREAM_EDIT_DEBOUNCE_MS`
   (700ms) so a fast stream doesn't flood the room with edit events
4. Final edit with the complete reply; `room.typing_notice(false)`

If the stream errors or yields no content (e.g. a backend without SSE support),
the handler falls back to a single non-streaming `chat()` call. Only the `content`
field is surfaced — reasoning-model `reasoning` deltas are ignored.

## Audio pipeline

1. Matrix sends an `m.audio` event with encrypted media
2. `matrix_sdk::media` downloads and decrypts via `MediaRequestParameters`
3. Raw bytes POST'd to Speaches (`/v1/audio/transcriptions`, model `Systran/faster-whisper-small`)
4. Transcript text fed into normal LLM flow

## LiteLLM proxy

All cloud LLM calls go through a LiteLLM Docker container on `srv:4000`. This:
- Keeps the Anthropic API key on one machine (`srv`) only
- The `ai` machine uses a LiteLLM virtual key (`GATEWAY_VKEY`) with no direct Anthropic access
- Lets backends be swapped without changing roger's config

## Profile routing

`ReloadableState` holds one `ProfileLlm` per profile (`llms: HashMap<profile, ProfileLlm>`),
built from `profiles.toml` at startup and on reload. A profile that has no usable
backend is skipped with a warning; `chat` is required.

### Fallback chains

A `ProfileLlm` wraps an ordered list of clients: the profile's primary `backend`
followed by its `fallback` backends (same profile params, different provider). Each
`chat`/`chat_stream` call tries clients in order, advancing to the next only on a
transport error or non-2xx status; the first client that responds (even with empty
text) ends the chain. This lets a local profile fail over to a cloud provider — e.g.
`chat` runs on LM Studio but falls back to Anthropic via the gateway when LM Studio
is down. Streaming falls over too: failure happens on connect (before any token is
sent), so the user never sees a half-stream from a dead backend.

Each room resolves to a profile via `ReloadableState::llm_for_room`:
1. a runtime `/model` override (`room_profiles`), else
2. the room's `profile` config field, else
3. `chat`.

If the resolved profile has no built client, it falls back to `chat`. The resolved
profile + model are shown in `/status`.

Runtime `/model` overrides are persisted to `roger_session/room_profiles.json`
(`RoomProfileStore`) and reloaded on startup; overrides for profiles that don't
build on this host are dropped on load and on reload.

## Config hot-reload

Reloadable config lives behind `Arc<RwLock<ReloadableState>>` (in `matrix/handler.rs`),
shared by every event handler via the cloned `BotCtx`. `ReloadableState` holds the
LLM client, model name, global system prompt, and per-room configs.

A `SIGHUP` listener task (`reload_on_sighup` in `main.rs`) re-reads `config/` and
swaps the state in place. Reload is fail-safe: a bad config logs a warning and the
running config is kept. Handlers clone what they need out of the lock and release it
before any LLM/network call, so reloads never block in-flight requests.

Fixed for the process lifetime (restart required): Matrix credentials, homeserver,
room allowlist, the logging setup, and the speaches client.

## Logging

`init_logging` builds a layered `tracing` subscriber:
- **stderr** — human-readable, captured by journald under systemd.
- **file** — JSON lines, daily-rotated via `tracing-appender` into `ROGER_LOG_DIR`
  (default `roger_session/logs/`).

A single `EnvFilter` (`RUST_LOG`) gates both sinks. The non-blocking writer's
`WorkerGuard` is held in `main` so buffered logs flush at shutdown.

## Metrics

`Metrics` (`src/metrics.rs`) holds lock-free process-lifetime counters: total
responses, errors, and cumulative latency (for an average). Each completed response
calls `metrics.record(latency_ms, ok)` and emits a structured log line
(`responded` with `room`, `profile`, `model`, `latency_ms`, `ok`) — so the JSON log
sink doubles as a metrics scrape source. Live totals are shown in `/status`.
Counters reset on restart.

## Backend kinds

- `open-ai` — standard OpenAI-compatible REST API (LM Studio, Ollama, LiteLLM)
- `claude-code` — reserved for spawning `claude -p` subprocess (not yet implemented)
- `open-code` — reserved for `opencode run` subprocess (future)

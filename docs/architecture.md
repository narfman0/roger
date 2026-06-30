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
2. Stream the LLM response (`ProfileLlm::chat_stream`): SSE deltas are accumulated
   and pushed over an mpsc channel. The handler flushes (posts the first message,
   then edits it in place via `m.replace`) when a **sentence boundary** appears OR
   `MAX_FLUSH_WAIT_MS` (1s) has passed since the last flush — whichever comes
   first — but never more often than `MIN_FLUSH_GAP_MS` (250ms). The flush decision
   is `should_flush()`; sentence boundaries are found by `last_sentence_end()`.
   There is no placeholder; the typing indicator covers the wait before the first
   flush.
3. Final render: edit the message with the complete reply (if it changed), or — for
   a short response that never triggered a flush — send it as a single fresh
   message. `room.typing_notice(false)`.

If the stream errors or yields no content (e.g. a backend without SSE support),
the handler falls back to a single non-streaming `chat()` call. Only the `content`
field is surfaced — reasoning-model `reasoning` deltas are ignored. Slash commands
reply directly with a single message (no streaming, no placeholder).

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

A profile's chain is built from named backends, dispatched on `kind`
(`Config::build_client` → `llm::Backend`):

- `open-ai` — OpenAI-compatible REST (LM Studio, Ollama, LiteLLM). `Backend::Http`.
- `claude-code` — spawns the `claude` CLI as an agentic subprocess per turn
  (`src/subprocess.rs`, `Backend::Subprocess`). Implemented.
- `open-code` — spawns `opencode run --format json` (same module). Implemented.
  opencode is self-configured (provider + baseURL in its own config), so the
  gateway env vars don't apply and the model is `provider/model`. Its `--format
  json` is not token-streamed — the full reply arrives in one `text` event, so the
  message paints once at the end.

### Subprocess backends

A subprocess backend inverts the HTTP model: the CLI owns its own agentic tool
loop (file edits, bash, web), so roger's `ToolExecutor` is bypassed; history is
passed statelessly each turn (rendered into the prompt — system prompt via
`--append-system-prompt`, no `--session-id`), so the *files in the working
directory* are the persistent state. claude is run with `--print --output-format
stream-json --verbose --include-partial-messages --permission-mode <mode>
--model <m>`; `content_block_delta`/`text_delta` events feed the same accumulated
-text channel as the HTTP streamer, and the terminal `result` event is the
authoritative final string. Auth maps the gateway via `ANTHROPIC_BASE_URL` +
`ANTHROPIC_AUTH_TOKEN` (the Anthropic key never leaves `srv`). Lifecycle: a
process-wide semaphore (`max_concurrent_children`), per-line idle timeout,
wall-clock absolute ceiling, optional `--max-budget-usd`/`--max-turns`, and
`kill_on_drop` + `process_group(0)` + `killpg` for whole-tree kill.

### Working directory selection

The agent's cwd is chosen by the LLM, not roger: a `set_workdir(project)` tool
(in `src/tools.rs`) resolves a name against the `[projects]` map (name → path) and
records it for the room in `RoomWorkdirStore` (`~/.roger/room_workdirs.json`). Per
turn the handler resolves the workdir as **room selection → `comms.default_workdir`**
and passes it to the subprocess via a `WORKDIR` task-local (the room id rides a
`ROOM_ID` task-local so `set_workdir` knows its target); task-locals avoid
threading these through the whole chat-call chain. Roger keeps no scratch space and
leaves no artifacts — the workdir is an external, real project (which may be roger's
own repo). Because roger's own state lives in `~/.roger`, pointing an agent at the
repo doesn't expose its Matrix token/crypto store.

## Orchestrator: comms modes

The whole response pipeline for a turn (produce → stream → fallback → metrics →
history append → final render) runs as **one self-contained task**
(`run_response_job`), so it is correct whether the handler awaits or detaches it.
Each profile has a `comms` mode (`ReloadableState::comms_mode_for_profile`):

- `sync` — handler awaits the task (today's UX: typing indicator, streamed edits).
- `async` — the task posts a "🛠️ Working…" anchor, the handler registers it and
  returns; it streams edits into that anchor while the user keeps chatting.
- `auto` — `select!` the task against `sleep(sync_budget_ms)`; if the budget fires
  first, post a "still working" note and let the task finish in the background.

Background jobs live in a `Workers` registry (`src/workers.rs`) in `BotCtx`: a
job id → handle map with abort handles. It powers `/status` (active count),
`/jobs` (list), `/cancel <id>` (abort → kill the subprocess tree), a soft cap
warning (`soft_worker_cap`), and per-room serialization of agentic jobs (one
subprocess per room workdir at a time). Flush cadence comes from the reloadable
`CommsConfig::edit_debounce_ms`.

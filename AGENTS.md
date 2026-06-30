# AGENTS.md

Rust Matrix bot that responds to messages in allowlisted rooms using configurable LLM backends.

## Workflow

Commit directly on `master` and keep going — no feature branches or PRs unless
asked. Build + test before each commit. Push when a unit of work is done.

## Key source files

| File | Purpose |
|------|---------|
| `src/main.rs` | Entry point, wires all components together |
| `src/config.rs` | Config loading: profiles.toml + backends.<HOST_ROLE>.toml + .env |
| `src/llm.rs` | LLM client, OpenAI-compatible chat completions |
| `src/history.rs` | Per-room conversation history, JSON-backed on disk |
| `src/audio.rs` | Speaches/Whisper audio transcription client |
| `src/matrix/client.rs` | Matrix client build + session persistence |
| `src/matrix/handler.rs` | Event handlers: invite, message, ack+edit flow |

## Build and test

```bash
cargo build
cargo test
```

Requires: Rust 2021 edition, matrix-sdk 0.8, tokio.

## Running locally

Copy `.env.example` (or see README) and `config/backends.example.toml` to `config/backends.local.toml`. Then:

```bash
RUST_LOG=roger=info ./target/debug/roger
```

State is stored in `roger_session/`: SQLite crypto store, `session.json` (Matrix tokens), `history/` (per-room JSON), `logs/` (daily-rotated JSON logs).

## Hot-reload

`SIGHUP` reloads `config/` live (LLM client, system prompt, per-room settings) via
`Arc<RwLock<ReloadableState>>` — no Matrix re-login. `kill -HUP <pid>` or
`systemctl --user reload roger`. Credentials, homeserver, and room allowlist need a
restart. See `docs/architecture.md` → Config hot-reload.

## Logging

`init_logging` (`src/main.rs`): human-readable to stderr + JSON daily-rotated to
`ROGER_LOG_DIR` (default `roger_session/logs/`). `RUST_LOG` gates both.

## Config

- `config/profiles.toml` — committed, defines LLM profiles and routing
- `config/backends.<HOST_ROLE>.toml` — **gitignored**, backend URLs + api_key_env names
- `.env` — **gitignored**, Matrix credentials + GATEWAY_VKEY

Never commit `.env` or `backends.*.toml` (except `backends.example.toml`).

## Message / streaming flow

1. Typing indicator sent (no placeholder message)
2. Response is streamed; the first message is posted only once
   `FIRST_PAINT_MIN_CHARS` have buffered, then edited in place via `m.replace` as
   more arrives (debounced). Short replies are sent as a single message.
3. See `docs/architecture.md` → Response UX for the full flow (fallback, slash
   commands, token budgeting).

## Adding a new backend kind

Add a variant to `BackendKind` in `src/config.rs` (kebab-case serde name), then handle it in the dispatch logic.

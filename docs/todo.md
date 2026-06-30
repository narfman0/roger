# Phase 1

## Goals

Get roger to a daily-driveable state: reliable responses with memory, good UX, stable process.

## Tasks

| # | Task | Status |
|---|------|--------|
| Bug | Message editing — send ack then edit in-place via `m.replace` | ✅ Done |
| 1 | Per-room conversation history, persisted to disk | ✅ Done |
| 2 | System prompt / persona for roger | ✅ Done |
| 3 | Slash commands: `/help`, `/clear`, `/status` | ✅ Done |
| 4 | Graceful LLM error recovery (edit ack with error text) | ✅ Done |
| 5 | Systemd units: roger + lmstudio-server on `ai` machine | ✅ Done |

## Completed

### Bug: Message editing
Roger now sends "Working on it…" immediately on receiving a message, then edits that message in-place with the real LLM response via Matrix's `m.replace` relation. No more silent waits or accumulating extra messages.

### Conversation history
Per-room JSON history in `roger_session/history/`. Last 20 messages passed to the LLM as context on every request. Persists across restarts. User and assistant messages both recorded. History is room-scoped (isolated per room).

### Error recovery
LLM errors are edited into the ack message ("Sorry, I hit an error: …") rather than leaving "Working on it…" hanging.

### System prompt (#2)
`config/system_prompt.txt` loaded at startup with `{date}` injected. Passed as the first `system` role message in every LLM call. Per-room override planned for a future phase.

### Slash commands (#3)
Commands are intercepted before the LLM — handled inline and edited into the ack message:
- `/help` — lists available commands
- `/clear` — wipes history for the current room
- `/status` — uptime, history count for the room
- Unknown commands get a helpful error pointing to `/help`

### Systemd units (#5)
`deploy/lmstudio-server.service` and `deploy/roger.service` created with install instructions in `deploy/README.md`. Run `systemctl --user enable/start` after copying to `~/.config/systemd/user/`.

---

# Phase 2

## Goals

Operability: change behavior without restarts, observe what roger is doing, tailor it per room.

## Tasks

| # | Task | Status |
|---|------|--------|
| 1 | Config hot-reload (`SIGHUP`) | ✅ Done |
| 2 | `/status` model name display | ✅ Done |
| 3 | Per-room system prompt override | ✅ Done |
| 4 | Structured logging with rotation | ✅ Done |

## Completed

### Config hot-reload (#1)
Roger installs a `SIGHUP` handler (`reload_on_sighup` in `main.rs`). On signal it
re-reads `config/` and atomically swaps the reloadable state behind an
`Arc<RwLock<ReloadableState>>`: the LLM client (model/temperature/max_tokens),
the global system prompt, and all per-room settings. The Matrix session, room
allowlist, and credentials are fixed for the process lifetime — those still need a
restart. A failed reload (bad TOML, missing backend) logs a warning and keeps the
current config rather than crashing. Trigger with `kill -HUP <pid>` or
`systemctl --user reload roger` (`ExecReload` added to the unit).

The read lock is never held across the LLM call — handlers snapshot the client +
prompt and drop the guard first, so a reload never blocks in-flight responses.

### `/status` model name (#2)
`/status` now reports the active chat model alongside uptime and per-room history
count. Reads the live `model_name` from reloadable state, so it reflects a hot-reload.

### Per-room system prompt (#3)
`RoomConfig` gained an optional `system_prompt` field. When set, it replaces the
global prompt for that room; `{date}` is injected the same way. Falls back to the
global prompt when unset. See the coding-room example in `config/profiles.toml`.

### Structured logging with rotation (#4)
`init_logging` layers two sinks: human-readable to stderr (journald-friendly) and
JSON with daily rotation to `ROGER_LOG_DIR` (default `roger_session/logs/`, file
`roger.log.YYYY-MM-DD`) via `tracing-appender`. The non-blocking writer guard is
held for the process lifetime so logs flush on shutdown. `RUST_LOG` controls both
sinks.

## Phase 3 (next)

Roadmap candidates — not yet started:
- Profile routing: pick LLM profile per task (`code`/`reason`/`research`) instead of always `chat`
- Streaming responses: edit the ack incrementally as tokens arrive (debounced)
- `/model <profile>` command to switch a room's backend at runtime
- Token/context budgeting: summarize or trim history beyond a token threshold
- Multi-backend dispatch: implement the reserved `claude-code` / `open-code` subprocess kinds
- Metrics: request counts, latency, error rate (Prometheus endpoint or structured log scrape)

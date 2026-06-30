# Deploy

Systemd user unit for running Roger on the `ai` machine.

## Install

```bash
# Build release binary first
cd ~/.openclaw/workspace/roger
cargo build --release

# Copy unit
mkdir -p ~/.config/systemd/user
cp deploy/roger.service ~/.config/systemd/user/

# Enable and start
systemctl --user daemon-reload
systemctl --user enable roger
systemctl --user start roger

# Check status
systemctl --user status roger
journalctl --user -u roger -f
```

## State directory

Roger keeps all mutable state (Matrix crypto store, `session.json` token,
`history/`, `logs/`, `room_profiles.json`) in `ROGER_STATE_DIR`, default `~/.roger`
— **not** in the repo. The working directory only needs to contain `config/`.

Migrating from the old in-repo `roger_session/` layout (do this once, while roger is
stopped, to preserve the crypto store + Matrix token and avoid a re-login that would
churn the device ID):

```bash
systemctl --user stop roger
mv ~/.openclaw/workspace/roger/roger_session ~/.roger
systemctl --user start roger
```

## Lingering (survive logout)

```bash
loginctl enable-linger $USER
```

This keeps the user session alive so services run even when you're not logged in.

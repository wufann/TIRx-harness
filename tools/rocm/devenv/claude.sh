#!/bin/bash
# Runs INSIDE the dev container: start Claude Code from the host account's installation,
# with its login and settings (dev_up.sh mounts the host home when it has one).
# Usage (in a prepared run): cd <run_dir>/worktree && . .venv/bin/activate && claude.sh
if [ -z "$HOST_HOME" ] || [ ! -x "$HOST_HOME/.local/bin/claude" ]; then
  echo "no host Claude Code in this container; recreate it with DEV_RECREATE=1 dev_up.sh" >&2
  exit 3
fi
export HOME=$HOST_HOME
exec "$HOST_HOME/.local/bin/claude" "$@"

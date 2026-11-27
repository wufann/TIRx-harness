#!/bin/bash
# Runs INSIDE the dev container: start Claude Code from the host account's installation,
# with its login and settings (dev_up.sh mounts the host home when it has one).
# `docker exec` shells do not read the host ~/.bashrc, so the Claude Code / API gateway
# variables exported there (AMD_LLM_GATEWAY_KEY, ANTHROPIC_*, CLAUDE_CODE_*) are taken from
# it explicitly; CLAUDE_ENV_FILE selects another file with such `export` lines.
# Usage (in a prepared run): cd <run_dir>/worktree && . .venv/bin/activate && claude.sh
if [ -z "$HOST_HOME" ] || [ ! -x "$HOST_HOME/.local/bin/claude" ]; then
  echo "no host Claude Code in this container; recreate it with DEV_RECREATE=1 dev_up.sh" >&2
  exit 3
fi
export HOME=$HOST_HOME
env_file=${CLAUDE_ENV_FILE:-$HOST_HOME/.bashrc}
if [ -f "$env_file" ]; then
  eval "$(grep -E '^[[:space:]]*export[[:space:]]+(AMD_LLM_GATEWAY_KEY|ANTHROPIC_[A-Z_]+|CLAUDE_CODE_[A-Z_]+)(=|[[:space:]]*$)' "$env_file")"
fi
[ -n "$CLAUDE_CODE_TMPDIR" ] && mkdir -p "$CLAUDE_CODE_TMPDIR"
# Optimization runs use the strongest model by default (the host ~/.bashrc may pin an
# older one for interactive work); TIRX_CLAUDE_MODEL overrides it.
model=${TIRX_CLAUDE_MODEL:-claude-opus-5.5}
export ANTHROPIC_MODEL=$model ANTHROPIC_DEFAULT_OPUS_MODEL=$model CLAUDE_CODE_SUBAGENT_MODEL=$model
exec "$HOST_HOME/.local/bin/claude" "$@"

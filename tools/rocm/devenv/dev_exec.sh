#!/bin/bash
# Run a command inside the dev container with the venv active.
# Usage: crs login node: dev_exec.sh <jobid> '<cmd>'; compute node / conductor machine: dev_exec.sh [-] '<cmd>'
set -o pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
reexec_on_node "$(readlink -f "$0")" "$@"
is_job_arg "$1" && shift
if ! docker inspect -f '{{.State.Running}}' "$CTR" 2>/dev/null | grep -q true; then
  echo "container $CTR not running on $(hostname); run dev_up.sh first" >&2; exit 3
fi
# Attach stdin only for interactive use: a piped caller (e.g. `ssh host bash -s < script`)
# would otherwise have the rest of its script swallowed by docker exec.
tty_flag=; [ -t 0 ] && [ -t 1 ] && tty_flag=-it
# DEV_ROOT=1 runs as root, for tools that want to write inside the image (e.g. aiter's JIT
# dir under /sgl-workspace; as a normal user aiter copies ~4 GB of it into $HOME instead).
user_flag=; [ "${DEV_ROOT:-0}" = 1 ] && user_flag="-u 0"
exec docker exec $tty_flag $user_flag -w "$CW" "$CTR" bash -c "source $DEVENV/container_env.sh; $*"

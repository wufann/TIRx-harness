#!/bin/bash
# Print the state needed to resume work: repos, venv, build, container, logs.
# Usage: status.sh [jobid]   (job id only needed on the crs login node)
source "$(dirname "$(readlink -f "$0")")/env.sh"
echo "== repos"
for r in $REPOS; do
  d=$CW/$r
  printf '%-14s %s  %s\n' "$r" "$(git -C "$d" rev-parse --abbrev-ref HEAD)" "$(git -C "$d" log -1 --format='%h %cd %s' --date=format:'%m-%d %H:%M' | cut -c1-80)"
  git -C "$d" status --porcelain | head -10 | sed 's/^/    /'
  ahead=$(git -C "$d" rev-list --count '@{u}..HEAD' 2>/dev/null) && [ "$ahead" != 0 ] && echo "    ($ahead commit(s) not pushed)"
done
echo "== venv"
if [ -f "$VENV/pyvenv.cfg" ]; then echo "$VENV (req stamp $(cut -c1-8 "$VENV/.req.sha" 2>/dev/null))"; else echo "missing"; fi
echo "== build"
b=$CW/tvm-tirx/build
if [ -d "$b" ]; then echo "$b  last modified $(stat -c %y "$b" | cut -c1-16)"; ls -t "$b"/*.so 2>/dev/null | head -3; else echo "no build dir yet"; fi
echo "== logs (latest)"
ls -t "$LOGDIR" 2>/dev/null | head -5 | sed "s|^|$LOGDIR/|"
echo "== jobs"
squeue -u "$(id -un)" -o "%i %j %T %L %N" 2>/dev/null
echo "== container"
if on_compute_node; then
  docker ps -a --filter name=$CTR --format '{{.Names}} {{.Status}}'
elif [ -n "$1" ]; then
  spur exec "$1" bash -c "docker ps -a --filter name=$CTR --format '{{.Names}} {{.Status}}'" 2>&1
fi

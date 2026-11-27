#!/bin/bash
# Update the three checkouts under $CW (fast-forward only) and show where each one is.
# Code reaches the cluster only through git: commit + push elsewhere, then run this.
# Usage: pull.sh [branch]   (default: each checkout's current branch)
set -o pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
rc=0
for r in $REPOS; do
  d=$CW/$r
  if [ -n "$1" ] && [ "$(git -C "$d" rev-parse --abbrev-ref HEAD)" != "$1" ]; then
    git -C "$d" checkout -q "$1" || { rc=1; continue; }
  fi
  if ! git -C "$d" pull -q --ff-only; then
    echo "!! $r: pull failed (local changes or diverged branch)" >&2; rc=1
  fi
  printf '%-14s %-12s %s\n' "$r" "$(git -C "$d" rev-parse --abbrev-ref HEAD)" "$(git -C "$d" log -1 --format='%h %s' | cut -c1-70)"
  git -C "$d" status --porcelain --untracked-files=no | sed 's/^/    /'
done
exit $rc

#!/bin/bash
# Bring up (or reuse) the dev container on the job's compute node.
# Usage: crs login node: dev_up.sh <jobid>; compute node / conductor machine: dev_up.sh [-]
set -eo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
reexec_on_node "$(readlink -f "$0")" "$@"
mkdir -p "$LOGDIR" "$CW/.cache" "$CW/.home"
log() { echo "[dev_up $(date +%H:%M:%S) $(hostname)] $*"; }

# 1. Image, pinned by digest. Pull time is logged so we learn the cost of a new node.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  log "image not on node, pulling $IMAGE_PULL"
  t0=$(date +%s)
  docker pull "$IMAGE_PULL"
  echo "$(date -Is) $(hostname) pull_seconds=$(( $(date +%s) - t0 ))" >> "$LOGDIR/image_pull.log"
fi
log "image ok: $(docker image inspect "$IMAGE" --format '{{.Id}}' | cut -c1-19)"

# 2. Container: reuse if running, restart if stopped, else create.
state=$(docker inspect -f '{{.State.Status}}' "$CTR" 2>/dev/null || true)
if [ "$state" = running ]; then
  log "container $CTR already running"
elif [ -n "$state" ]; then
  log "container $CTR is $state, starting"
  docker start "$CTR" >/dev/null
else
  render_gid=$(getent group render | cut -d: -f3)
  video_gid=$(getent group video | cut -d: -f3)
  log "creating container $CTR (uid $(id -u), render gid $render_gid, video gid $video_gid)"
  docker run -d --name "$CTR" \
    --network host --ipc host --shm-size 64g \
    --device /dev/kfd --device /dev/dri \
    --group-add "$video_gid" --group-add "$render_gid" \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    -u "$(id -u):$(id -g)" \
    -e HOME="$CW/.home" -e USER="$(id -un)" \
    -v "$CW:$CW" -w "$CW" \
    --entrypoint sleep "$IMAGE" infinity >/dev/null
fi

# aiter (benchmark baselines) writes JIT modules / tuned-config locks into the image;
# as a normal user it would copy its ~4 GB JIT directory into $HOME instead.
docker exec -u 0 "$CTR" bash -c 'chmod -R a+rwX /sgl-workspace/aiter/aiter/jit 2>/dev/null;
  mkdir -p /tmp/aiter_configs && chmod -R a+rwX /tmp/aiter_configs' || true

# 3. Python env inside the container (idempotent).
docker exec "$CTR" bash "$DEVENV/setup_venv.sh"
log "ready: dev_exec.sh <jobid> '<cmd>'   or on the node: docker exec -it $CTR bash"

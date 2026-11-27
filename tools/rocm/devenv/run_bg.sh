#!/bin/bash
# Run a container-side script detached (survives the ssh/spur session); log to $CW/logs.
# Usage: run_bg.sh <jobid|-> <name> <script> [args...]   ("-" off the crs login node)
#   script: a file in this directory (e.g. build_tvm_tirx.sh) or a path relative to $CW
source "$(dirname "$(readlink -f "$0")")/env.sh"
reexec_on_node "$(readlink -f "$0")" "$@"
is_job_arg "$1" && shift
name=$1; script=$2; shift 2
log=$LOGDIR/${name}_$(date +%Y%m%d_%H%M%S).log
[ -f "$DEVENV/$script" ] && script=$DEVENV/$script
docker exec -d -w "$CW" "$CTR" bash -c "bash $script $* > $log 2>&1; echo \"[run_bg] exit \$?\" >> $log"
echo "started $name on $(hostname); log: $log"

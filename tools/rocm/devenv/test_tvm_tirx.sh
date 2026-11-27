#!/bin/bash
# Runs INSIDE the dev container: pytest over tvm-tirx tests (default: ROCm-relevant set).
# Usage: test_tvm_tirx.sh [pytest args/paths...]    GPU via HIP_VISIBLE_DEVICES (default 7)
source "$(dirname "$(readlink -f "$0")")/container_env.sh"
cd "$CW/tvm-tirx"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-7}
if [ $# -eq 0 ]; then
  set -- tests/python/backend/rocm tests/python/codegen/gpu_common/test_device.py \
    tests/python/codegen/test_bool.py tests/python/codegen/test_common.py tests/python/target/test_target.py
fi
ts=$(date +%Y%m%d_%H%M%S)
echo "[test] $(date -Is) $(hostname) commit $(git rev-parse --short HEAD) HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
python -m pytest -q -rfEs -p no:cacheprovider --junitxml="$CW/logs/pytest_$ts.xml" "$@" 2>&1 | grep -v DEPRECATION

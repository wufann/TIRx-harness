#!/bin/bash
# Runs INSIDE the dev container: editable build of tvm-tirx with LLVM (AMDGPU) + ROCm.
# Incremental: the CMake build dir is tvm-tirx/build on /home, reused across nodes.
# Usage: build_tvm_tirx.sh [extra pip args]   (log: $CW/logs/build_tvm_tirx_<ts>.log via run_bg.sh)
set -eo pipefail
source "$(dirname "$(readlink -f "$0")")/container_env.sh"
cd "$CW/tvm-tirx"
export CMAKE_BUILD_PARALLEL_LEVEL=${JOBS:-96}
export CMAKE_ARGS="-DTVM_TIRX_USE_LLVM=$LLVM_CONFIG -DTVM_TIRX_USE_ROCM=ON -DTVM_TIRX_USE_CUDA=OFF ${EXTRA_CMAKE_ARGS}"
echo "[build] $(date -Is) $(hostname) commit $(git rev-parse --short HEAD) CMAKE_ARGS=$CMAKE_ARGS jobs=$CMAKE_BUILD_PARALLEL_LEVEL"
t0=$(date +%s)
python -m pip install --no-build-isolation --no-deps -v -e . -Cbuild.tool-args=-k0 "$@"
echo "[build] done in $(( $(date +%s) - t0 ))s"
python "$DEVENV/smoke_tvm_tirx.py"

#!/bin/bash
# Runs INSIDE the dev container. Creates/updates $VENV (idempotent).
# The venv reuses the image's torch/ROCm (chained /opt/venv site-packages), but shadows
# the image's old tvm-ffi (~v0.1.8, needed by sglang's tilelang/xgrammar) with
# the version tvm-tirx requires.
set -eo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
source "$DEVENV/container_env.sh" --no-venv
REQ="$DEVENV/requirements-venv.txt"
STAMP="$VENV/.req.sha"
if [ ! -x "$VENV/bin/python" ]; then
  echo "[setup_venv] creating $VENV from /opt/venv/bin/python"
  /opt/venv/bin/python -m venv "$VENV"
fi
# /opt/venv is itself a venv, so `venv --system-site-packages` would expose the
# base interpreter's site-packages, not /opt/venv's (where torch lives). Chain it
# explicitly instead: addsitedir appends it after our own site-packages (ours win)
# and also processes its .pth files (editable aiter/sglang installs).
SP=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "import site; site.addsitedir('/opt/venv/lib/python3.12/site-packages')" > "$SP/zz_opt_venv.pth"
# TIRx-kernels from the checkout (the effect of `pip install -e`, without a build step).
echo "$CW/TIRx-kernels" > "$SP/tirx_kernels_dev.pth"
want=$(sha1sum "$REQ" | cut -c1-40)
if [ "$(cat "$STAMP" 2>/dev/null)" != "$want" ]; then
  echo "[setup_venv] installing $REQ"
  "$VENV/bin/python" -m pip install -q -r "$REQ"
  echo "$want" > "$STAMP"
fi
"$VENV/bin/python" - <<'PY'
import sys, tvm_ffi, torch
assert tvm_ffi.__file__.startswith(sys.prefix), f"tvm_ffi not shadowed: {tvm_ffi.__file__}"
print(f"[setup_venv] tvm_ffi {tvm_ffi.__version__} ({tvm_ffi.__file__})")
print(f"[setup_venv] torch {torch.__version__} hip {torch.version.hip} gpus {torch.cuda.device_count()}")
import tirx_kernels
print(f"[setup_venv] tirx_kernels {tirx_kernels.__file__}")
PY

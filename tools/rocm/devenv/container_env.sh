# Sourced INSIDE the dev container. Pass --no-venv to skip venv activation.
DEVENV=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
CW=${CW:-$(readlink -f "$DEVENV/../../../..")}
export HOME=$CW/.home
export XDG_CACHE_HOME=$CW/.cache
export PIP_CACHE_DIR=$CW/.cache/pip
export UV_CACHE_DIR=$CW/.cache/uv
export ROCM_PATH=/opt/rocm
export LLVM_CONFIG=/opt/rocm/llvm/bin/llvm-config
if [ "$1" != --no-venv ] && [ -f "$CW/venv/bin/activate" ]; then
  source "$CW/venv/bin/activate"
fi

# Shared settings for the TIRx ROCm dev workflow (sourced by the other scripts).
# Everything stateful lives under $CW on /home; the container itself is disposable.
# $CW is the directory holding the three checkouts side by side (tvm-tirx, TIRx-kernels,
# TIRx-harness); by default derived from this file's location, override with CW=...
#
# Host profiles (picked by hostname):
#   crs login node   crsuse2-slog-*   scripts re-run themselves on the job's node via `spur exec`
#   crs compute node crsuse2-m2m-*    docker on the node
#   gbt350-*                          conductor machine: docker directly, no scheduler
DEVENV=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
CW=${CW:-$(readlink -f "$DEVENV/../../../..")}
CTR=${CTR:-${USER:-$(id -un)}-tirx-dev}
VENV=$CW/venv
LOGDIR=$CW/logs
REPOS="tvm-tirx TIRx-kernels TIRx-harness"

case "$(hostname)" in
  gbt350-*)
    HOST_PROFILE=local
    # Same toolchain as the crs image (AMD LLVM 23.0git 8f497e0, ROCm 10.0.0, torch 2.11);
    # already on the machine, whose disk cannot fit the 57.8 GB crs image.
    IMAGE_TAG=rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261112
    IMAGE=sha256:042c23c4b24c7d53924709fc0643679d0682810f5544dec6ce45aa569f3399f5
    IMAGE_PULL=$IMAGE_TAG  # an image id cannot be pulled; the tag may have moved since
    ;;
  crsuse2-slog-*)
    HOST_PROFILE=crs-login
    ;;
  *)
    HOST_PROFILE=crs-node
    ;;
esac
if [ "$HOST_PROFILE" != local ]; then
  IMAGE_TAG=lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261124
  IMAGE=lmsysorg/sglang-rocm@sha256:7908705b0da24c2b7b3e5709818f280713bbc6dd2bc3a7e53da80f1829ea6f92
  IMAGE_PULL=$IMAGE
fi

on_compute_node() { [ "$HOST_PROFILE" != crs-login ]; }

# First argument of the user-facing scripts: a crs job id, or "-" where there is no
# scheduler (on a compute node or the conductor machine).
is_job_arg() { [[ "$1" == "-" || "$1" =~ ^[0-9]+$ ]]; }

# Re-run this script on the job's compute node when invoked from the crs login node.
# Usage at the top of a script: reexec_on_node "$0" "$@"  (first arg = job id)
reexec_on_node() {
  local self=$1; shift
  if on_compute_node; then return 0; fi
  local job=$1; shift || true
  if [ -z "$job" ]; then echo "usage: $(basename "$self") <jobid> ..." >&2; exit 2; fi
  # forward the knobs the node-side scripts read (spur exec does not pass the environment)
  exec spur exec "$job" env DEV_ROOT="${DEV_ROOT:-0}" DEV_RECREATE="${DEV_RECREATE:-0}" CW="$CW" \
    bash "$self" "$job" "$@"
}

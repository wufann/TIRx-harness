# ROCm (gfx950) tools

Dev environment and analysis tools for TIRx kernels on MI350X / MI355X. The kernels themselves
live in TIRx-kernels (`tirx_kernels/rocm/`), the compiler changes in tvm-tirx; all three use the
`rocm-gfx950` branch.

## Layout on the machine

Everything stateful lives in one directory on a shared filesystem, `$CW`; the container is
disposable and recreated on whatever node the scheduler hands out:

```
$CW/                    e.g. /home/<user>/compiler_work
  tvm-tirx/             checkout + its CMake build dir (incremental across nodes)
  TIRx-kernels/         checkout (on the venv's path via a .pth file)
  TIRx-harness/         checkout; these scripts run from tools/rocm/devenv/
  venv/                 python venv chained to the image's /opt/venv (torch, ROCm)
  logs/ artifacts/ .cache/ .home/
```

`$CW` defaults to the directory above the TIRx-harness checkout; override with `CW=...`.

## devenv/

| Script | Where | What |
| --- | --- | --- |
| `env.sh` | sourced | paths, container name, image (pinned by digest), host profiles |
| `pull.sh [branch]` | host | `git pull --ff-only` the three checkouts; code reaches the cluster only through git |
| `dev_up.sh <jobid\|->` | host | pull image if missing, create/restart the container (`DEV_RECREATE=1`: recreate), `setup_venv.sh` |
| `dev_exec.sh <jobid\|-> '<cmd>'` | host | run a command in the container with the venv active (`DEV_ROOT=1`: as root) |
| `run_bg.sh <jobid\|-> <name> <script> [args]` | host | detached run, log in `$CW/logs/<name>_<ts>.log` |
| `status.sh [jobid]` | host | repos, venv, build, latest logs, jobs, container |
| `setup_venv.sh` | container | create/update the venv (idempotent; reinstalls when `requirements-venv.txt` changes) |
| `build_tvm_tirx.sh` | container | editable tvm-tirx build with LLVM (AMDGPU) + ROCm, then `smoke_tvm_tirx.py` |
| `test_tvm_tirx.sh [pytest args]` | container | ROCm-relevant tvm-tirx tests (`HIP_VISIBLE_DEVICES`, default 7) |
| `inspect_image.sh` | image | read-only toolchain inspection of a candidate image |
| `claude.sh [args]` | container | Claude Code from the host account's installation and login (host home mounted by `dev_up.sh`) |

On the crs login node the first argument is the job id and the host-side scripts re-run
themselves on the job's compute node via `spur exec`; on a compute node or a machine without a
scheduler pass `-`.

Daily start on a new node:

```bash
D=$CW/TIRx-harness/tools/rocm/devenv
$D/pull.sh                                   # after pushing from the workstation
$D/dev_up.sh <jobid>                         # ~1 s if the image is cached on the node
$D/run_bg.sh <jobid> build build_tvm_tirx.sh # only when tvm-tirx C++ changed
$D/dev_exec.sh <jobid> 'cd TIRx-kernels && HIP_VISIBLE_DEVICES=7 python -m pytest -o addopts="" tests/rocm'
```

## analysis/

| Tool | What |
| --- | --- |
| `isa_stats.py` | instruction histogram, MFMA / memory / waitcnt forms and hottest loop of `.s` / code objects (`--symbols` for bundles with zero-size symbols) |
| `prof_gemm.py` | rocprofv3 PMC passes over our kernel, an exported hsaco or rocBLAS, metrics side by side |
| `att_summary.py` | rocprofv3 ATT per-instruction stats grouped by instruction class |
| `att_timeline.py` | MFMA issue cadence of one wave from an ATT timeline |
| `export_hsaco.py` | BF16 GEMM variant -> code object + kernel name, for `hsaco_gemm` |
| `hsaco_gemm.cpp` / `rocblas_gemm.cpp` | C++ workloads for rocprofv3 (built by `build.sh`) |
| `hipblaslt_kernels.py` | which hipBLASLt kernels torch.matmul launches |
| `aiter_a4w4_bench.py` | aiter `gemm_a4w4` MXFP4 baseline (run with `DEV_ROOT=1`) |

rocprofv3 hangs on any Python process that uses HIP in the sglang ROCm 10 images (it cannot be
stopped with SIGTERM; use `timeout -s KILL`), hence the C++ workloads and the torch-free `run`
mode of `prof_gemm.py`.

## Optimization runs (evolution)

Prepare a ROCm run inside the container, then start the agent in it:

```bash
$D/dev_exec.sh <jobid> 'cd TIRx-harness && python -m evolution.cli init --task mxfp4_gemm_qwen38_shared_gate_up_m8192 --name <run>'
# interactive shell on the node (srun --jobid <jobid> --overlap --pty bash), then:
docker exec -it <user>-tirx-dev bash
cd $CW/TIRx-harness/kda_flow_runs/<run>/worktree && . .venv/bin/activate
$CW/TIRx-harness/tools/rocm/devenv/claude.sh
# in Claude Code: /goal Read ../PROMPT.md and beat the baseline while passing correctness.
```


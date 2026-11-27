"""aiter's MXFP4 GEMM (gemm_a4w4: tuned asm / CK / triton per shape), the MXFP4 baseline.

Run as root (DEV_ROOT=1 dev_exec.sh ...): as a normal user aiter copies its ~4 GB JIT
directory from /sgl-workspace into $HOME. Usage: python aiter_a4w4_bench.py M,N,K [M,N,K ...]
"""

import sys

import torch

sys.path.insert(0, "/sgl-workspace/aiter/op_tests")
torch.set_default_device("cuda")
import test_gemm_a4w4 as t  # noqa: E402

for shape in sys.argv[1:]:
    M, N, K = map(int, shape.split(","))
    r = t.test_gemm(torch.bfloat16, M, N, K)
    keys = [k for k in r if k.endswith(("TFLOPS", "us", "err")) or k == "configured kernel"]
    print(
        f"[aiter] {M}x{N}x{K}: "
        + ", ".join(f"{k}={r[k]:.4g}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in keys)
    )

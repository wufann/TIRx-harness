"""Which hipBLASLt kernels does torch.matmul (bf16, TN) launch?

Usage: python hipblaslt_kernels.py [OUT.txt]   (OUT: full kernel names, one per line)
"""

import torch
from torch.profiler import ProfilerActivity, profile

shapes = [(4096, 4096, 4096), (8192, 8192, 8192), (2048, 2048, 2048)]
for m, n, k in shapes:
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    bt = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ bt.T
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        a @ bt.T
        torch.cuda.synchronize()
    names = {e.name for e in prof.events() if e.device_type.name == "CUDA"}
    print(f"[hipblaslt] {m}x{n}x{k}: {sorted(names)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:  # write the full kernel names, one per line
        with open(sys.argv[1], "w") as f:
            for m, n, k in shapes:
                a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                bt = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
                a @ bt.T
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    a @ bt.T
                    torch.cuda.synchronize()
                for e in prof.events():
                    if e.device_type.name == "CUDA":
                        f.write(f"{m}x{n}x{k} {e.name}\n")

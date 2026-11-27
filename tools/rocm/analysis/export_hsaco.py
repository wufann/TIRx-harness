"""Export a BF16 GEMM variant (tirx_kernels.rocm.bf16_gemm) as a code object + launch info,
for the C++ profiling harness hsaco_gemm.cpp.

Usage: python export_hsaco.py VARIANT M N K OUT.hsaco
"""

import re
import subprocess
import sys

import tvm_tirx
from tirx_kernels.rocm.bf16_gemm import make_gemm_v1, target_for

variant, M, N, K, out = (
    sys.argv[1],
    int(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
    sys.argv[5],
)
f = tvm_tirx.compile(make_gemm_v1(M, N, K, variant), target=target_for(variant))
dev = f.mod.imports[0]
# the ROCm runtime module cannot WriteToFile: assemble the ISA text into a code object

# tvm-tirx emits the legacy "-hcc" environment, which the clang driver rejects
open(out + ".s", "w").write(
    dev.inspect_source("asm").replace("amdgcn-amd-amdhsa-hcc-", "amdgcn-amd-amdhsa--")
)
llvm = "/opt/rocm/llvm/bin"
subprocess.run(
    [
        f"{llvm}/clang",
        "-x",
        "assembler",
        "-target",
        "amdgcn-amd-amdhsa",
        "-mcpu=gfx950",
        "-c",
        out + ".s",
        "-o",
        out + ".o",
    ],
    check=True,
)
subprocess.run([f"{llvm}/ld.lld", "-shared", out + ".o", "-o", out], check=True)
ll = dev.inspect_source("hip")
m = re.search(r"define (?:dllexport )?amdgpu_kernel void @(\w+)\(([^)]*)\)", ll)
print("kernel:", m.group(1))
print("params:", m.group(2)[:300])

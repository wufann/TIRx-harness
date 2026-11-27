---
name: tirx-rocm-wiki
description: >
  Reference for writing gfx950 (AMD Instinct MI350X / MI355X, CDNA4) kernels in TIRx
  TVMScript with the T.amdgcn dialect: dialect API, MFMA operand layouts, memory and
  synchronization rules, what the LLVM AMDGPU backend does with explicit control,
  production kernel patterns (aiter, hipBLASLt, HipKittens) and the gfx950 GEMMs in
  tirx_kernels/rocm. Pair it with tirx-rocm-debug-kernel or tirx-rocm-profile-kernel
  for runnable diagnosis or measurement.
---

# TIRx ROCm Wiki

## Route operational work

Keep this skill for knowledge and source lookup.

- Invoke `$tirx-rocm-debug-kernel` for wrong results, hangs, GPU faults, or
  suspected waitcnt / barrier / LDS-sharing mistakes on a runnable kernel.
- Invoke `$tirx-rocm-profile-kernel` for timing, generated ISA inspection,
  hardware counters, or ATT instruction timelines.

## Sources, by question

| Question | Open |
|---|---|
| How do I write a kernel entry, allocate LDS / registers, launch threads? | [TVMScript on ROCm](references/tvmscript-rocm.md) |
| Which `T.amdgcn` call emits instruction X, and with which operand types? | [T.amdgcn dialect](references/tamdgcn.md) |
| MFMA lane layouts, LDS size and banks, waitcnt counters, AGPRs, buffer resources | [gfx950 facts](references/gfx950.md) |
| Will LLVM keep my waits / schedule / register class? Why did AGPR copies appear? | [LLVM AMDGPU backend](references/llvm-amdgpu.md) |
| How do production gfx950 GEMMs schedule their main loop? | [Kernel patterns](references/patterns.md) |
| Where is a working gfx950 GEMM to start from, and what did its variants teach? | [tirx_kernels/rocm](references/kernels.md) |
| Exact instruction semantics, cycle counts, encodings | CDNA4 ISA text (below) |

## CDNA4 ISA reference

`scripts/fetch_references.py` downloads AMD's *CDNA4 Instruction Set Architecture*
reference guide and extracts `references/manuals/cdna4_isa.txt` (about 21k lines;
grep it, e.g. `grep -n "V_MFMA_SCALE_F32_16X16X128" references/manuals/cdna4_isa.txt`).
Run setup normally fetches it; if the file is missing, run the script with network
access. Survey it broadly: an instruction or operand form the current kernel does not
use can suggest a different structure.

## Source lookup

- Dialect source: `python -c "import tvm_tirx.backend.rocm.amdgcn as m; print(m.__file__)"`.
- Kernel sources: `python -c "import tirx_kernels.rocm as m; print(m.__path__[0])"`.
- Read the dialect docstrings before using an entry; they state operand layouts and
  hazards that the compiler does not check.

---
name: tirx-rocm-profile-kernel
description: >
  Measure and explain the performance of runnable gfx950 (MI350X / MI355X) TIRx kernels:
  timing, generated-ISA inspection of the hot loop, register / LDS budgets, rocprofv3
  hardware counters and ATT instruction timelines. Use whenever performance on ROCm must
  be measured, explained, compared or improved.
---

# Profile a TIRx Kernel on gfx950

Measure before explaining. Reading the source or a guide is not performance evidence
when a runnable case exists. Pick an idle GPU first (`rocm-smi --showuse --showmemuse`)
and pass `HIP_VISIBLE_DEVICES`; wrap every GPU command in `timeout -s KILL <s>`.

## Tool routing

| Question | Action |
|---|---|
| Is it faster? | The task's benchmark command (authoritative; cold caches). For quick A/B inside one process use HIP events around many calls, but confirm with the benchmark |
| What did the compiler emit? Why is the loop slow? | [ISA inspection](#isa-inspection) — always, after every structural change |
| Occupancy, spills, LDS size | Kernel metadata at the end of the assembly |
| Stalls, LDS bank conflicts, MFMA utilization, cache traffic | [Hardware counters](#hardware-counters) |
| MFMA issue cadence, what fills each MFMA gap, barrier waits | [ATT timeline](#att-instruction-timeline) |

## ISA inspection

```python
f = tvm_tirx.compile(func, target=TARGET)
open("k.s", "w").write(f.mod.imports[0].inspect_source("asm"))
```

```bash
python tools/rocm/analysis/isa_stats.py k.s --loop-dump 400
```

`isa_stats.py` prints the instruction-family histogram, MFMA variants, memory forms,
waitcnt forms and the hottest loop (largest backward-branch range). Check in the main loop:

- MFMA count per iteration vs. the expected `tiles x substeps`;
- `v_accvgpr_read/write/mov` — any non-trivial count is a register-allocation storm;
- `s_waitcnt` count and forms (`vmcnt(0)` before every LDS read means LLVM could not
  separate your stages);
- VALU / SALU address math per DMA; scratch loads/stores (spills);
- order: are `ds_read`s and DMAs spread between MFMAs or bunched?

Metadata: `.vgpr_count`, `.agpr_count`, `.sgpr_count`, `.group_segment_fixed_size`
(LDS bytes), `.private_segment_fixed_size` (non-zero = spills). Waves per SIMD are
bounded by 512 / (vgpr + agpr, rounded to the allocation granule) and by LDS per CU.

## Hardware counters

rocprofv3 hangs on any Python process that uses HIP in the sglang ROCm 10 images, and
SIGTERM does not stop it. Profile a C++ launcher instead:

1. Export the kernel as a code object. `tools/rocm/analysis/export_hsaco.py` shows how:
   rewrite the assembly's `amdgcn-amd-amdhsa-hcc-` target to `amdgcn-amd-amdhsa--`,
   assemble with `/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa
   -mcpu=gfx950 -c`, link with `ld.lld -shared`. The kernel symbol and parameters are in
   `inspect_source("hip")` (LLVM IR).
2. Launch it from C++ with `hipModuleLoad` / `hipModuleLaunchKernel`
   (`tools/rocm/analysis/hsaco_gemm.cpp` is a template for a GEMM ABI; build with
   `hipcc -O2`).
3. Collect counters, at most ~6 per pass:

```bash
timeout -s KILL 300 rocprofv3 --pmc SQ_WAVE_CYCLES SQ_BUSY_CYCLES SQ_WAIT_INST_ANY SQ_INSTS_MFMA \
    SQ_VALU_MFMA_BUSY_CYCLES SQ_INSTS_LDS -d out -o run --output-format csv -- ./launcher ...
```

`rocprofv3 -L` lists gfx950 counters. Useful groups (see `prof_gemm.py` `PASSES`):
wave/busy/wait cycles; `SQ_VALU_MFMA_BUSY_CYCLES` and `SQ_INSTS_VALU_MFMA_MOPS_*`;
`SQ_LDS_BANK_CONFLICT`, `SQ_LDS_ADDR_CONFLICT`, `SQ_LDS_DATA_FIFO_FULL`;
`SQ_INSTS_VMEM_RD`, `SQ_INST_LEVEL_VMEM`, `SQ_VMEM_TA_ADDR_FIFO_FULL`;
`TCC_HIT_sum`, `TCC_MISS_sum`, `TCC_EA0_RDREQ_DRAM_sum`.

## ATT instruction timeline

```bash
timeout -s KILL 600 rocprofv3 --att --att-target-cu 1 --kernel-include-regex <kernel> \
    --kernel-iteration-range "[2]" -d att_out -o r -- ./launcher ...
python tools/rocm/analysis/att_summary.py ours=att_out/.../stats_*.csv --top 30
python tools/rocm/analysis/att_timeline.py ours=att_out/.../ui_output_* --show 40
```

`att_summary.py` groups per-instruction latency / stall / idle by instruction class;
`att_timeline.py` measures the interval between consecutive MFMA issues of one wave and
what was issued in between. Reference point: hipBLASLt's BF16 GEMM issues one 16x16x32
MFMA every ~16.6 cycles per wave on MI355X; the LLVM-scheduled TIRx GEMM ~22.

## Baselines

`tools/rocm/analysis/rocblas_gemm.cpp` (hipBLASLt through rocBLAS, profilable) and
`aiter_a4w4_bench.py` (aiter MXFP4) give comparison points; `hipblaslt_kernels.py` names
the hipBLASLt kernel torch launches for a shape.

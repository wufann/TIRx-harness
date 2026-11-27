---
name: tirx-rocm-debug-kernel
description: >
  Debug and validate runnable gfx950 (MI350X / MI355X) TIRx kernels: wrong results,
  nondeterminism, hangs, GPU memory faults, and waitcnt / barrier / LDS-sharing or
  inline-asm hazards. Use for any failure or suspect result, and after changes to
  synchronization, LDS layout, or numerics.
---

# Debug a TIRx Kernel on gfx950

Work from a runnable case and evidence. Pick an idle GPU (`rocm-smi --showuse
--showmemuse`), set `HIP_VISIBLE_DEVICES`, and wrap every GPU command in
`timeout -s KILL <seconds>`: a hung kernel ignores SIGTERM.

## 1. Reproduce small

- Shrink to the smallest shape the kernel supports (one or two workgroups) and compare
  against a float32 torch reference in a standalone script.
- Print the error map, not just the max: which rows / columns / tiles are wrong. Map the
  pattern to the structure — a 16x16 block is one MFMA tile; rows `(l//16)*4..+4` of a
  column are one lane's accumulators; a 128x128 block is one wave; whole workgroup tiles
  are a stage / DMA / indexing problem.
- Run the same inputs several times. Results that change between runs are a race
  (missing wait or barrier), not an arithmetic error.

## 2. Synchronization checklist

For every LDS buffer, check in the generated ISA (`inspect_source("asm")`):

- **DMA -> read**: `s_waitcnt vmcnt(n)` covering this wave's DMAs into the buffer, then
  `s_barrier` (other waves' DMAs), before the first `ds_read` of it.
- **read -> overwrite**: a barrier after every wave's last `ds_read` of the old contents
  and before the DMA or `ds_write` that refills it.
- **vmcnt arithmetic**: count all vector-memory loads issued after the stage you wait
  for, including scale / auxiliary loads, not only the main DMAs.
- **ds_write -> ds_read** across waves: `lgkmcnt(0)` + barrier.
- **Inline asm**: LLVM does not track hazards or waits for values produced inside
  `T.amdgcn.asm`. A result of `mfma_scale_acc` read outside an MFMA needs `wait=True`;
  an asm that writes M0 must declare `~{m0}`; an asm load result needs your own
  `s_waitcnt` before use.
- Single-wave workgroups: LLVM deletes `s_barrier`; that is correct, not a bug.

## 3. Numerics checklist

- bf16 data are int16 bit patterns; fp16 uses `float16`. A buffer declared `bfloat16`
  is legalized to f32 and breaks byte-level layouts.
- MFMA lane layouts (tirx-rocm-wiki gfx950.md): wrong k-grouping (`(l//16)*8`) or a
  swapped A/Bt role transposes or permutes the result in recognizable ways.
- FP4: the even K element is the low nibble; E8M0 scale = 2^(byte - 127); the scale byte
  a lane uses is selected by `opsel`.
- `cvt_pk_bf16_f32` rounds to nearest even; torch's `.to(torch.bfloat16)` does too.
- Swizzled LDS: the reader must apply the same `chunk ^ f(row)` as the writer.

## 4. Hangs and faults

- Hang: usually a barrier not reached by every wave (barrier inside a wave-divergent
  branch, different trip counts per wave) or a wait that can never be satisfied. Kill
  with `timeout -s KILL`, then confirm the GPU is idle again with `rocm-smi`.
- `Memory access fault by GPU node-N ... Page not present or supervisor privilege`:
  an out-of-bounds global address. Re-run with `AMD_SERIALIZE_KERNEL=3` and
  `HIP_LAUNCH_BLOCKING=1` to pin the faulting launch, and `AMD_LOG_LEVEL=3` for runtime
  logs. Check index math at tile edges and the `num_records` of buffer resources
  (out-of-range buffer loads return 0 instead of faulting — a silent wrong result).
- `rocgdb` (on PATH in the sglang ROCm images) can attach to a hung process
  (`rocgdb -p <pid>`, then `info agents`, `info dispatches`, `info threads` and `bt`
  on a wave) to show which instruction each wave is stuck at.

## 5. Validate

After a fix: the standalone small-shape check, several repeated runs, then the task's
benchmark command (its correctness gate runs multiple times and after timing).

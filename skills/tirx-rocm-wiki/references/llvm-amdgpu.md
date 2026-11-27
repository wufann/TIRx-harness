# What the LLVM AMDGPU backend does with explicit control

TIRx compiles ROCm kernels through LLVM (AMD LLVM 23 in the sglang ROCm 10 images).
Everything below was measured on gfx950 with tvm-tirx `rocm-gfx950`.

## Kept as written

- `s_waitcnt`, `s_setprio`, `s_barrier` stay where they are written. LLVM only **adds**
  waits that data dependences need (SIInsertWaitcnts); it never removes yours.
- `sched_group_barrier` patterns are followed (e.g. `load x2 -> MFMA` repeated appears in
  that order) as long as the requested instructions exist in the region.
- `s_barrier` is deleted when the workgroup is a single wave.

## Waits LLVM adds

- Any use of a value produced by an outstanding load gets the minimal `vmcnt` /
  `lgkmcnt`. If an MFMA consumes the load issued right before it, you get `vmcnt(0)`
  there regardless of your schedule.
- LDS reads after DMAs into the **same LDS allocation** wait for all DMAs (LLVM cannot
  tell which part a DMA writes). Give each stage its own allocation.
- Without `s_setprio` around the MFMA block, LLVM inserted about 20 `lgkmcnt(0)` per
  iteration in a 4-wave GEMM; with it, the loop kept the intended waits.

## Register classes

- 256 accumulator floats per lane need AGPRs. That takes both the function attribute
  `amdgpu-agpr-alloc` (non-zero, e.g. `"256"`, via `rocm.llvm_fn_attrs`) **and** the
  backend option `-amdgpu-mfma-vgpr-form=0` (target `llvm-options:
  ["amdgpu-mfma-vgpr-form=0"]`). With only one of them LLVM uses the VGPR form and
  copies accumulators between VGPRs and AGPRs every iteration.
- Scaled MFMAs (`v_mfma_scale_*`) are never selected with dst == srcC; use
  `mfma_scale_acc` (inline asm, accumulator tied) to accumulate in place.
- **AGPR copy storms**: some schedules make the register allocator move hundreds of
  accumulators per iteration (`v_accvgpr_read/write`, `v_accvgpr_mov`). Observed
  triggers: DMAs issued in the SGPR-base + VGPR-offset form via inline asm or zext
  offsets; forcing single fillers between MFMAs with SALU groups in
  `sched_group_barrier`; `s_setprio` combined with a cross-stage-prefetch loop shape.
  Always count `v_accvgpr` in the main loop after a schedule change — a storm costs
  more than any scheduling gain.

## Scheduling limits

- The machine scheduler may cluster fillers (DMAs, `ds_read`, M0 writes) instead of
  spreading one per MFMA gap, and may sink `ds_read`s next to their consumer. The
  best measured LLVM-scheduled BF16 GEMM issues an MFMA every ~22 cycles where
  hipBLASLt manages ~16.6.
- Backend scheduler options tried without benefit: `amdgpu-igrouplp-exact-solver`,
  `amdgpu-sched-strategy=max-ilp`, `amdgpu-use-amdgpu-trackers`,
  `amdgpu-disable-unclustered-high-rp-reschedule`.
- `amdgpu-waves-per-eu` and `amdgpu-agpr-alloc` change register budgets; check the
  `.vgpr_count` / `.agpr_count` / spill counts in the assembly metadata.

## Reading the result

The assembly (`inspect_source("asm")`) ends with kernel metadata: `.sgpr_count`,
`.vgpr_count`, `.agpr_count`, `.group_segment_fixed_size` (LDS bytes),
`.private_segment_fixed_size` (scratch: spills), occupancy hints. Check them along with
the main-loop instruction counts after every structural change.

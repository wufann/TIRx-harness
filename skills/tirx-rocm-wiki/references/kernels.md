# gfx950 kernels in tirx_kernels/rocm

Installed with the run's kernel package
(`python -c "import tirx_kernels.rocm as m; print(m.__path__[0])"`). Study, copy and
modify them; a candidate must carry its own copy rather than import them.

## bf16_gemm.py — BF16 TN GEMM

`make_gemm_v1(M, N, K, variant)` returns a `@T.function` computing
C = A @ Bt^T with A [M*K] and Bt [N*K] as int16 bit patterns, C as bf16 packed in
int32 (`out="bf16"`) or fp32. Compile with `target_for(variant)` (adds
`amdgpu-mfma-vgpr-form=0` for AGPR variants). Requires M, N multiples of the tile and
K a multiple of 2*BK.

Variant knobs (`BASE` and `VARIANTS`):

| Knob | Meaning |
|---|---|
| `BM, BN, BK`, `waves` | workgroup tile; wave grid (2x2 = 4 waves of 128x128, 2x4 = 8 waves of 128x64) |
| `agpr` | accumulators in AGPRs (needed for 128x128 wave tiles) |
| `early` | next stage's DMAs right after the barrier instead of in the last substep |
| `prio` | `s_setprio 1` around the MFMA block |
| `bdma` | `buffer_load_lds` with loop-invariant voffset (hipBLASLt form) instead of `global_load_lds` |
| `lds_epi` | bf16 epilogue staged through swizzled LDS, 16-byte stores |
| `out` | `f32`, `f32t` (operands swapped, one 16 B store per tile), `bf16` |
| `sdma`, `zdma`, `fine`, `spread`, `m0split` | schedules kept as negative results (AGPR copy storms or longer MFMA gaps) |

Measured on MI355X (cold-ish, best of 3): `w4e_prio_bf16_agpr_bdma_lepi` 1288 TFLOPS at
4096^3 (~95-99% of hipBLASLt), 1440 TFLOPS at 8192^3 (88%). For grids smaller than one
256x256 tile per CU, `t128_*` (128x128 tiles) wins; `dispatch.py` holds the measured
shape -> variant table and a fallback heuristic.

Structure to reuse: per-stage LDS allocations with the stage helper called twice per
loop iteration, 16-byte DMA chunks XOR-swizzled (`chunk ^ (row % 8)`), substep-1
fragments read during substep-0 MFMAs, the schedule emitted by a plain Python function
(`emit_schedule`) as `sched_group_barrier` groups.

## mxfp4_gemm.py — MXFP4 TN GEMM

`make_mxfp4_gemm(M, N, K, variant)`: A [M, K/2] and B [N, K/2] fp4 bytes and E8M0
scales SA [M, K/32], SB [N, K/32], all passed as int32 views; bf16 output packed in
int32. Same skeleton as the BF16 kernel: one 256-element K stage is 128 bytes per row,
exactly a BF16 BK=64 stage, so the data path is reused; scales come in by DMA into small
LDS buffers. Requires M, N multiples of 256 and K a multiple of 512.

| Variant | Mechanism | 8192^3 TFLOPS |
|---|---|---|
| `w4` | raw scales, per-lane shifts select the scale byte | 3241 |
| `w4s` | scales pre-shuffled (`shuffle_scales`) -> one u16 read per tile, `opsel` = substep | 3533 |
| `w4sa_noprio` | + `mfma_scale_acc` (accumulate in place), no setprio | ~4000-4100 (78-80% of aiter) |
| `w4sab*` | + B pre-shuffled (`shuffle_b`) straight into VGPRs | no gain over `w4sa_noprio` |

`random_mxfp4` makes test data; `shuffle_scales` / `shuffle_b` are host-side layout
transforms (fine for weights; activation scales must be handled inside the timed call).

## common.py

`bench(fn)` (event timing), `loop_summary(asm)` (instruction counts and waits of the
main loop), `spread(total, groups)`.

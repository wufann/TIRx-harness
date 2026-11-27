# Production gfx950 GEMM patterns

Main-loop structure of three production BF16 GEMMs (256x256 workgroup tiles), read from
their gfx950 assembly (hipBLASLt from its kernel name, counters and ATT only: its code
object was not disassembled), and what transferred to TIRx.

| | aiter (hand-written asm) | hipBLASLt (Tensile, `MT256x256x64_MI16x16x1`) | HipKittens (HIP + inline asm) |
|---|---|---|---|
| Waves / wave tile | 4 waves, 128x128 each | 4 waves, 128x128 each | 8 waves, 128x64 each, ping-pong |
| MFMA | 256 x 16x16x32 bf16 per loop | 16x16x32 bf16 (`MI16x16x1`) | 128 x 16x16x32 bf16 per loop |
| Accumulators | AGPR | AGPR | VGPR |
| Global -> LDS | `buffer_load_dwordx4 ... lds`, 32 per loop | `buffer_load ... lds`, soffset advance | `buffer_load_dwordx4 ... lds` |
| LDS -> VGPR | `ds_read_b128` | `ds_read_b128` | `ds_read_b128` |
| Waits per loop | 4 (`vmcnt(8/12) lgkmcnt(0)`) | not extracted | ~10 |
| Barriers per loop | 4 | not extracted | 16 |
| `s_setprio` | none | not extracted | 1/0 around every MFMA burst |
| Epilogue | `v_accvgpr_read`, `v_cvt_pk_bf16_f32`, `v_perm`, `buffer_store_dwordx4` | not extracted | per-element stores |
| MFMA issue interval (ATT) | not measured | ~16.6 cycles | not measured |

## aiter: fine-grained interleave + multi-stage LDS ring

```
s_waitcnt vmcnt(8) lgkmcnt(0); s_barrier
MFMA DMA MFMA ds_read MFMAx2 ds_read MFMAx2 ds_read ... (one filler per MFMA gap)
s_waitcnt vmcnt(12) lgkmcnt(0); s_barrier
```

- A memory instruction sits in almost every MFMA shadow: ~1 `ds_read_b128` per 2-3
  MFMAs, 1 DMA per 8 MFMAs, so the MFMA pipe never drains.
- Stage readiness is counted with vmcnt (stages in flight x DMAs per stage); one wait
  and one barrier per half iteration.
- B-preshuffle variants load the (pre-arranged) weight straight into VGPRs with
  `buffer_load_dwordx4`; only A goes through LDS.

## HipKittens: 8-wave ping-pong

```
[ds_read x8 + DMA x2] s_waitcnt lgkmcnt(8) s_barrier s_waitcnt lgkmcnt(0)
s_setprio 1  MFMA x16  s_setprio 0  s_barrier
```

Two wave groups alternate memory and compute phases aligned by barriers; `s_setprio 1`
favours the group issuing MFMAs. Even this C++ library pins its core instructions with
inline asm to stop the compiler from rescheduling them.

## MXFP4 (aiter `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256`)

256x256 tiles with `v_mfma_scale_f32_16x16x128_f8f6f4`, B pre-shuffled into the MFMA
lane order and loaded straight to registers, scales pre-shuffled so each lane reads its
bytes with one load, accumulation in place in AGPRs.

## What transferred to TIRx (measured on MI355X)

- 4 waves x 128x128 with AGPR accumulators beat 8 waves x 128x64 once the AGPR form is
  forced (see llvm-amdgpu.md).
- `buffer_load_lds` with a loop-invariant voffset removed all VALU address math from
  the loop; on its own it did not change speed — the bottleneck was MFMA issue cadence.
- Next-stage DMAs issued early (right after the barrier) beat issuing them late.
- An LDS-staged epilogue (cvt to bf16, write swizzled LDS, 16-byte global stores)
  gained 1-2%.
- Small grids (fewer 256x256 tiles than CUs) need 128x128 tiles; tile count vs. 256
  CUs (wave quantization) dominates mid-size shapes.
- Cutting the loop to aiter's 4 waits with a BK=32 multi-stage ring and cross-stage
  prefetch did not beat the 2-stage BK=64 kernel: waits were not the bottleneck.

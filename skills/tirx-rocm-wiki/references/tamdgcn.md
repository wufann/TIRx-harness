# The T.amdgcn dialect

`T.amdgcn.<name>(...)` inside TVMScript; source `tvm_tirx/backend/rocm/amdgcn.py`.
Entries lower to `llvm.amdgcn.*` intrinsics, or to LLVM inline asm where no intrinsic
exists. LLVM still schedules and inserts waits around them (see llvm-amdgpu.md).

## Matrix

| Call | Instruction | Operands (per lane) |
|---|---|---|
| `mfma("16x16x32", "bf16", a, b, c)` | `v_mfma_f32_16x16x32_bf16` | a, b `int16x8`; c / result `float32x4` |
| `mfma("32x32x16", "bf16", a, b, c)` | `v_mfma_f32_32x32x16_bf16` | a, b `int16x8`; c `float32x16` |
| `mfma("16x16x32", "f16", ...)`, `("32x32x16", "f16")` | f16 variants | a, b `float16x8` |
| `mfma("16x16x32", "fp8", ...)` | `v_mfma_f32_16x16x32_fp8_fp8` | a, b `int64` (8 x fp8) |
| `mfma_scale(shape, a, b, c, scale_a, scale_b, fmt_a="fp4", fmt_b="fp4", opsel_a=0, opsel_b=0)` | `v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4` | fp4: a, b `int32x4` (32 values); scales `int32`, byte `opsel` used |
| `mfma_scale_acc(..., wait=False)` | same, inline asm, accumulator tied in place in AGPRs | as `mfma_scale` |

Usage: `acc[T.Ramp(i * 4, 1, 4)] = T.amdgcn.mfma("16x16x32", "bf16", af[...], bf[...], acc[T.Ramp(i * 4, 1, 4)])`.
Layouts are in gfx950.md. `MFMA`, `MFMA_SCALE`, `MX_FORMAT` hold the tables.

`mfma_scale_acc` exists because LLVM never selects a scaled MFMA with dst == srcC: a
kernel with many accumulators then needs a second register set and copies / spills
them. The asm hides the MFMA from LLVM's hazard recognizer: do not read the result
outside another MFMA until it has been written back — `wait=True` puts the required
`s_nop`s inside the same asm block (a separate asm `s_nop` can be reordered away).

## Memory

| Call | Instruction | Notes |
|---|---|---|
| `global_load_lds(src_addr, lds_addr, size=16, offset=0, aux=0)` | `global_load_lds_dword{,x2,x3,x4}` | lane i copies `size` bytes from its `src_addr` to `lds_addr + offset + i*size`; `lds_addr` must be wave-uniform (goes to M0); tracked by vmcnt |
| `buffer_load_lds(base_addr, num_records, lds_addr, voffset, soffset=0, size=16, offset=0, aux=0)` | `buffer_load_dword* ... lds` | raw buffer resource over `[base, base+num_records)` built in SGPRs; per-lane 32-bit `voffset` (keep it loop-invariant), uniform `soffset` carries the moving part; out-of-range lanes read 0 |
| `ds_read_tr16_b64(lds_addr)` | `ds_read_b64_tr_b16` | transposing LDS read, returns `int16x4` |

Addresses are `T.address_of(buf[index])`. Ordinary loads / stores of `T.Ramp`
vectors produce `global_load_dwordx4`, `ds_read_b128`, `ds_write_b128` etc.

## Synchronization and scheduling

| Call | Effect |
|---|---|
| `s_waitcnt(vmcnt=None, expcnt=None, lgkmcnt=None)` | wait until at most N operations of each named counter are outstanding (omitted = not waited) |
| `s_barrier()` | workgroup barrier (LLVM drops it for single-wave workgroups) |
| `s_setprio(p)` | wave issue priority 0..3 |
| `s_sleep(n)` | about 64*n cycles |
| `sched_barrier(mask)` | scheduler fence; only classes in `mask` may cross |
| `sched_group_barrier(mask, size, sync_id)` | request a group of `size` instructions of class `mask`; consecutive calls with one `sync_id` form an ordered pipeline |

`SG` masks: `ALU VALU SALU MFMA VMEM VMEM_READ VMEM_WRITE DS DS_READ DS_WRITE TRANS`.

## Lanes, conversion, escape hatch

| Call | Effect |
|---|---|
| `readfirstlane(v)` | `v_readfirstlane_b32`, value of the first active lane as a uniform `int32` |
| `cvt_pk_bf16_f32(lo, hi)` | two f32 -> packed bf16x2 (RNE) in an `int32`, `lo` in bits 15:0 |
| `asm(text, constraints, *args, ty="void", side_effects=True)` | LLVM inline asm; `$0..` operands, constraints `v` / `s` / `a` (AGPR), `=` outputs, tied `"0"`, clobbers `~{m0}` |

Inline asm is opaque to LLVM's hazard recognizer and scheduler model: account for MFMA
result latency and M0 / exec side effects yourself.

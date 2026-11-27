# TVMScript on ROCm (gfx950)

A minimal kernel: one wave per workgroup copies 64 int32 values through LDS.

```python
import torch
import tvm_tirx
from tvm_tirx.script import tirx as T

TARGET = {"kind": "rocm", "mcpu": "gfx950"}

@T.function
def copy(src: T.Tensor((64,), "int32"), dst: T.Tensor((64,), "int32")):
    T.func_attr({"tirx.noalias": True})
    with T.launch_thread("blockIdx.x", 1) as bx:
        with T.launch_thread("threadIdx.x", 64) as tid:
            buf = T.alloc_tensor(ty_args=[T.Tensor((64,), "int32", scope="shared")])
            buf[tid] = src[tid]
            T.amdgcn.s_waitcnt(lgkmcnt=0)
            T.amdgcn.s_barrier()
            dst[tid] = buf[63 - tid]

f = tvm_tirx.compile(copy, target=TARGET)
a = torch.arange(64, dtype=torch.int32, device="cuda"); b = torch.empty_like(a)
f(a, b)                                      # torch tensors are passed directly
asm = f.mod.imports[0].inspect_source("asm")  # gfx950 assembly
llvm_ir = f.mod.imports[0].inspect_source("hip")  # LLVM IR (amdgcn)
```

## Entry, launch, memory

- Entry parameters are `T.Tensor(shape, dtype)`; flat 1-D tensors keep index math
  explicit. Pass torch tensors; reinterpret with `t.view(torch.int16).flatten()` when
  the kernel types operands as bit patterns.
- `T.launch_thread("blockIdx.x"/"blockIdx.y", n)` and `T.launch_thread("threadIdx.x", n)`;
  a wave is 64 lanes: `wave = tid // 64`, `lane = tid % 64`. Make the wave index
  uniform with `T.amdgcn.readfirstlane(tid // 64)` so role / address math stays in SGPRs.
- **Allocate inside the `threadIdx` scope.** A `scope="shared"` allocation at function
  root is turned into a global-memory kernel parameter shared by all workgroups.
- LDS: `T.alloc_tensor(ty_args=[T.Tensor((n,), dtype, scope="shared")])`, up to 160 KB
  per workgroup. Registers: `scope="local"`; constant-indexed local arrays become VGPRs
  (or AGPRs, see llvm-amdgpu.md).
- **One LDS allocation per pipeline stage.** With a ring in one allocation LLVM cannot
  tell which stage a DMA writes and waits for every outstanding DMA before each LDS read.
  Separate allocations plus a stage loop unrolled at trace time (call the stage helper
  once per buffer) keep the waits you write.
- Function attributes reach LLVM: `T.func_attr({"rocm.llvm_fn_attrs":
  {"amdgpu-agpr-alloc": "256", "amdgpu-waves-per-eu": "1,1"}})`. Backend options go in
  the target: `{"kind": "rocm", "mcpu": "gfx950", "llvm-options":
  ["amdgpu-mfma-vgpr-form=0"]}` (applied per build).

## Types

- bf16 (and fp8 / fp4) data are **integer bit patterns** in TIR on ROCm: `int16` for
  bf16, `int32` / `int64` / `uint8` containers for fp8 / fp4. A `bfloat16` buffer is
  legalized to f32 (an LDS buffer silently doubles in size and breaks DMA layouts).
  fp16 has a native type (`float16`, `float16x8` MFMA operands).
- Vector access: `buf[T.Ramp(base, 1, 8)]` loads/stores 8 consecutive elements (one
  16-byte `ds_read_b128` / `global_load_dwordx4` when aligned).
- `T.reinterpret(value, ty="int32x4")` reinterprets bits between same-sized types.

## Trace-time structure

- `@T.inline` helpers are expanded at the call site; helpers called from parsed code
  must be `@T.inline`. `T.unroll(n)` unrolls; plain `range` makes a loop.
- Plain Python functions called inside parsed code run at trace time: use them to emit
  long `sched_group_barrier` sequences with ordinary Python loops and lists.
- Comparisons on Python values inside parsed code become TIR expressions; compute
  configuration booleans in Python before the function and branch on those.
- Every `if` / `elif` branch of parsed code is traced, so a branch must be valid TIR
  even when its condition is constant false.
- Do not alias a shared buffer with `T.decl_tensor(buf.data, ...)`: the alias was
  observed to write to address 0. Allocate the buffer with the element type you read.

## Compile and inspect

- `tvm_tirx.compile(func, target=TARGET)` returns a callable module; compile once in
  `setup()`. Launch configuration comes from the `launch_thread` extents.
- `inspect_source("asm")` is the ground truth for what runs. Find the main loop as the
  block between a label and its backward `s_cbranch_*`, then count `v_mfma`,
  `ds_read`, `buffer_load ... lds`, `s_waitcnt` and `v_accvgpr_*` there.

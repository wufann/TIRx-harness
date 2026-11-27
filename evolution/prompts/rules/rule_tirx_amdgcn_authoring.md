- Author every production GPU kernel in TIRx's TVMScript (`@T.function`, with
  `from tvm_tirx.script import tirx as T`) and compile it with
  `tvm_tirx.compile(...)` for the ROCm target `{"kind": "rocm", "mcpu": "gfx950"}`
  (target options such as `llvm-options` are allowed). Every kernel launched by
  the timed callable must come from such a function.
- Spell AMDGPU instructions through the `T.amdgcn` dialect
  (`tvm_tirx.backend.rocm.amdgcn`): MFMA / scaled MFMA, global and buffer loads
  into LDS, `s_waitcnt`, barriers, `s_setprio`, `sched_group_barrier`, lane
  operations. `T.amdgcn.asm` may express single instructions the dialect lacks;
  do not write a kernel's main loop or whole body as one inline-asm block.
- Do not call hipBLASLt / rocBLAS, aiter, Composable Kernel, Triton, FlyDSL,
  torch.matmul or other PyTorch tensor operators, HIP C++ or another runtime for
  production computation, and do not load prebuilt code objects.
- The gfx950 kernels under `tirx_kernels/rocm/` are reference material: study,
  copy and modify them freely, but a candidate must be self-contained — copy what
  you use into its own `solution.py` instead of importing a `tirx_kernels.rocm`
  kernel module.

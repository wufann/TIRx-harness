"""Post-build smoke check of a tvm-tirx ROCm build (runs inside the dev container)."""

import tvm_ffi

import tvm_tirx
from tvm_tirx import base
from tvm_tirx.target import Target

print("[smoke] tvm_tirx", getattr(tvm_tirx, "__version__", "?"), tvm_tirx.__file__)
print("[smoke] tvm_ffi", tvm_ffi.__version__)

for name, lib in sorted(base._LOADED_LIBS.items()):
    print(f"[smoke] loaded {name}: {lib._name}")

t = Target({"kind": "rocm", "mcpu": "gfx950"})
print("[smoke] target:", t)
for name in ["device_api.rocm", "target.build.rocm", "ffi.Module.load_from_file.hsaco"]:
    print(
        f"[smoke] global func {name}:",
        tvm_ffi.get_global_func(name, allow_missing=True) is not None,
    )
dev = tvm_tirx.rocm(0) if hasattr(tvm_tirx, "rocm") else None
if dev is not None:
    print("[smoke] rocm(0) exist:", dev.exist)

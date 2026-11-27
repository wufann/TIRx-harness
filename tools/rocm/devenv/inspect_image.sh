#!/bin/bash
# Read-only inspection of a container image's toolchain for building tvm-tirx (ROCm).
# Usage (on a compute node): docker run --rm -i --entrypoint bash IMAGE -s < inspect_image.sh
echo "== OS / ROCm"; . /etc/os-release; echo "$PRETTY_NAME"; cat /opt/rocm/.info/version 2>/dev/null; ls -d /opt/rocm* 2>/dev/null
echo "== Python"; which python python3; python3 --version
echo "== pip (tvm/llvm/related)"; python3 -m pip list 2>/dev/null | grep -iE "^(tvm|apache-tvm|tilelang|triton|torch |torch$|pytorch|aiter|flydsl|mlir|llvm|tvm-ffi|apache-tvm-ffi|cmake|ninja|scikit-build|hip|amd|sglang|cutlass|rocm)"
echo "== import tvm"; python3 - <<'PY'
for m in ["tvm", "tvm_ffi", "tvm_tirx", "tilelang", "triton", "torch"]:
    try:
        mod = __import__(m)
        print(m, getattr(mod, "__version__", "?"), getattr(mod, "__file__", "?"))
    except Exception as e:
        print(m, "NOT importable:", type(e).__name__, str(e)[:120])
try:
    import tvm
    info = tvm.support.libinfo()
    for k in ["LLVM_VERSION", "USE_LLVM", "USE_ROCM", "ROCM_PATH", "GIT_COMMIT_HASH", "USE_CUDA"]:
        print("  tvm.libinfo", k, "=", info.get(k))
except Exception as e:
    print("tvm libinfo n/a:", type(e).__name__, str(e)[:120])
try:
    import torch; print("torch.hip", torch.version.hip)
except Exception: pass
PY
echo "== find other tvm copies"; find / -xdev \( -name "libtvm*.so" -o -name "libtilelang*.so" \) 2>/dev/null | grep -v proc | head -20
echo "== LLVM candidates"
for c in /opt/rocm/llvm/bin/llvm-config /opt/rocm/lib/llvm/bin/llvm-config $(ls /usr/lib/llvm-*/bin/llvm-config 2>/dev/null) $(which -a llvm-config 2>/dev/null); do
  [ -x "$c" ] && echo "$c: $($c --version) shared=$($c --shared-mode 2>/dev/null) targets=[$($c --targets-built)] libdir=$($c --libdir)"
done
for d in /opt/rocm/llvm /opt/rocm/lib/llvm; do
  [ -d $d ] || continue
  echo "-- $d"; ls $d/bin | grep -E "^(clang|llc|ld.lld|llvm-config|llvm-objdump|llvm-mc)$" | tr '\n' ' '; echo
  $d/bin/clang --version 2>/dev/null | head -2
  ls $d/lib/cmake 2>/dev/null | tr '\n' ' '; echo
  ls -d $d/include/llvm $d/include/llvm-c 2>/dev/null
  ls $d/lib/libLLVM*.so* $d/lib/libLLVMCore.a $d/lib/libLLVMAMDGPUCodeGen.a 2>/dev/null | head
  $d/bin/llc -march=amdgcn -mcpu=help 2>&1 | grep -oE "gfx95[0-9]" | sort -u | tr '\n' ' '; echo
  echo 'declare <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32>, <8 x i32>, <16 x float>, i32, i32, i32, i32, i32, i32)' > /tmp/t.ll
  $d/bin/llvm-as /tmp/t.ll -o /dev/null 2>&1 && echo "scaled f8f6f4 MFMA intrinsic: known" || echo "scaled f8f6f4 MFMA intrinsic: unknown"
done
echo "== build tools"
for t in cmake ninja make gcc g++ clang clang++ hipcc git uv cargo rustc; do printf "%-8s " $t; (command -v $t >/dev/null && $t --version 2>&1 | head -1) || echo "missing"; done
echo "== disk"; df -h / /tmp 2>/dev/null | tail -2

"""Hardware-counter profile of BF16 GEMM implementations with rocprofv3.

python prof_gemm.py profile --impl ours:w4e_prio_bf16_agpr rocblas --shape 8192,8192,8192
    runs one rocprofv3 --pmc pass per counter group for each implementation and prints
    per-kernel metrics side by side. Workloads avoid torch (rocprofv3 hangs on it):
    ours runs on tvm runtime tensors, the hipBLASLt baseline is rocblas_gemm (C++).
python prof_gemm.py run --impl IMPL --shape M,N,K     (the workload rocprofv3 wraps)
"""

import argparse
import csv
import glob
import os
import subprocess
import sys

PASSES = [
    [
        "GRBM_GUI_ACTIVE",
        "SQ_WAVE_CYCLES",
        "SQ_BUSY_CYCLES",
        "SQ_WAIT_ANY",
        "SQ_WAIT_INST_ANY",
        "SQ_WAIT_INST_LDS",
    ],
    [
        "SQ_VALU_MFMA_BUSY_CYCLES",
        "SQ_INSTS_VALU_MFMA_MOPS_BF16",
        "SQ_INSTS_MFMA",
        "SQ_BUSY_CU_CYCLES",
        "SQ_INSTS_VALU",
        "SQ_INSTS_SALU",
    ],
    [
        "SQ_LDS_BANK_CONFLICT",
        "SQ_LDS_ADDR_CONFLICT",
        "SQ_LDS_IDX_ACTIVE",
        "SQ_INSTS_LDS",
        "SQ_LDS_UNALIGNED_STALL",
        "SQ_LDS_DATA_FIFO_FULL",
    ],
    [
        "SQ_INSTS_VMEM_RD",
        "SQ_ACTIVE_INST_VMEM",
        "SQ_INST_LEVEL_VMEM",
        "SQ_ACTIVE_INST_LDS",
        "SQ_INST_LEVEL_LDS",
        "SQ_VMEM_TA_ADDR_FIFO_FULL",
    ],
    ["TCC_HIT_sum", "TCC_MISS_sum", "TCC_EA0_RDREQ_sum", "TCC_EA0_RDREQ_DRAM_sum"],
]


def run(impl, shape, iters):
    """Workload for rocprofv3. No torch: rocprofv3 hangs on Python processes that use HIP
    in the sglang ROCm 10 image (tvm runtime tensors work)."""
    import numpy as np

    M, N, K = shape
    if not impl.startswith("ours:"):
        raise ValueError(f"{impl}: only ours:<variant> runs in Python; rocblas runs as a binary")
    import tvm_tirx
    from tirx_kernels.rocm.bf16_gemm import make_gemm_v1, target_for

    variant = impl.split(":", 1)[1]
    f = tvm_tirx.compile(make_gemm_v1(M, N, K, variant), target=target_for(variant))
    dev = tvm_tirx.rocm(0)
    rng = np.random.default_rng(0)

    def bits(n):  # bf16 bit patterns of N(0, 1) samples
        f32 = rng.standard_normal(n).astype(np.float32).view(np.uint32)
        return (f32 >> 16).astype(np.uint16).view(np.int16)

    a = tvm_tirx.runtime.tensor(bits(M * K), dev)
    bt = tvm_tirx.runtime.tensor(bits(N * K), dev)
    c = tvm_tirx.runtime.empty((M * N // 2,), "int32", dev)
    for _ in range(3 + iters):
        f(a, bt, c)
    dev.sync()


HERE = os.path.dirname(os.path.abspath(__file__))
ROCBLAS_BIN = os.path.join(HERE, "rocblas_gemm")
HSACO_BIN = os.path.join(HERE, "hsaco_gemm")


def workload_cmd(impl, shape, iters):
    if impl == "rocblas":
        return [ROCBLAS_BIN, *map(str, shape), str(iters)]
    if impl.startswith("hsaco:"):  # hsaco:<file>:<kernel>:<BM>:<BN>:<NT> from export_hsaco.py
        _, path, kernel, bm, bn, nt = impl.split(":")
        return [HSACO_BIN, path, kernel, *map(str, shape), bm, bn, nt, str(iters)]
    return [
        sys.executable,
        os.path.abspath(__file__),
        "run",
        "--impl",
        impl,
        "--shape",
        ",".join(map(str, shape)),
        "--iters",
        str(iters),
    ]


def collect(outdir):
    """kernel name -> {counter: mean value per dispatch}."""
    rows = {}
    for path in glob.glob(os.path.join(outdir, "**", "*counter_collection.csv"), recursive=True):
        for r in csv.DictReader(open(path)):
            name = r.get("Kernel_Name") or r.get("Kernel-Name") or "?"
            cname = r.get("Counter_Name") or r.get("Counter-Name")
            val = float(r.get("Counter_Value") or r.get("Counter-Value") or 0)
            disp = r.get("Dispatch_Id") or r.get("Dispatch-Id")
            rows.setdefault(name, {}).setdefault(cname, {}).setdefault(disp, 0.0)
            rows[name][cname][disp] += val  # sum over instances / dimensions within a dispatch
    return {k: {c: sum(d.values()) / len(d) for c, d in v.items()} for k, v in rows.items()}


class _V(float):
    """float whose division by zero gives nan (counters can legitimately read 0)."""

    def __truediv__(self, o):
        return _V(float(self) / o) if o else _V(float("nan"))


def metrics(c):
    g = lambda k: _V(c.get(k, float("nan")))  # noqa: E731
    out = {
        "gui_active_cycles": g("GRBM_GUI_ACTIVE"),
        "mfma_busy/busy_cu": g("SQ_VALU_MFMA_BUSY_CYCLES") / g("SQ_BUSY_CU_CYCLES"),
        "wait_any/wave_cycles": g("SQ_WAIT_ANY") / g("SQ_WAVE_CYCLES"),
        "wait_inst_any/wave_cycles": g("SQ_WAIT_INST_ANY") / g("SQ_WAVE_CYCLES"),
        "wait_inst_lds/wave_cycles": g("SQ_WAIT_INST_LDS") / g("SQ_WAVE_CYCLES"),
        "lds_bank_conflict/lds_active": g("SQ_LDS_BANK_CONFLICT") / g("SQ_LDS_IDX_ACTIVE"),
        "lds_addr_conflict/lds_active": g("SQ_LDS_ADDR_CONFLICT") / g("SQ_LDS_IDX_ACTIVE"),
        "lds_data_fifo_full": g("SQ_LDS_DATA_FIFO_FULL"),
        "lds_insts": g("SQ_INSTS_LDS"),
        "vmem_rd_insts": g("SQ_INSTS_VMEM_RD"),
        "avg_vmem_latency": g("SQ_INST_LEVEL_VMEM") / g("SQ_ACTIVE_INST_VMEM"),
        "avg_lds_latency": g("SQ_INST_LEVEL_LDS") / g("SQ_ACTIVE_INST_LDS"),
        "ta_addr_fifo_full": g("SQ_VMEM_TA_ADDR_FIFO_FULL"),
        "mfma_insts": g("SQ_INSTS_MFMA"),
        "valu_insts": g("SQ_INSTS_VALU"),
        "salu_insts": g("SQ_INSTS_SALU"),
        "l2_hit_rate": g("TCC_HIT_sum") / (float(g("TCC_HIT_sum")) + float(g("TCC_MISS_sum"))),
        "hbm_rdreq": g("TCC_EA0_RDREQ_DRAM_sum"),
    }
    return out


def profile(impls, shape, iters, workdir):
    results = {}
    for impl in impls:
        merged = {}
        for i, pmc in enumerate(PASSES):
            tag = impl.replace(":", "_").replace("/", "_")[-60:]
            out = os.path.join(workdir, "x".join(map(str, shape)), tag, f"p{i}")
            subprocess.run(["rm", "-rf", out], check=False)
            cmd = [
                "timeout",
                "-s",
                "KILL",
                "300",
                "rocprofv3",
                "--pmc",
                *pmc,
                "-d",
                out,
                "-o",
                "run",
                "--output-format",
                "csv",
                "--",
                *workload_cmd(impl, shape, iters),
            ]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                print(
                    f"[prof] {impl} pass {i} rc={p.returncode}: "
                    f"{p.stderr.strip().splitlines()[-1:]} "
                )
            for k, v in collect(out).items():
                merged.setdefault(k, {}).update(v)
        # the GEMM kernel = the one with the most MFMA instructions
        gemm = max(merged, key=lambda k: merged[k].get("SQ_INSTS_MFMA", 0), default=None)
        results[impl] = (gemm, metrics(merged[gemm]) if gemm else {}, merged.get(gemm, {}))
        print(f"[prof] {impl}: kernel {gemm[:100] if gemm else None}")
    raw = sorted({c for impl in impls for c in results[impl][2]})
    keys = list(next(iter(results.values()))[1].keys())
    names = [
        i.split(":")[1].rsplit("/", 1)[-1][:26] if i.startswith("hsaco:") else i[:26] for i in impls
    ]
    print(f"{'metric':34s}" + "".join(f"{n:>28s}" for n in names))
    for k in keys:
        print(f"{k:34s}" + "".join(f"{results[i][1].get(k, float('nan')):28.4g}" for i in impls))
    print("-- raw counters (mean per dispatch, summed over instances)")
    for k in raw:
        print(f"{k:34s}" + "".join(f"{results[i][2].get(k, float('nan')):28.4g}" for i in impls))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "profile"])
    p.add_argument("--impl", nargs="+", default=["rocblas"])
    p.add_argument("--shape", default="8192,8192,8192")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--workdir", default="artifacts/prof")
    args = p.parse_args()
    shape = tuple(map(int, args.shape.split(",")))
    if args.mode == "run":
        run(args.impl[0], shape, args.iters)
    else:
        profile(args.impl, shape, args.iters, args.workdir)


if __name__ == "__main__":
    main()

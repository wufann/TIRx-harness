#!/usr/bin/env python3
"""Standalone benchmark for MXFP4 GEMM candidates (ROCm, gfx950)."""

from __future__ import annotations

import torch

from ...benchmark_common import (
    BenchConfig,
    choose_device,
    config_from_env,
    load_candidate_override,
    load_task_workloads,
    run_benchmark,
)

TASK_NAME = "mxfp4_gemm"

# bf16 output: one bf16 ulp is 2^-8 relative; fp32 accumulation order adds far less.
ATOL = 1.0
RTOL = 1e-2
REQUIRED_MATCHED_RATIO = 0.9999
# E8M0 exponents of the random scales (2^-3 .. 2^3) keep every dot product finite.
SCALE_MIN, SCALE_MAX = 124, 130


def default_config() -> BenchConfig:
    """This task's configuration, read from the environment at call time."""

    return config_from_env(warmup=10, iters=50, trials=3)


def make_workloads(config: BenchConfig | None = None):
    cfg = config or default_config()
    official = []
    if cfg.include_official:
        for row in load_task_workloads(
            TASK_NAME, cfg.max_official, shape_mode=cfg.shape_mode, shape_axes=("M", "N", "K")
        ):
            workload = row["workload"]
            official.append(
                {
                    "suite": "official",
                    "id": workload["uuid"],
                    "axes": workload["axes"],
                    "raw": workload,
                }
            )
    return official


def make_inputs(entry, device):
    axes = entry["axes"]
    m, n, k = axes["M"], axes["N"], axes["K"]
    gen = torch.Generator(device=device).manual_seed(0)

    def codes(rows):
        return torch.randint(
            0, 256, (rows, k // 2), dtype=torch.uint8, device=device, generator=gen
        )

    def scales(rows):
        return torch.randint(
            SCALE_MIN,
            SCALE_MAX + 1,
            (rows, k // 32),
            dtype=torch.uint8,
            device=device,
            generator=gen,
        )

    return [codes(m), scales(m), codes(n), scales(n)]


def perturb_activations(bound_args, seed=None):
    """The input-dependence gate: the activations A / SA (weights may be repacked in setup).

    Called with ``seed`` it overwrites them in place with fresh codes and in-range scales;
    without, it names the tensors it would change.
    """

    a, sa = bound_args[0], bound_args[1]
    if seed is not None:
        gen = torch.Generator(device=a.device).manual_seed(1000 + seed)
        a.copy_(torch.randint(0, 256, a.shape, dtype=torch.uint8, device=a.device, generator=gen))
        sa.copy_(
            torch.randint(
                SCALE_MIN,
                SCALE_MAX + 1,
                sa.shape,
                dtype=torch.uint8,
                device=sa.device,
                generator=gen,
            )
        )
    return [a, sa]


def tirx_prepare(solution_module, a, sa, b, sb):
    """Compile and bind a candidate outside timing."""

    m, n, k = a.shape[0], b.shape[0], a.shape[1] * 2
    data = {
        "A": a,
        "SA": sa,
        "B": b,
        "SB": sb,
        "D": torch.empty((m, n), dtype=torch.bfloat16, device=a.device),
    }
    kernel_fn = solution_module.setup(data, m, n, k)
    return kernel_fn, data


def tirx_run(kernel_fn, data):
    kernel_fn()
    return data["D"]


def _print_aggregate(rows):
    valid = [
        row
        for row in rows
        if row["passed"] and row["baseline_ms"] is not None and row["kernel_ms"] is not None
    ]
    total_baseline_ms = sum(row["baseline_ms"] for row in valid)
    total_kernel_ms = sum(row["kernel_ms"] for row in valid)
    if total_kernel_ms > 0:
        latency = f"{total_kernel_ms * 1000:.1f}us"
        speedup_text = f"{total_baseline_ms / total_kernel_ms:.3f}x"
    else:
        latency = "999999999us"
        speedup_text = "0.000x"
    print()
    print("# Aggregate result")
    print(f"| candidate | candidate ({len(valid)}/{len(rows)} PASS) | {latency} | {speedup_text} |")


def run_suite(
    config: BenchConfig | None = None, candidate_fn=None, workloads=None, candidate_prepare_fn=None
):
    """Run the correctness gate and timing sweep for this task."""

    from . import baseline as baseline_module

    cfg = config or default_config()
    device = choose_device(cfg.device)
    candidate = candidate_fn
    if candidate is None:
        candidate = load_candidate_override(
            "BENCH_MXFP4_GEMM_KERNEL", baseline_module.run_prepared, "mxfp4_gemm_candidate"
        )
        if candidate is baseline_module.run_prepared:
            candidate_prepare_fn = baseline_module.prepare

    rows = run_benchmark(
        name=TASK_NAME,
        workloads=make_workloads(cfg) if workloads is None else workloads,
        make_inputs=make_inputs,
        candidate_fn=candidate,
        candidate_prepare_fn=candidate_prepare_fn,
        baseline_fn=baseline_module.run_prepared,
        baseline_prepare_fn=baseline_module.prepare,
        reference_fn=baseline_module.reference,
        device=device,
        warmup=cfg.warmup,
        iters=cfg.iters,
        trials=cfg.trials,
        atol=ATOL,
        rtol=RTOL,
        required_matched_ratio=REQUIRED_MATCHED_RATIO,
        group_axis="M",
        correctness_runs=cfg.correctness_runs,
        check_after_timing=cfg.check_after_timing,
        require_repeatable_outputs=cfg.require_repeatable_outputs,
        perturb_fn=perturb_activations,
    )
    _print_aggregate(rows)
    return rows


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("a GPU is required")
    run_suite()


if __name__ == "__main__":
    main()

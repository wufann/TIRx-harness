"""ROCm task declarations, prompt rendering, and the MXFP4 reference on CPU."""

from pathlib import Path

import pytest
import torch

from evolution import cli
from evolution.benchmark import adapter
from evolution.preparation.declare import (
    PLATFORM_TOOLSETS,
    load_task,
    load_toolset,
    task_path,
    toolset_path,
)

adapter._bench_repo_on_path()
from flashinfer_bench_evolve.benchmark_common import BenchConfig  # noqa: E402
from flashinfer_bench_evolve.tasks.mxfp4_gemm import baseline  # noqa: E402

REPO_ROOT = Path(cli.__file__).resolve().parents[1]
ROCM_TASKS = ["fp16_gemm_floor_mi355x", "mxfp4_gemm_qwen38_shared_gate_up_m8192"]


@pytest.mark.parametrize("name", ROCM_TASKS)
def test_rocm_task_declaration(name):
    task = load_task(task_path(REPO_ROOT, name))
    assert task.platform == "rocm"
    assert task.kernel_authoring == "TIRx-amdgcn"
    assert adapter.workload_key(task.workload_dir) in adapter.PACKAGED
    toolset = load_toolset(toolset_path(REPO_ROOT, PLATFORM_TOOLSETS[task.platform]))
    assert all(skill.startswith("tirx-rocm-") for skill in toolset.skills)
    for skill in toolset.skills:
        assert (REPO_ROOT / "skills" / skill / "SKILL.md").is_file()


def test_authoring_must_match_platform(tmp_path):
    source = task_path(REPO_ROOT, ROCM_TASKS[0]).read_text()
    path = tmp_path / "bad.yaml"
    path.write_text(source.replace("kernel_authoring: TIRx-amdgcn", "kernel_authoring: TIRx-lite"))
    with pytest.raises(ValueError, match="does not target rocm"):
        load_task(path)


@pytest.mark.parametrize("name", ROCM_TASKS)
def test_rocm_prompt(name, tmp_path):
    task = load_task(task_path(REPO_ROOT, name))
    toolset = load_toolset(toolset_path(REPO_ROOT, "rocm_flow"))
    prompt = cli._prompt(task=task, toolset=toolset, worktree=tmp_path)
    assert "HIP_VISIBLE_DEVICES=<picked>" in prompt
    assert "rocm-smi" in prompt
    assert "CUDA_VISIBLE_DEVICES" not in prompt
    assert "nvidia-smi" not in prompt
    assert "T.amdgcn" in prompt and "tirx-rocm-wiki" in prompt
    assert "{{" not in prompt


def test_rocm_remote_is_rejected():
    with pytest.raises(ValueError, match="ROCm"):
        cli.setup(task_name=ROCM_TASKS[0], remote_url="http://127.0.0.1:1", run_name="x")


def test_mxfp4_pinned_workload():
    bench = adapter.import_benchmark("mxfp4_gemm")
    rows = adapter.select_workloads(bench, BenchConfig(shape_mode="pinned"))
    assert [row["axes"] for row in rows] == [{"M": 8192, "N": 1280, "K": 2560}]


def test_mxfp4_dequantize_layout():
    # Row 0: elements 0..3 = codes 1 (0.5), 15 (-6), 7 (6), 8 (-0); all scales 2^1.
    packed = torch.zeros(1, 16, dtype=torch.uint8)
    packed[0, 0] = 0xF1
    packed[0, 1] = 0x87
    scales = torch.full((1, 1), 128, dtype=torch.uint8)
    values = baseline.dequantize(packed, scales)
    assert values.shape == (1, 32)
    assert values[0, :4].tolist() == [1.0, -12.0, 12.0, 0.0]
    assert values[0, 4:].abs().sum() == 0


def test_mxfp4_reference_matches_float_matmul():
    gen = torch.Generator().manual_seed(0)
    a = torch.randint(0, 256, (32, 64), dtype=torch.uint8, generator=gen)
    b = torch.randint(0, 256, (16, 64), dtype=torch.uint8, generator=gen)
    sa = torch.randint(124, 131, (32, 4), dtype=torch.uint8, generator=gen)
    sb = torch.randint(124, 131, (16, 4), dtype=torch.uint8, generator=gen)
    expected = baseline.dequantize(a, sa) @ baseline.dequantize(b, sb).T
    out = baseline.reference(a, sa, b, sb)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "mode, ok", [("honest", True), ("precompute", False), ("weights_copied", True)]
)
def test_mxfp4_input_dependence_covers_integer_activations(mode, ok, monkeypatch):
    """uint8 activations are invisible to the float perturbation; the task's hook covers them."""
    from flashinfer_bench_evolve.benchmark_common import (
        KernelRun,
        clone_args,
        input_dependence_check,
    )

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    bench = adapter.import_benchmark("mxfp4_gemm")
    gen = torch.Generator().manual_seed(0)
    a = torch.randint(0, 256, (32, 64), dtype=torch.uint8, generator=gen)
    b = torch.randint(0, 256, (16, 64), dtype=torch.uint8, generator=gen)
    sa = torch.randint(124, 131, (32, 4), dtype=torch.uint8, generator=gen)
    sb = torch.randint(124, 131, (16, 4), dtype=torch.uint8, generator=gen)
    bound = clone_args((a, sa, b, sb))

    def setup(a, sa, b, sb):
        if mode == "weights_copied":  # repacking weights in setup is allowed
            b, sb = b.clone(), sb.clone()
        d, stored = torch.empty(32, 16, dtype=torch.bfloat16), baseline.reference(a, sa, b, sb)

        def run():
            d.copy_(stored if mode == "precompute" else baseline.reference(a, sa, b, sb))

        return run, {"A": a, "SA": sa, "B": b, "SB": sb, "D": d}

    candidate = KernelRun(lambda run, data: (run(), data["D"])[1], tuple(setup(*bound)))
    candidate.run()
    kwargs = dict(
        candidate_runner=candidate,
        bound_args=bound,
        correctness_fn=baseline.reference,
        correctness_prepare=None,
        device=torch.device("cpu"),
        atol=bench.ATOL,
        rtol=bench.RTOL,
        required_matched_ratio=bench.REQUIRED_MATCHED_RATIO,
        required_rms_error_ratios=None,
        phase="input-dependence trial 1",
        seed=1,
    )
    assert input_dependence_check(**kwargs) is None  # the float-only default sees nothing
    result = input_dependence_check(**kwargs, perturb_fn=bench.perturb_activations)
    assert result is not None and result.ok == ok

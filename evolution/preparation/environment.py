"""Create a worktree-local venv and remove task-banned kernel sources."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

from .guards import CANONICAL_KERNELS_SOURCE_DIR, _sanitize_banned_paths

# .pth files of the launcher environment that must not reach a ROCm run venv: they
# would put an unsanitized tirx_kernels source tree on sys.path.
_ROCM_SKIPPED_PTH_PREFIX = "tirx_kernels"


@dataclass(frozen=True)
class Environment:
    prefix: Path
    python: Path
    site_packages: Path
    kernels: Path

    @classmethod
    def create(
        cls, worktree: Path, *, platform: str = "cuda", kernels_source: Path | None = None
    ) -> Environment:
        """Create a fresh venv and install the worktree's runtime packages."""
        prefix = worktree / ".venv"
        if prefix.exists():
            raise ValueError(f"Run venv already exists: {prefix}")
        if platform == "rocm":
            return cls._create_rocm(prefix, kernels_source)
        python = prefix / "bin/python"
        # `uv run` exports the launcher's venv; the run venv is selected explicitly.
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        subprocess.run(
            [
                "uv",
                "sync",
                "--locked",
                "--group",
                "benchmark",
                "--no-editable",
                "--link-mode",
                "copy",
                "--python",
                sys.executable,
            ],
            cwd=worktree,
            env={**env, "UV_PROJECT_ENVIRONMENT": str(prefix)},
            check=True,
        )
        # -I ignores PYTHONPATH and the working directory. Discover packages
        # without importing their CUDA/compiler dependencies during preparation.
        result = subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import importlib.util, json, sysconfig; "
                "print(json.dumps({'site': sysconfig.get_path('purelib'), "
                "'packages': {n: getattr(importlib.util.find_spec(n), 'origin', None) "
                "for n in ('tirx_kernels', 'tirx_harness')}}))",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        paths = json.loads(result.stdout)
        site = Path(paths["site"]).resolve()
        if not site.is_relative_to(prefix):
            raise ValueError("The venv's site-packages must be inside its own directory.")
        packages = {}
        for name, origin in paths["packages"].items():
            if not origin or not Path(origin).resolve().is_relative_to(site):
                raise ValueError(f"{name} was not installed into the run venv.")
            packages[name] = Path(origin).resolve().parent
        return cls(prefix, python, site, packages["tirx_kernels"])

    @classmethod
    def _create_rocm(cls, prefix: Path, kernels_source: Path | None) -> Environment:
        """A venv layered on the launcher's ROCm environment.

        The locked uv groups carry CUDA wheels, so a ROCm run reuses what the launcher
        already runs on: the container image's torch / aiter and the locally built
        tvm-tirx. The launcher's .pth files are replicated (editable tvm-tirx, the
        image site-packages chain) and its site-packages directory is appended as a
        plain path, ahead of the image chain, so its tvm-ffi shadows the image's. The
        kernel package is copied from ``kernels_source`` into the venv, where task bans
        can remove files; a .pth pointing at that source tree is never replicated.
        tirx_harness (CUDA analysis tools) is not installed.
        """
        if kernels_source is None or not (kernels_source / "tirx_kernels").is_dir():
            raise ValueError(f"ROCm runs need a TIRx-kernels checkout, got {kernels_source}")
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(prefix)], check=True)
        python = prefix / "bin/python"
        site = Path(
            subprocess.run(
                [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ).resolve()
        launcher_site = Path(sysconfig.get_path("purelib")).resolve()
        for entry in sorted(launcher_site.iterdir()):
            if entry.name.startswith(_ROCM_SKIPPED_PTH_PREFIX):
                continue
            if entry.suffix == ".pth" or (
                entry.name.startswith("_editable") and entry.suffix == ".py"
            ):
                shutil.copy2(entry, site / entry.name)
        # Sorts before the image chain (zz_*.pth): plain path, its own .pth files unread.
        (site / "zy_launcher_site.pth").write_text(f"{launcher_site}\n")
        shutil.copytree(
            kernels_source / "tirx_kernels",
            site / "tirx_kernels",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        return cls(prefix, python, site, site / "tirx_kernels")

    def kernel_bans(self, banned_paths: list[str]) -> list[str]:
        """Translate checkout-relative task restrictions to installed paths."""
        # Task declarations retain checkout-relative kernel paths. Translate
        # them to the pip-installed package, without touching another venv.
        package_bans = []
        for pattern in banned_paths:
            normalized = pattern.removeprefix("./")
            if normalized in (
                str(CANONICAL_KERNELS_SOURCE_DIR),
                f"{CANONICAL_KERNELS_SOURCE_DIR}/**",
            ):
                normalized = "tirx_kernels/**"
            else:
                normalized = normalized.removeprefix(f"{CANONICAL_KERNELS_SOURCE_DIR}/")
            if normalized == "tirx_kernels" or normalized.startswith("tirx_kernels/"):
                package_bans.append(str(self.site_packages / normalized))
        return package_bans

    def sanitize(self, banned_paths: list[str]) -> None:
        """Remove task-banned kernel sources and the wheel's skill copies from this venv."""
        _sanitize_banned_paths(self.site_packages, self.kernel_bans(banned_paths))
        # Runs install skills from the worktree; leave no unsanitized second copy.
        shutil.rmtree(self.site_packages / "tirx_harness" / "_skills", ignore_errors=True)

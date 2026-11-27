#!/usr/bin/env python3
"""Fetch the CDNA4 ISA reference guide and extract it to grep-able text.

Writes references/manuals/cdna4_isa.txt next to this script and prints its path. The
PDF and the text are cached (default ~/.cache/tirx-rocm-wiki) so later runs on the same
machine need no network. Text extraction uses ``pdftotext -layout`` when available,
otherwise pypdf (installed into the cache directory on first use).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUALS = ROOT / "references" / "manuals"
ISA_URL = (
    "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/"
    "instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf"
)
ISA_NAME = "cdna4_isa"


def _extract(pdf: Path, text: Path, cache: Path) -> None:
    if shutil.which("pdftotext"):
        subprocess.run(["pdftotext", "-layout", str(pdf), str(text)], check=True)
        return
    vendor = cache / "pypdf"
    if not (vendor / "pypdf").is_dir():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--target", str(vendor), "pypdf"],
            check=True,
        )
    sys.path.insert(0, str(vendor))
    from pypdf import PdfReader

    pages = PdfReader(str(pdf)).pages
    text.write_text("\n\f".join(page.extract_text() or "" for page in pages), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    parser.add_argument("--cache-dir", type=Path, default=default_cache / "tirx-rocm-wiki")
    args = parser.parse_args()

    cache = args.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    cached_text = cache / f"{ISA_NAME}.txt"
    if not cached_text.is_file():
        pdf = cache / f"{ISA_NAME}.pdf"
        if not pdf.is_file():
            partial = pdf.with_suffix(".part")
            with urllib.request.urlopen(ISA_URL, timeout=120) as response:
                partial.write_bytes(response.read())
            partial.rename(pdf)
        partial = cached_text.with_suffix(".part")
        _extract(pdf, partial, cache)
        partial.rename(cached_text)

    MANUALS.mkdir(parents=True, exist_ok=True)
    target = MANUALS / f"{ISA_NAME}.txt"
    shutil.copyfile(cached_text, target)
    print(target.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

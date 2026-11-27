"""MFMA issue cadence from an ATT per-wave timeline (rocprofv3 ui_output_*/se*_wv*.json).

For one wave: the interval between consecutive MFMA issues and what was issued in between,
over the steady-state middle of the kernel. Usage:
  python att_timeline.py NAME=ui_output_dir [NAME=dir ...] [--wave se0_sm0_sl0_wv0] [--show 40]
"""

import argparse
import collections
import itertools
import json
import os
import re


def klass(t):
    for name, pat in [
        ("MFMA", r"^v_mfma"),
        ("DS", r"^ds_"),
        ("DMA", r"^(buffer|global)_load.*lds"),
        ("VMEM", r"^(buffer|global)_"),
        ("WAIT", r"^s_waitcnt"),
        ("BAR", r"^s_barrier"),
        ("PRIO", r"^s_setprio"),
        ("NOP", r"^s_nop"),
        ("ACC", r"^v_accvgpr"),
        ("SALU", r"^s_"),
        ("VALU", r"^v_"),
    ]:
        if re.search(pat, t):
            return name
    return "OTHER"


def load(d, wave):
    code = json.load(open(os.path.join(d, "code.json")))["code"]
    text = {c[2]: c[0] for c in code}
    w = json.load(open(os.path.join(d, wave + ".json")))["wave"]
    return [(i[0], text.get(i[-1], "?")) for i in w["instructions"]]


def analyze(name, seq, show):
    n = len(seq)
    mid = seq[n // 4 : 3 * n // 4]  # steady state
    mf = [k for k, (_, t) in enumerate(mid) if t.startswith("v_mfma")]
    gaps = collections.Counter()
    between = collections.Counter()
    tot_gap = 0
    for a, b in itertools.pairwise(mf):
        g = mid[b][0] - mid[a][0]
        tot_gap += g
        gaps[min(g, 64) // 4 * 4] += 1
        kinds = tuple(sorted({klass(t) for _, t in mid[a + 1 : b]}))
        between[kinds] += g
    cyc = mid[-1][0] - mid[0][0]
    print(
        f"== {name}: {len(mf)} MFMAs in {cyc} cycles -> {cyc / max(len(mf), 1):.2f} cycles/MFMA "
        f"(mean MFMA->MFMA gap {tot_gap / max(len(mf) - 1, 1):.2f})"
    )
    print(
        "   gap histogram (cycles, bucket of 4): "
        + ", ".join(f"{k}:{v}" for k, v in sorted(gaps.items()))
    )
    print("   cycles between MFMAs, by what was issued in between:")
    for kinds, c in between.most_common(8):
        print(f"     {c / max(cyc, 1) * 100:5.1f}%  {'+'.join(kinds) or '(nothing)'}")
    if show:
        start = mf[len(mf) // 2]
        print(f"   timeline sample from MFMA #{len(mf) // 2}:")
        t0 = mid[start][0]
        for t, txt in mid[start : start + show]:
            print(f"     +{t - t0:5d}  {txt[:80]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--wave", default="se0_sm0_sl0_wv0")
    p.add_argument("--show", type=int, default=0)
    args = p.parse_args()
    for item in args.inputs:
        name, d = item.split("=", 1)
        analyze(name, load(d, args.wave), args.show)


if __name__ == "__main__":
    main()

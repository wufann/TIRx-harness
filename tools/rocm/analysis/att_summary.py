"""Summarize rocprofv3 ATT per-instruction stats (stats_*.csv) by instruction class.

Usage: python att_summary.py NAME=stats.csv [NAME=stats.csv ...] [--top N]
Latency = cycles from issue to next issue of the wave, Stall = cycles the instruction
waited to issue (dependency / waitcnt / barrier / pipe busy), Idle = wave had nothing to do.
"""

import argparse
import csv
import re
from collections import defaultdict

CLASSES = [
    ("mfma", r"^v_mfma"),
    ("ds_read", r"^ds_read"),
    ("ds_write", r"^ds_write"),
    ("dma(global_load_lds)", r"^global_load_lds"),
    ("buffer_load(lds)", r"^buffer_load.*\blds\b"),
    ("buffer_load", r"^buffer_load"),
    ("global_load", r"^global_load"),
    ("store", r"^(global|buffer)_store"),
    ("s_waitcnt", r"^s_waitcnt"),
    ("s_barrier", r"^s_barrier"),
    ("s_setprio", r"^s_setprio"),
    ("s_nop", r"^s_nop"),
    ("branch", r"^s_(c)?branch"),
    ("accvgpr", r"^v_accvgpr"),
    ("salu", r"^s_"),
    ("valu", r"^v_"),
]


def klass(inst):
    for name, pat in CLASSES:
        if re.search(pat, inst):
            return name
    return "other"


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        inst = r["Instruction"].strip()
        if inst.startswith(";") or not inst:
            continue
        rows.append((inst, int(r["Hitcount"]), int(r["Latency"]), int(r["Stall"]), int(r["Idle"])))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args()
    data = {}
    for item in args.inputs:
        name, path = item.split("=", 1)
        data[name] = load(path)
    names = list(data)
    agg = {n: defaultdict(lambda: [0, 0, 0, 0]) for n in names}
    for n, rows in data.items():
        for inst, hit, lat, stall, idle in rows:
            a = agg[n][klass(inst)]
            a[0] += hit
            a[1] += lat
            a[2] += stall
            a[3] += idle
    tot = {n: [sum(v[i] for v in agg[n].values()) for i in range(4)] for n in names}
    print("per class: hits / latency% / stall% / idle%   (of each kernel's total latency)")
    print(f"{'class':24s}" + "".join(f"{n:>44s}" for n in names))
    keys = sorted(
        {k for n in names for k in agg[n]}, key=lambda k: -max(agg[n][k][1] for n in names)
    )
    for k in keys:
        cells = []
        for n in names:
            h, lat, st, idl = agg[n][k]
            T = tot[n][1] or 1
            cells.append(
                f"{h:10d} {lat / T * 100:7.1f}% {st / T * 100:7.1f}% {idl / T * 100:7.1f}%"
            )
        print(f"{k:24s}" + "".join(f"{c:>44s}" for c in cells))
    print(f"{'TOTAL latency cycles':24s}" + "".join(f"{tot[n][1]:>44d}" for n in names))
    for n in names:
        print(f"\n== {n}: top {args.top} instructions by stall")
        for inst, hit, lat, stall, idle in sorted(data[n], key=lambda r: -r[3])[: args.top]:
            print(f"  stall {stall:9d} lat {lat:9d} hits {hit:6d}  {inst[:90]}")


if __name__ == "__main__":
    main()

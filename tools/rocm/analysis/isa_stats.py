"""Instruction statistics of AMDGPU code objects / assembly (runs inside the dev container).

For each kernel: instruction-family histogram, MFMA variants, memory ops, waitcnt
forms, scheduling hints, and the hottest loop body (the largest backward branch range).

Usage: python isa_stats.py FILE [FILE ...] [--kernel REGEX] [--loop-dump N] [--json OUT]
  FILE: .co/.hsaco/.o (disassembled with llvm-objdump) or .s (assembly text)
"""

# ruff: noqa: E741

import argparse
import collections
import json
import re
import subprocess

OBJDUMP = "/opt/rocm/llvm/bin/llvm-objdump"

FAMILIES = [
    ("mfma", r"^v_mfma"),
    ("smfmac", r"^v_smfmac"),
    ("buffer_load_lds", r"^buffer_load_\w+.*\blds\b"),
    ("global_load_lds", r"^global_load_lds"),
    ("buffer_load", r"^buffer_load"),
    ("buffer_store", r"^buffer_store"),
    ("global_load", r"^global_load"),
    ("global_store", r"^global_store"),
    ("ds_read_tr", r"^ds_read_b64_tr|^ds_read_b96_tr|^ds_read_\w*tr"),
    ("ds_read", r"^ds_read"),
    ("ds_write", r"^ds_write"),
    ("ds_other", r"^ds_"),
    ("s_waitcnt", r"^s_waitcnt"),
    ("s_barrier", r"^s_barrier"),
    ("sched_barrier", r"^s_sched_barrier|^s_sched_group_barrier|^;\s*sched"),
    ("s_setprio", r"^s_setprio"),
    ("s_nop", r"^s_nop"),
    ("v_cvt", r"^v_cvt"),
    ("v_perm/pack", r"^v_perm|^v_pack|^v_alignbit|^v_lshl_or|^v_and_or|^v_bfi"),
    ("dpp/permlane", r"_dpp|^v_permlane|^ds_bpermute|^ds_swizzle"),
    ("v_accvgpr", r"^v_accvgpr"),
    ("valu", r"^v_"),
    ("salu", r"^s_"),
]


def disasm(path, symbols=None):
    if path.endswith(".s"):
        return open(path).read()
    cmd = [OBJDUMP, "-d", "--no-show-raw-insn", "--mcpu=gfx950", path]
    if not symbols:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    # Huge libraries (hipBLASLt: ~4500 kernels, 800+ MB): disassemble only the address
    # ranges of these FUNC symbols (labels inside are symbols too, so
    # --disassemble-symbols would stop at the first one).
    out = subprocess.run(
        [OBJDUMP.replace("objdump", "readelf"), "-sW", path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    funcs = []
    for l in out.splitlines():
        f = l.split()
        if len(f) >= 8 and f[3] == "FUNC":
            funcs.append((int(f[1], 16), int(f[2]), f[-1]))
    starts = sorted({a for a, _, _ in funcs})
    ranges = {}
    for addr, size, name in funcs:
        if name in symbols:
            if size == 0:  # Tensile kernels carry no symbol size: run to the next kernel
                nxt = [a for a in starts if a > addr]
                size = (nxt[0] - addr) if nxt else 1 << 20
            ranges[name] = (addr, size)
    missing = set(symbols) - set(ranges)
    if missing:
        raise SystemExit(f"symbols not found: {sorted(missing)[:3]}")
    text = []
    for start, size in ranges.values():
        text.append(
            subprocess.run(
                [*cmd, f"--start-address={start:#x}", f"--stop-address={start + size:#x}"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
    return "\n".join(text)


def func_symbols(path):
    """Names of FUNC symbols (kernels); other symbols in .text are just labels."""
    if path.endswith(".s"):
        return None
    out = subprocess.run(
        [OBJDUMP.replace("objdump", "readelf"), "-sW", path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {l.split()[-1] for l in out.splitlines() if " FUNC " in l}


def split_kernels(text, funcs=None):
    """Map kernel name -> list of (addr, mnemonic, line). Labels become '<label>' entries."""
    kernels, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"^([0-9a-f]+) <([^>]+)>:\s*$", line) or re.match(
            r"^()([A-Za-z_.$][\w.$]*):\s*(;.*)?$", line
        )
        if m:
            name = m.group(2)
            is_label = name.startswith(".L") or (funcs is not None and name not in funcs)
            if not is_label:
                cur = kernels.setdefault(name, [])
                continue
            if cur is not None:
                cur.append((int(m.group(1), 16) if m.group(1) else None, "<label>", name))
            continue
        if cur is None:
            continue
        s = line.strip()
        if not s or s.startswith(("//", ";", ".")):
            continue
        s = re.sub(r"\s*//.*$", "", s)
        addr = None
        ma = re.match(r"^([0-9a-f]+):\s+(.*)$", s)
        if ma:
            addr, s = int(ma.group(1), 16), ma.group(2)
        mnem = s.split()[0]
        cur.append((addr, mnem, s))
    return {k: v for k, v in kernels.items() if any(m != "<label>" for _, m, _ in v)}


def family(mnem, line):
    for fam, pat in FAMILIES:
        if re.search(pat, line if "lds" in pat else mnem):
            return fam
    return "other"


def hottest_loop(insts):
    """Largest backward branch range (by instruction count)."""
    addr_idx, label_idx = {}, {}
    for i, (a, m, line) in enumerate(insts):
        if a is not None:
            addr_idx.setdefault(a, i)
        if m == "<label>":
            label_idx[line] = i
    best = None
    for i, (a, m, line) in enumerate(insts):
        if not (m.startswith("s_cbranch") or m == "s_branch"):
            continue
        mt = re.search(r"<([^+>]+)(?:\+0x([0-9a-f]+))?>", line)
        j = None
        if mt and mt.group(1) in label_idx and not mt.group(2):
            j = label_idx[mt.group(1)]
        elif mt and mt.group(2) is not None:
            base = next((ia for ia, im, il in insts if im == "<label>" and il == mt.group(1)), None)
            base = insts[0][0] if base is None else base
            j = addr_idx.get(base + int(mt.group(2), 16)) if base is not None else None
        else:
            ml = re.search(r"(\.L\w+|\bL\w+|\blabel_\w+)\b", line)
            j = label_idx.get(ml.group(1)) if ml else None
        if j is not None and j < i and (best is None or i - j > best[1] - best[0]):
            best = (j, i)
    return best


def analyze(insts):
    fam = collections.Counter()
    mfma, mem, waits, hints = (
        collections.Counter(),
        collections.Counter(),
        collections.Counter(),
        collections.Counter(),
    )
    for _, m, line in insts:
        if m == "<label>":
            continue
        f = family(m, line)
        fam[f] += 1
        if f in ("mfma", "smfmac"):
            mfma[m] += 1
        elif f.startswith(("buffer", "global", "ds_")):
            mem[m + (" lds" if re.search(r"\blds\b", line) else "")] += 1
        elif f == "s_waitcnt":
            waits[re.sub(r"\s+", " ", line)] += 1
        elif f in ("sched_barrier", "s_setprio", "s_barrier"):
            hints[re.sub(r"\s+", " ", line)] += 1
    return {
        "n": sum(fam.values()),
        "families": dict(fam.most_common()),
        "mfma": dict(mfma.most_common()),
        "mem": dict(mem.most_common(20)),
        "waitcnt": dict(waits.most_common(12)),
        "sched": dict(hints.most_common(12)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--kernel", default=".")
    p.add_argument(
        "--loop-dump", type=int, default=0, help="print up to N lines of the hottest loop"
    )
    p.add_argument("--json")
    p.add_argument("--symbols", nargs="*", help="disassemble only these symbols (.co input)")
    args = p.parse_args()
    report = {}
    for path in args.files:
        for name, insts in split_kernels(disasm(path, args.symbols), func_symbols(path)).items():
            if not re.search(args.kernel, name):
                continue
            r = {"file": path, "all": analyze(insts)}
            loop = hottest_loop(insts)
            if loop:
                body = insts[loop[0] : loop[1] + 1]
                r["loop"] = analyze(body)
                r["loop_range"] = [loop[0], loop[1]]
                if args.loop_dump:
                    r["loop_text"] = [l for _, _, l in body[: args.loop_dump]]
            report[name] = r
            short = name if len(name) < 90 else name[:87] + "..."
            print(f"=== {short}  ({path.rsplit('/', 1)[-1]})")
            for scope in ("all", "loop"):
                if scope not in r:
                    continue
                a = r[scope]
                print(f"  [{scope}] {a['n']} insts; families: {a['families']}")
                print(f"  [{scope}] mfma: {a['mfma']}")
                print(f"  [{scope}] mem: {a['mem']}")
                print(f"  [{scope}] waitcnt: {a['waitcnt']}")
                print(f"  [{scope}] sched/barrier/prio: {a['sched']}")
            for l in r.get("loop_text", []):
                print("    | " + l)
    if args.json:
        json.dump(report, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()

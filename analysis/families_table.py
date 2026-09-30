"""Table: speedup to matched recall, by index family, at the standard degree.

The family figure shows the whole curve; this is the one number per cell that
the text argues from. Each entry is Vanilla's t@r divided by SYN's at the same
recall, on the MS MARCO hard decile, at the standard construction of
\\S5.1 -- HNSW M=16, NSG and Vamana R=25, IVF 4096 lists.

The three recall targets are read in order because the trend across them is the
point. On the graph families the ratio grows with the target: the deeper into
the tail a search has to go, the more a fragmented target set costs. IVF goes
the other way, 6.8x at 0.90 down to 1.4x at 0.99, because a partition index
reaching recall 0.99 is probing most of its lists on either embedding and
whatever SYN saved at a small nprobe is spent by the time the budget is that
large.

The tag is __own, the run with the finer efSearch grid. The paper's reference
speedup of 10.6x for HNSW comes from Figure 5's run instead, which times the
hard decile alone; the two differ because the grids and the batch sizes differ,
and the paper says so where it first quotes both.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT                                           # noqa: E402

RES = ROOT / "results"
CORPUS = "msmarco-1m"
TAG = "__own"
FAMILIES = [("hnswlib", "HNSW"), ("faiss-nsg", "NSG"),
            ("vamana", "Vamana"), ("faiss-ivf", "IVF")]
TARGETS = (0.90, 0.95, 0.99)


def t_at(family, arm, target, tag=TAG, corpus=CORPUS):
    """Hard-decile search time at which this arm first reaches `target`."""
    p = RES / f"family_{family}_{corpus}__{arm}{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    grid = sorted((r["ms_per_query"], r["recall"])
                  for r in d.get("grid_hard", []) if "recall" in r)
    for ms, recall in grid:
        if recall >= target:
            return ms
    return None


def main(out="families_table.tex"):
    rows, missing = [], []
    for fam, label in FAMILIES:
        cells = []
        for t in TARGETS:
            van, syn = t_at(fam, "vanilla", t), t_at(fam, "syn", t)
            cells.append(van / syn if van and syn else None)
        if all(c is None for c in cells):
            missing.append(label)
            continue
        rows.append((label, cells))
    if not rows:
        print("no family runs at the standard degree")
        return False

    lines = [f"% MS MARCO hard decile, standard degree, tag {TAG}",
             "% each cell is Vanilla t@r / SYN t@r at the same recall"]
    for label, cells in rows:
        body = " & ".join("--" if c is None else f"${c:.1f}\\times$"
                          for c in cells)
        lines.append(f"    {label:<7}& {body} \\\\")
    (RES / out).write_text("\n".join(lines) + "\n")
    print(f"wrote results/{out}")
    head = "".join(f"{f'r{t:.2f}':>10}" for t in TARGETS)
    print(f"  {'family':<8}{head}")
    for label, cells in rows:
        print(f"  {label:<8}" + "".join(
            "        --" if c is None else f"{c:9.1f}x" for c in cells))
    if missing:
        print(f"  !! not measured: {', '.join(missing)}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="families_table.tex")
    main(ap.parse_args().out)

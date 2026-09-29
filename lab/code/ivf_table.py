"""The IVF counterpart of fragmentation, as a table.

phi counts components under graph edges and cannot be asked of a partition
index. The two quantities that play its part are computed from the embedding and
the quantizer alone, before any search runs:

    cells     distinct inverted lists the query's exact top-10 falls into
    nprobe*   rank of the deepest of those lists in the query's own ordering of
              the centroids. Not a proxy for the budget IVF needs -- it IS the
              nprobe at which that query's recall@10 reaches 1.0

The figure drew both as distributions. The table reports the mean of the first
and the median and 90th percentile of the second, which is what the text quotes;
the shape of the distributions is not argued anywhere.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT                                           # noqa: E402

RES = ROOT / "results"
COLLS = [("msmarco-1m", "MS MARCO 1M"), ("fever", "FEVER 1M"),
         ("nq", "NQ 1M"), ("quora", "Quora 523K")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", default="hard")
    a = ap.parse_args()
    p = RES / f"ivf_spread_{a.pop}.json"
    if not p.exists():
        raise SystemExit(f"no {p.name}; run code/ivf_spread.py")
    d = json.loads(p.read_text())

    print(f"IVF, nlist=4096, {a.pop} decile, before any search")
    print(f"  {'collection':14s}{'cells V':>9s}{'cells S':>9s}"
          f"{'nprobe* V':>11s}{'nprobe* S':>11s}{'ratio':>8s}"
          f"{'p90 V':>8s}{'p90 S':>8s}")
    tex = [f"% IVF nlist=4096, {a.pop} decile; computed from the embedding and the",
           "% quantizer alone, with no search. cells: distinct lists holding the",
           "% exact top-10. nprobe*: the nprobe at which recall@10 reaches 1.0."]
    for c, lab in COLLS:
        v, s = d.get(f"{c}__vanilla"), d.get(f"{c}__syn")
        if not (v and s):
            continue
        r = v["nprobe_median"] / s["nprobe_median"] if s["nprobe_median"] else float("nan")
        print(f"  {lab:14s}{v['cells_mean']:9.2f}{s['cells_mean']:9.2f}"
              f"{v['nprobe_median']:11.0f}{s['nprobe_median']:11.0f}{r:7.1f}x"
              f"{v['nprobe_p90']:8.0f}{s['nprobe_p90']:8.0f}")
        tex.append(f"    {lab:14s} & {v['cells_mean']:.2f} & {s['cells_mean']:.2f} & "
                   f"{v['nprobe_median']:.0f} & {s['nprobe_median']:.0f} & "
                   f"{r:.1f}$\\times$ \\\\")
    (RES / "ivf_table.tex").write_text("\n".join(tex) + "\n")
    print(f"\nwrote ivf_table.tex")


if __name__ == "__main__":
    main()

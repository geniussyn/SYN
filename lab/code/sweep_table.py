"""The build-parameter sweep, as a table.

The figure drew six recall-vs-time curves per collection. What the text quotes
from them is two numbers per setting -- the time to reach recall 0.90 and 0.95
on the hard decile -- and one fact the curves show but a ratio cannot: at the
smallest degree the untrained arm does not reach 0.95 at any budget in the grid,
so there is no ratio to report and the cell says what it reached instead.

Times come from each cell's own hard-decile column, which is the same runner and
the same grid for every row, so the rows are comparable to each other. They are
NOT comparable with the strata figure, which times each stratum on its own
queries; that difference is about a tenth and is why the two are never mixed.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT                                           # noqa: E402

RES = ROOT / "results"
CELLS = [(8, 200), (16, 200), (32, 200), (16, 100), (16, 300)]
TARGETS = (0.90, 0.95)


def curve(corpus, arm, M, efc):
    p = RES / f"cell_{corpus}__{arm}__M{M}__own_efc{efc}.json"
    if not p.exists():
        return None
    g = json.loads(p.read_text())["grid"]
    return sorted((r["ms_per_query_hard"], r.get("recall_hard"), r["ef"])
                  for r in g if r.get("recall_hard") is not None)


def at(g, r):
    ok = [x for x in g if x[1] >= r]
    return min(ok, key=lambda x: x[0]) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="msmarco-1m,fever")
    a = ap.parse_args()
    tex = ["% hard decile, each arm against its own exact top-10, times from the",
           "% cell's own hard-decile column. 'n/r' is not reached at any ef in the",
           "% grid; the deepest recall the arm did reach is given instead."]
    for c in a.corpora.split(","):
        print(f"\n{c}  (hard decile)")
        print(f"  {'setting':16s}" + "".join(
            f"{f't@{t:g} V':>10s}{f't@{t:g} S':>10s}{'x':>8s}" for t in TARGETS))
        tex.append(f"% {c}")
        for M, efc in CELLS:
            gv, gs = curve(c, "vanilla", M, efc), curve(c, "syn", M, efc)
            if not (gv and gs):
                continue
            lab = f"M={M}, efC={efc}"
            line, cells = f"  {lab:16s}", []
            for t in TARGETS:
                pv, ps = at(gv, t), at(gs, t)
                if pv and ps:
                    line += f"{pv[0]:10.2f}{ps[0]:10.2f}{pv[0]/ps[0]:7.2f}x"
                    cells += [f"{pv[0]:.2f}", f"{ps[0]:.2f}", f"{pv[0]/ps[0]:.2f}$\\times$"]
                else:
                    top = max(x[1] for x in gv)
                    who = "V" if not pv else "S"
                    line += f"{'n/r':>10s}{(ps[0] if ps else float('nan')):10.2f}" \
                            f"{f'({who} {top:.3f})':>8s}"
                    cells += [f"n/r\\,({top:.3f})",
                              f"{ps[0]:.2f}" if ps else "n/r", "--"]
            print(line)
            tex.append(f"    {lab:16s} & " + " & ".join(cells) + r" \\")
    (RES / "sweep_table.tex").write_text("\n".join(tex) + "\n")
    print(f"\nwrote sweep_table.tex")


if __name__ == "__main__":
    main()

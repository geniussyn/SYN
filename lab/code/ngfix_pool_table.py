"""NGFix given the query pool SYN drew from, swept over how much of it it gets.

The split-half runs manufactured a history out of the evaluation queries because
BEIR ships no query log, which favoured NGFix and still left the objection that
its paper uses millions of them. Both are answered by taking the history from MS
MARCO's BEIR *train* split -- the pool SYN's 500 training queries were sampled
from, leak-filtered by syn/mine.py -- and sweeping over nested prefixes of it. At
500 it is literally the queries SYN was given; at 59,242 it is all of them.

Because the history is a disjoint split, the evaluation is the full query set and
the hard decile the rest of the paper uses, so these numbers sit beside every
other table.

Printed for each budget: the cheapest point on each curve that reaches the target
recall, and the ratio. The comparison is inside one arm -- what the graph repair
buys on that embedding -- because a millisecond from the authors' binary is not a
millisecond from cpp/hnsw_bench.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT                                          # noqa: E402

RES = ROOT / "results"
BUDGETS = [500, 4000, 16000, 59242]
ARMS = ["vanilla", "syn"]
POPS = [("hard", "hard decile"), ("test", "all query")]


def at(rows, r):
    ok = [x for x in rows if x["recall"] >= r]
    return min(ok, key=lambda x: x["ms"]) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="msmarco-1m")
    ap.add_argument("--target", type=float, default=0.90)
    a = ap.parse_args()

    out, any_ = {}, False
    for pop, plab in POPS:
        print(f"\nNGFix over the shared pool — {plab}, recall {a.target:g}")
        print(f"{'arm':9s}{'history':>9s}{'no repair':>12s}{'HNSW-NGFix':>13s}"
              f"{'gain':>8s}{'deepest recall (none / fix)':>30s}")
        for arm in ARMS:
            for n in BUDGETS:
                p = RES / f"ngfix_{a.corpus}__{arm}__pool{n}.json"
                if not p.exists():
                    continue
                any_ = True
                g = json.loads(p.read_text())["grids"]
                b = (g.get("bottom") or {}).get(pop) or []
                f = (g.get("both") or {}).get(pop) or []
                hb, hf = at(b, a.target), at(f, a.target)
                gain = f"{hb['ms']/hf['ms']:7.2f}x" if (hb and hf) else "      --"
                tb = max((x["recall"] for x in b), default=float("nan"))
                tf = max((x["recall"] for x in f), default=float("nan"))
                print(f"{arm:9s}{n:9d}"
                      f"{(hb['ms'] if hb else float('nan')):12.3f}"
                      f"{(hf['ms'] if hf else float('nan')):13.3f}{gain}"
                      f"{tb:18.4f} /{tf:9.4f}")
                out.setdefault(pop, {}).setdefault(arm, {})[str(n)] = {
                    "no_repair_ms": hb["ms"] if hb else None,
                    "ngfix_ms": hf["ms"] if hf else None,
                    "gain": (hb["ms"] / hf["ms"]) if (hb and hf) else None,
                    "no_repair_max_recall": tb, "ngfix_max_recall": tf}
    if not any_:
        print("no pool sweep results; run work/ngfix_pool.sh")
        return
    p = RES / f"ngfix_pool_{a.corpus}_R{a.target:g}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p.name}")


if __name__ == "__main__":
    main()

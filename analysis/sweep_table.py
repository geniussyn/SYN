"""Appendix I: the hyperparameter sweep, as a table.

The review's W6 asks for sensitivity to lambda_pull, tau_p, L and the number of
training queries, and notes that the appendix promising sweeps contains almost
none. The thirteen points were measured; they were never tabulated, because the
figure that would have carried them was cut.

Every column here is INDEX-FREE. phi-hat_1 is the fragmentation of a query's
exact top-k under the best graph a builder with out-degree maxM0 = 2M could
make, so it needs no index and no search; the three quality columns are scored
on the exact top-10, so they need no index and no reranker either. One sweep
point is therefore an adapter and a screen, not an adapter and a build, which is
why thirteen of them exist at all.

That also makes the table do a second job. Because quality and fragmentation are
measured on the same points by the same procedure, the sweep traces the trade
the paper is making rather than asserting it: the correlation between phi-hat_1
and exhaustive MRR across these points says whether lowering fragmentation
always costs ranking, or only does so past some setting.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT                                           # noqa: E402

RES = ROOT / "results"
CORPUS = "msmarco-1m"

# arm suffix, which knob it moves, the value, and the shipped setting for it
POINTS = [
    ("vanilla",      "—",                "untrained",   False),
    ("abl-nopull",   "lambda_pull",      "0 (no pull)", False),
    ("abl-lam0.3",   "lambda_pull",      "0.3",         False),
    ("syn",          "lambda_pull",      "1.0 (ship)",  True),
    ("abl-lam2.0",   "lambda_pull",      "2.0",         False),
    ("abl-lam3.0",   "lambda_pull",      "3.0",         False),
    ("abl-tp0.005",  "tau_p",            "0.005",       False),
    ("abl-tp0.05",   "tau_p",            "0.05",        False),
    ("abl-tp0.1",    "tau_p",            "0.1",         False),
    ("abl-L32",      "L",                "32",          False),
    ("abl-L130",     "L",                "130",         False),
    ("abl-n50",      "n_train",          "50",          False),
    ("abl-n100",     "n_train",          "100",         False),
    ("abl-n250",     "n_train",          "250",         False),
    ("abl-uniform",  "negative source",  "uniform",     False),
]
COLS = [("phi1_hat", "phi1"), ("pr_phi1_hat_eq_1", "P(phi1=1)"),
        ("median_mutual_rank", "med.mr"), ("pair_within_maxM0", "pair<=2M"),
        ("exhaustive_mrr10", "MRR@10"), ("exhaustive_ndcg10", "nDCG@10"),
        ("exhaustive_gold10", "gold@10")]


def main():
    rows = []
    for arm, knob, val, ship in POINTS:
        p = RES / f"screen_{CORPUS}__{arm}.json"
        if not p.exists():
            print(f"  missing {p.name}", flush=True)
            continue
        d = json.loads(p.read_text())
        rows.append((arm, knob, val, ship, d))

    w = max(len(f"{k} = {v}") for _a, k, v, _s, _d in rows)
    head = f"{'setting':{w}s}  " + "  ".join(f"{lab:>9s}" for _k, lab in COLS)
    print(head)
    print("-" * len(head))
    last = None
    for arm, knob, val, ship, d in rows:
        if knob != last and last is not None:
            print()
        last = knob
        name = f"{knob} = {val}" if knob != "—" else val
        cells = "  ".join(f"{d.get(k, float('nan')):9.4f}" for k, _l in COLS)
        print(f"{name:{w}s}  {cells}" + ("   <- shipped" if ship else ""))

    # The trade, over the sweep rather than asserted. Untrained is excluded: it
    # is not a setting of the method, and leaving it in would let one point at
    # the end of both axes manufacture the correlation.
    sw = [d for _a, k, _v, _s, d in rows if k != "—"]
    phi = np.array([d["phi1_hat"] for d in sw])
    for key, lab in (("exhaustive_mrr10", "MRR@10"),
                     ("exhaustive_ndcg10", "nDCG@10"),
                     ("exhaustive_gold10", "gold@10")):
        q = np.array([d[key] for d in sw])
        r = float(np.corrcoef(phi, q)[0, 1])
        print(f"\ncorr(phi1_hat, {lab}) over {len(sw)} sweep points = {r:+.3f}")

    out = RES / "sweep_table.json"
    out.write_text(json.dumps(
        {"corpus": CORPUS, "n_points": len(rows),
         "note": "index-free: phi-hat_1 needs no index, the quality columns are "
                 "scored on the exact top-10 with no reranker",
         "rows": [{"arm": a, "knob": k, "value": v, "shipped": s,
                   **{c: d.get(c) for c, _l in COLS}}
                  for a, k, v, s, d in rows]}, indent=2))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()

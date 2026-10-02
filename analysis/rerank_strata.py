"""The reranking pipeline, read on a difficulty stratum.

`rerank_pipeline.py` runs the expensive stage once, over all queries, and saves
the per-query metrics. Every stratum is then a slice of those arrays rather than
another pass of the cross-encoder -- the reranker sees the same candidate lists
either way, so re-running it on a subset would produce the same numbers at the
cost of an hour.

One thing is not a slice: search latency. A stratum is more expensive per query
than the full set, so `ms_search` is re-read from the timed cell's own
`ms_per_query_hard` column rather than inherited. Rerank latency is per query
and charged unchanged.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, cell_or_strata                           # noqa: E402

RES = ROOT / "results"
# named the way the per-stratum figure names them: the worst 20 / 10 / 5 / 1
STRATA = {"hard20": 0.20, "hard": 0.10, "P95": 0.05, "P99": 0.01}


def stratum_index(corpus, arm, frac):
    """The worst `frac` of queries by the seed-777 selector, this arm's own."""
    z = np.load(RES / f"harddecile_{corpus}__{arm}.npz")
    sel = z["mean_recall"]
    n = max(1, int(round(frac * len(sel))))
    return np.sort(np.argsort(sel, kind="stable")[:n])


def search_ms(cell, ef, stratum, strata=None):
    """Milliseconds for this ef on THIS stratum.

    strata_cell.py timed every stratum on its own queries, so when its file is
    there that is the number. The cell only has two columns -- the whole set and
    the decile -- and using the decile's for the worst 1% understates what that
    stratum costs, by the same gap the per-stratum figure exists to show.
    """
    key = {"hard20": "P80", "hard": "P90", "P95": "P95", "P99": "P99"}.get(stratum)
    if strata and key and key in (strata.get("strata") or {}):
        for r in strata["strata"][key]["grid"]:
            if r["ef"] == ef and "probe_for" not in r:
                return r["ms"]
    key2 = "ms_per_query" if stratum == "all" else "ms_per_query_hard"
    for r in cell["grid"]:
        if r["ef"] == ef and key2 in r:
            return r[key2]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--arms", default="vanilla,syn")
    ap.add_argument("-M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--stratum", choices=sorted(STRATA), default="hard")
    a = ap.parse_args()

    src = RES / f"rerank_pipeline_all_{a.corpus}.json"
    if not src.exists():
        raise SystemExit(f"missing {src.name}; run rerank_pipeline.py first")
    base = json.loads(src.read_text())
    frac = STRATA[a.stratum]
    out = {**{k: v for k, v in base.items() if k != "arms"},
           "stratum": a.stratum, "stratum_fraction": frac,
           "note": f"each arm on its own worst {frac * 100:g}% by the seed-777 "
                   f"selector; metrics sliced from the all-query pass, "
                   f"search ms taken from strata_cell.py's own timing of "
                   f"this stratum where it exists, else the cell's decile column",
           "arms": {}}

    for arm in a.arms.split(","):
        idx = stratum_index(a.corpus, arm, frac)
        pq = np.load(RES / f"rerank_pq_{a.corpus}__{arm}.npz")
        cell = cell_or_strata(a.corpus, arm, a.M, a.efc, results=RES)
        sp = RES / f"strata_{a.corpus}__{arm}__M{a.M}__strata_efc{a.efc}.json"
        strata = json.loads(sp.read_text()) if sp.exists() else None
        rows = []
        for r0 in base["arms"][arm]["rows"]:
            R, ef = r0["R"], r0["ef"]
            ms_s = search_ms(cell, ef, a.stratum, strata)
            row = {"R": R, "ef": ef, "ms_search": ms_s,
                   "ms_rerank": r0["ms_rerank"],
                   "ms_total": (ms_s + r0["ms_rerank"]) if ms_s else None}
            for tag in ("pre", "post"):
                for metric, name in (("ndcg", "ndcg10"), ("mrr", "mrr10"),
                                     ("gold", "gold10")):
                    key = f"{metric}_{tag}_{R}"
                    if key in pq.files:
                        row[f"{name}_{tag}"] = float(pq[key][idx].mean())
            rows.append(row)
        out["arms"][arm] = {"n_queries": int(len(idx)), "rows": rows}
        print(f"[{a.corpus}/{arm}] {a.stratum}: n={len(idx)}", flush=True)

    suffix = "" if a.stratum == "hard" else f"_{a.stratum}"
    p = RES / f"rerank_pipeline{suffix}_{a.corpus}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p.name}", flush=True)


if __name__ == "__main__":
    main()

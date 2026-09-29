"""One index, four difficulty strata, each timed on its own queries.

The hard decile generalises: it is the worst 10% by the selector's ANN recall,
and nothing stops the same rule giving the worst 5% or the worst 1%. Those are
the strata a tail claim is really about, and they are nested, so a single index
and a single set of returned neighbours answers all of them -- but NOT a single
timing. A harder stratum costs more per query (the decile already runs 1.05-1.28x
the mean), and the gap grows as the stratum narrows, so each one is timed on its
own queries here rather than inheriting the decile's milliseconds.

Selection is unchanged: the strata come from the seed-777 selector's per-query
recall, the same array the decile is cut from, so nothing is selected on the
build being measured.
"""
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import EMB, ROOT  # noqa: E402
# LAB comes from paths.ROOT: SYN_ROOT if set, else this file's parent.
# It used to be a fixed absolute path, which resolved only on the machine
# this was written on, where that path was a symlink into the tree.
LAB = ROOT; RES = LAB / "results"
IDX = Path("/dev/shm/synidx")
from index_hnsw import build, timed_grid, timing_lock           # noqa: E402
from metrics import recall_at_k, mrr_at_k, ndcg_at_k, qrels_recall_at_k   # noqa: E402
from run_cell import load_rels                                  # noqa: E402

# Difficulty percentiles, named the standard way: P90 is the tenth of queries
# above the 90th percentile of difficulty, P99 the hundredth above the 99th.
# P90 is the same set the rest of this project calls the hard decile -- same
# rule, same selector, same members -- and is written here under the percentile
# name so the four strata read as one scale.
FRACS = [("all", 1.00), ("P80", 0.20), ("P90", 0.10), ("P95", 0.05), ("P99", 0.01)]
GRID = [16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192,
        224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536,
        2048, 2560, 3072, 4096]
DEEP = [6144, 8192, 12288, 16384, 24576, 32768]
# Recorded next to a point, never used to stop one: a target is measured until
# it is reached, and the only ceiling is ef's own.
MS_NOTE = 64.0
EF_MAX = 131072
# At or above this budget one timed pass already runs far past the
# harness's three-second floor, so the min-of-three that protects a short
# run from scheduling noise buys nothing and costs three times the wall
# clock. The realised count is recorded per row.
DEEP_REPS_FROM = 1024
TARGETS = [0.90, 0.95, 0.99]


def attain(ip, Q, gt, idx, rows, R, tol=0.002, max_probes=14):
    """The measured cost of reaching recall R -- no interpolation, no budget.

    If nothing on the grid reaches R, ef is doubled until something does; then
    the bracket is bisected until the cheapest point at or above R is within
    `tol`. Every probe is executed and timed on the stratum's own queries, so
    what is reported is an ef that was run, the recall it returned and the
    milliseconds it cost. Only a target ef cannot reach at all is unreached.
    """
    def pts():
        return sorted({r["ef"]: (r["ms"], r["recall"]) for r in rows}.items())

    def probe(ef):
        reps = 1 if ef >= DEEP_REPS_FROM else 3
        r = timed_grid(str(ip), Q[idx], [ef], k=10, reps=reps, pin=True)[0]
        lab = r.pop("labels")
        row = {"ef": ef, "ms": r["ms_per_query"], "qps": r["qps"], "reps": reps,
               "recall": float(recall_at_k(lab, gt[idx], 10).mean()),
               "probe_for": R}
        rows.append(row)
        print(f"    probe ef={ef} -> {row['ms']:.3f} ms/q  "
              f"recall {row['recall']:.4f}", flush=True)
        return row

    probes = 0
    while not [1 for _, v in pts() if v[1] >= R]:
        deepest = pts()[-1][0]
        if deepest >= EF_MAX or probes >= max_probes:
            e, (ms, rc) = pts()[-1]
            return {"target": R, "reached": False, "deepest_ef": e,
                    "deepest_ms": ms, "deepest_recall": rc,
                    "stopped_by": "ef_ceiling" if deepest >= EF_MAX
                    else "probe_budget"}
        probe(min(deepest * 2, EF_MAX)); probes += 1

    below = [(e, v) for e, v in pts() if v[1] < R]
    above = [(e, v) for e, v in pts() if v[1] >= R]
    lo = below[-1][0] if below else None
    hi, (hi_ms, hi_rc) = above[0]
    while lo is not None and hi - lo > 1 and hi_rc - R > tol and probes < max_probes:
        mid = (lo + hi) // 2
        row = probe(mid); probes += 1
        if row["recall"] >= R:
            hi, hi_ms, hi_rc = mid, row["ms"], row["recall"]
        else:
            lo = mid
    return {"target": R, "reached": True, "ef": hi, "ms": hi_ms,
            "recall": hi_rc, "overshoot": hi_rc - R, "over_64ms": hi_ms > MS_NOTE,
            "bracket_below_ef": lo, "probes": probes, "interpolated": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("-M", type=int, default=16); ap.add_argument("--efc", type=int, default=200)
    a = ap.parse_args()
    name = f"{a.corpus}__{a.arm}__M{a.M}__strata_efc{a.efc}"

    sel = np.load(RES / f"harddecile_{a.corpus}__{a.arm}.npz")["mean_recall"]
    order = np.argsort(sel, kind="stable")
    Q = np.load(EMB / f"{a.corpus}__{a.arm}_queries.npy").astype(np.float32)
    gt = np.load(RES / f"gt_{a.corpus}__{a.arm}_k10.npz")["idx"]
    rels = load_rels(a.corpus)
    strata = {n: (np.arange(len(Q)) if f == 1.0
                  else np.sort(order[:max(1, int(round(f * len(Q))))]).astype(np.int32))
              for n, f in FRACS}

    IDX.mkdir(parents=True, exist_ok=True)
    ip = IDX / f"{name}.bin"
    print(f"[{name}] build M={a.M} efC={a.efc}", flush=True)
    with timing_lock():          # a 100-thread build spans the reserved core too
        ix, bt, n, d = build(a.corpus, a.arm, a.M, a.efc, seed=100,
                             index_path=str(ip))
    del ix; gc.collect()

    out = {"corpus": a.corpus, "arm": a.arm, "M": a.M, "efConstruction": a.efc,
           "build_time_s": bt, "n_docs": int(n), "n_queries": int(len(Q)),
           "search_threads": 1, "ms_cap": None, "targets": TARGETS,
           "strata_definition": {n: f"worst {f*100:g}% by the seed-777 selector's "
                                    f"mean ANN recall" for n, f in FRACS},
           "P90_is_the_hard_decile": True,
           "note": "each stratum timed on its own queries; strata cut from the "
                   "seed-777 selector, nested, worst-first",
           "strata": {}}
    for nm, _ in FRACS:
        idx = strata[nm]
        g = [e for e in GRID if e >= 10]
        rows = []
        print(f"[{name}] stratum {nm} (n={len(idx)})", flush=True)
        for r in (timed_grid(str(ip), Q[idx], [e for e in g if e < DEEP_REPS_FROM],
                             k=10, reps=3, pin=True)
                  + (timed_grid(str(ip), Q[idx], [e for e in g if e >= DEEP_REPS_FROM],
                                k=10, reps=1, pin=True)
                     if any(e >= DEEP_REPS_FROM for e in g) else [])):
            lab = r.pop("labels")
            rows.append({"ef": r["ef"], "ms": r["ms_per_query"], "qps": r["qps"],
                         "reps": 1 if r["ef"] >= DEEP_REPS_FROM else 3,
                         "recall": float(recall_at_k(lab, gt[idx], 10).mean()),
                         "mrr": float(mrr_at_k(lab, [rels[i] for i in idx], 10).mean()),
                         "ndcg": float(ndcg_at_k(lab, [rels[i] for i in idx], 10).mean()),
                         "qrels_recall": float(qrels_recall_at_k(
                             lab, [rels[i] for i in idx], 10).mean())})
        for e in DEEP:
            r = timed_grid(str(ip), Q[idx], [e], k=10, reps=1, pin=True)[0]
            lab = r.pop("labels")
            rows.append({"ef": e, "ms": r["ms_per_query"], "qps": r["qps"], "reps": 1,
                         "recall": float(recall_at_k(lab, gt[idx], 10).mean()),
                         "mrr": float(mrr_at_k(lab, [rels[i] for i in idx], 10).mean()),
                         "ndcg": float(ndcg_at_k(lab, [rels[i] for i in idx], 10).mean()),
                         "qrels_recall": float(qrels_recall_at_k(
                             lab, [rels[i] for i in idx], 10).mean())})
        tg = []
        for R in TARGETS:
            t = attain(str(ip), Q, gt, idx, rows, R)
            tg.append(t)
            if t["reached"]:
                print(f"  {nm}: recall {R:.2f} measured at ef={t['ef']} -> "
                      f"{t['ms']:.3f} ms (recall {t['recall']:.4f}, "
                      f"{t['probes']} extra probes)", flush=True)
            else:
                print(f"  {nm}: recall {R:.2f} UNREACHED; deepest "
                      f"{t['deepest_recall']:.4f} at {t['deepest_ms']:.1f} ms "
                      f"({t['stopped_by']})", flush=True)
        rows.sort(key=lambda r: r["ef"])
        out["strata"][nm] = {"n": int(len(idx)), "grid": rows, "targets": tg}
        print(f"  {nm}: {len(rows)} points, deepest {rows[-1]['ms']:.1f} ms, "
              f"recall {rows[-1]['recall']:.4f}", flush=True)
    ip.unlink(missing_ok=True)
    (RES / f"strata_{name}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote strata_{name}.json", flush=True)


if __name__ == "__main__":
    main()

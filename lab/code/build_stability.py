"""How much of what we measure is the builder rather than the embedding.

hnswlib inserts in parallel and assigns internal ids under a lock, so the level-0
graph is a function of the build, not of the embedding alone. Every graph-derived
quantity in this paper therefore has a build-to-build component, and the paper
owes the reader its size. This builds the same embedding N times under different
seeds and reports, per quantity, how far it moves:

    phi_1, phi_3   per query, from each build's own level-0 graph
    recall@10      per query, at a few search budgets, on each build

Nothing here is timed, so it does not need the exclusive harness; it does need
the lane, because a 100-thread build beside a timed run is what the timing lock
exists to prevent.

Reported per quantity: the mean and the max over queries of (max - min) across
builds, and for phi the share of queries whose value never changes. Two example
queries are printed separately because the paper names them.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hnsw_io import level0_csr_by_label                              # noqa: E402
from index_hnsw import BUILD_THREADS, build, timing_lock             # noqa: E402
from kappa import phi                                                # noqa: E402
from metrics import recall_at_k                                      # noqa: E402
from paths import EMB, ROOT                                          # noqa: E402

RES = ROOT / "results"
SCRATCH = Path("/dev/shm/stability")


def one_build(corpus, arm, M, efc, seed, qi, T, efs):
    """Build, read out the level-0 graph, and search. Returns phi and recall."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    ip = SCRATCH / f"{corpus}__{arm}__M{M}__efc{efc}__seed{seed}.bin"
    # the build is the heavy part and takes the lane's lock; nothing here is
    # timed, so the search afterwards uses the whole machine
    with timing_lock(True):
        ix, bt, _, _ = build(corpus, arm, M, efc, seed=seed, index_path=str(ip))
    indptr, indices, _ = level0_csr_by_label(str(ip))
    out = {"seed": seed, "build_time_s": bt,
           "phi1": phi(indptr, indices, T, h=1, log_every=0).astype(np.int16),
           "phi3": phi(indptr, indices, T, h=3, log_every=0).astype(np.int16)}
    Q = np.asarray(np.load(EMB / f"{corpus}__{arm}_queries.npy",
                           mmap_mode="r")[qi], np.float32)
    rec = []
    for ef in efs:
        ix.set_ef(ef)
        lab, _ = ix.knn_query(Q, k=10, num_threads=BUILD_THREADS)
        rec.append(recall_at_k(lab.astype(np.int32), T, 10))
    out["recall"] = np.stack(rec)
    ip.unlink(missing_ok=True)
    return out


def spread(vals, name, ints=True):
    """max - min across builds, per query."""
    A = np.stack(vals)                       # (builds, queries)
    rng = A.max(0) - A.min(0)
    d = {"mean_range": float(rng.mean()), "max_range": float(rng.max()),
         "share_identical": float((rng == 0).mean())}
    print(f"  {name:14s} mean range {d['mean_range']:.3f}   "
          f"max {d['max_range']:.0f}   identical on {d['share_identical']:.1%} "
          f"of queries" if ints else
          f"  {name:14s} mean range {d['mean_range']:.4f}   "
          f"max {d['max_range']:.4f}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="msmarco-1m")
    ap.add_argument("--arm", default="vanilla")
    ap.add_argument("-M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--seeds", default="100,101,102")
    ap.add_argument("--efs", default="16,64,128")
    ap.add_argument("--pop", default="hard")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    efs = [int(e) for e in a.efs.split(",")]
    gt = np.load(RES / f"gt_{a.corpus}__{a.arm}_k10.npz")["idx"]
    if a.pop == "all":
        qi = np.arange(len(gt))
    else:
        qi = np.load(RES / f"harddecile_{a.corpus}__{a.arm}.npz")["hard"]
    T = gt[qi].astype(np.int64)
    print(f"{a.corpus} / {a.arm}  M={a.M} efC={a.efc}  "
          f"{len(seeds)} builds, {len(qi)} {a.pop} queries")

    runs = []
    for s in seeds:
        print(f"\n[seed {s}]", flush=True)
        runs.append(one_build(a.corpus, a.arm, a.M, a.efc, s, qi, T, efs))
        r = runs[-1]
        print(f"  phi1 mean {r['phi1'].mean():.2f}  phi3 mean {r['phi3'].mean():.2f}"
              f"  build {r['build_time_s']/60:.1f} min", flush=True)

    print(f"\nacross {len(seeds)} builds, per query:")
    out = {"corpus": a.corpus, "arm": a.arm, "M": a.M, "efc": a.efc,
           "seeds": seeds, "n_queries": int(len(qi)), "pop": a.pop,
           "per_build_means": {
               "phi1": [float(r["phi1"].mean()) for r in runs],
               "phi3": [float(r["phi3"].mean()) for r in runs]},
           "spread": {}}
    out["spread"]["phi1"] = spread([r["phi1"] for r in runs], "phi_1")
    out["spread"]["phi3"] = spread([r["phi3"] for r in runs], "phi_3")
    for j, e in enumerate(efs):
        out["spread"][f"recall@ef{e}"] = spread(
            [r["recall"][j] for r in runs], f"recall@ef{e}", ints=False)

    # the queries the paper names
    qs = [json.loads(l) for l in
          (ROOT / "corpora" / f"{a.corpus}_queries.jsonl").read_text().splitlines()
          if l.strip()]
    pos = {q["qid"]: i for i, q in enumerate(qs)}
    where = {qid: int(np.where(qi == pos[qid])[0][0])
             for qid in ("1015641", "1030451")
             if qid in pos and pos[qid] in set(qi.tolist())}
    if where:
        print("\nnamed queries:")
        out["named"] = {}
        for qid, r in where.items():
            p1 = [int(x["phi1"][r]) for x in runs]
            p3 = [int(x["phi3"][r]) for x in runs]
            rc = {f"ef{e}": [float(x["recall"][j][r]) for x in runs]
                  for j, e in enumerate(efs)}
            print(f"  {qid} {qs[pos[qid]]['text'][:38]!r}")
            print(f"     phi1 {p1}   phi3 {p3}")
            for k, v in rc.items():
                print(f"     recall@{k} {['%.1f' % x for x in v]}")
            out["named"][qid] = {"text": qs[pos[qid]]["text"], "phi1": p1,
                                 "phi3": p3, "recall": rc}

    p = RES / f"build_stability_{a.corpus}_{a.arm}_M{a.M}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p.name}")


if __name__ == "__main__":
    main()

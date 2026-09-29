"""Why the partition index improves too.

phi counts components under graph edges and says nothing about IVF, which has no
edges. The quantity that plays the same role in a partition index is where the
answers sit relative to the partition:

    cells     how many distinct inverted lists the query's exact top-10 falls
              into. One means a single list holds all ten and nprobe=1 finds
              them; ten means ten lists and the search has to open all ten.

    nprobe*   the rank, in the query's own ordering of the centroids, of the
              deepest list holding one of its ten. This is not a proxy -- it is
              exactly the nprobe at which that query's recall@10 reaches 1.0,
              so the distribution of nprobe* is the distribution of the budget
              IVF needs, per query, before any search is run.

Both are properties of the embedding and the partition together, and neither
needs a timed run. The quantizer is retrained here with the parameters
run_family.py used (nlist=4096, 64 training points per centroid, 12 iterations)
and a fixed seed, so it is the same construction rather than the same object.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT                                          # noqa: E402

EMB, RES = ROOT / "emb", ROOT / "results"
NLIST, PPC, NITER, SEED = 4096, 64, 12, 1234


def quantizer(X, nlist=NLIST, seed=SEED):
    """The coarse centroids, as run_family builds them."""
    import faiss
    faiss.omp_set_num_threads(int(__import__("os").environ.get(
        "SYN_BUILD_THREADS", "100")))
    km = faiss.Kmeans(X.shape[1], nlist, niter=NITER, seed=seed, verbose=False,
                      max_points_per_centroid=PPC)
    km.train(np.ascontiguousarray(X, dtype=np.float32))
    return km.centroids.reshape(nlist, -1)


def spread(corpus, arm, pop="hard", k=10, qb=256, xb=200_000):
    import torch
    X = np.load(EMB / f"{corpus}__{arm}_docs.npy", mmap_mode="r")
    Q = np.load(EMB / f"{corpus}__{arm}_queries.npy", mmap_mode="r")
    T = np.load(RES / f"gt_{corpus}__{arm}_k{k}.npz")["idx"]
    if pop != "all":
        z = RES / f"harddecile_{corpus}__{arm}.npz"
        if not z.exists():
            return None
        T = T[np.load(z)["hard"]]
        qi = np.load(z)["hard"]
    else:
        qi = np.arange(len(Q))

    C = quantizer(X)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Ct = torch.from_numpy(np.ascontiguousarray(C, np.float32)).to(dev)
    # which list each document belongs to: nearest centroid by L2, the metric
    # the IVF cells were cut with
    cn = (Ct * Ct).sum(1)
    assign = np.empty(X.shape[0], np.int32)
    for s in range(0, X.shape[0], xb):
        e = min(s + xb, X.shape[0])
        xc = torch.from_numpy(np.ascontiguousarray(X[s:e], np.float32)).to(dev)
        d2 = cn.unsqueeze(0) - 2.0 * (xc @ Ct.T)
        assign[s:e] = d2.argmin(1).cpu().numpy()
        del xc, d2

    cells, need = np.empty(len(T), np.int32), np.empty(len(T), np.int32)
    for s in range(0, len(T), qb):
        e = min(s + qb, len(T))
        q = torch.from_numpy(np.ascontiguousarray(Q[qi[s:e]], np.float32)).to(dev)
        # the query's own ordering of the lists
        order = (cn.unsqueeze(0) - 2.0 * (q @ Ct.T)).argsort(1).cpu().numpy()
        rank = np.empty_like(order)
        rows = np.arange(order.shape[0])[:, None]
        rank[rows, order] = np.arange(order.shape[1])[None, :]
        for i in range(e - s):
            a = assign[T[s + i]]
            cells[s + i] = len(np.unique(a))
            need[s + i] = rank[i, a].max() + 1        # 1-based nprobe
        del q
    return {"n": int(len(T)),
            "cells_mean": float(cells.mean()),
            "cells_counts": np.bincount(cells, minlength=k + 1)[1:k + 1].tolist(),
            "nprobe_mean": float(need.mean()),
            "nprobe_median": float(np.median(need)),
            "nprobe_p90": float(np.percentile(need, 90)),
            "nprobe": need.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="msmarco-1m,fever,nq,quora")
    ap.add_argument("--arms", default="vanilla,syn")
    ap.add_argument("--pop", default="hard")
    a = ap.parse_args()

    p = RES / f"ivf_spread_{a.pop}.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    for c in a.corpora.split(","):
        for arm in a.arms.split(","):
            key = f"{c}__{arm}"
            if key in out:
                continue
            r = spread(c, arm, pop=a.pop)
            if r is None:
                print(f"  {key}: no stratum, skipped", flush=True)
                continue
            out[key] = r
            print(f"  {key:24s} n={r['n']:5d}  cells {r['cells_mean']:5.2f}  "
                  f"nprobe* median {r['nprobe_median']:6.0f}  "
                  f"p90 {r['nprobe_p90']:6.0f}  mean {r['nprobe_mean']:7.1f}",
                  flush=True)
            p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p.name}")


if __name__ == "__main__":
    main()

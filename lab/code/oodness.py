"""How far out of the corpus's own distribution a collection's queries sit.

The claim this paper makes is about a query's top-k being mutually reachable in a
graph built from document-document proximity. That graph knows nothing about
queries, so it is built well exactly where queries land in regions the documents
already agree about, and badly where they do not. "Out of distribution" in the
ANN sense is precisely that gap, and it is measurable without an index:

    drift      |top-k(q) INTERSECT top-k(d1)| / k, where d1 is the query's own
               nearest document. If a query's neighbourhood is also its nearest
               document's neighbourhood, the graph's edges already lead there.
               RoarGraph's diagnostic, and the one that separates the
               text-to-image benchmarks from the in-domain ones.

    reach      the mean over queries of |{pairs of its top-k within rank 2M of
               each other}|, the pair term phi_1 is built from.

    gap        mean sim(q, its nearest document) minus mean sim(d, its nearest
               other document). Negative means queries sit further from the
               corpus than the corpus sits from itself.

All three on the untrained arm, which is the distribution SYN is asked to repair.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT                                          # noqa: E402

EMB, RES = ROOT / "emb", ROOT / "results"


@torch.no_grad()
def topk_and_best(V, X, k, device="cuda", vb=256, xb=200_000):
    """top-k ids and the best similarity, for rows of V against X."""
    n = X.shape[0]
    I = np.empty((len(V), k), np.int64)
    S = np.empty(len(V), np.float32)
    for s in range(0, len(V), vb):
        e = min(s + vb, len(V))
        q = torch.from_numpy(np.ascontiguousarray(V[s:e], np.float32)).to(device)
        bi = torch.zeros((e - s, 0), dtype=torch.long, device=device)
        bs = torch.zeros((e - s, 0), device=device)
        for cs in range(0, n, xb):
            ce = min(cs + xb, n)
            xc = torch.from_numpy(np.ascontiguousarray(X[cs:ce], np.float32)).to(device)
            sim = q @ xc.T
            kk = min(k, ce - cs)
            v, i = torch.topk(sim, kk, dim=1)
            bs = torch.cat([bs, v], 1)
            bi = torch.cat([bi, i + cs], 1)
            if bs.shape[1] > k:
                v, sel = torch.topk(bs, k, dim=1)
                bs, bi = v, torch.gather(bi, 1, sel)
            del xc, sim
        I[s:e] = bi.cpu().numpy()
        S[s:e] = bs[:, 0].cpu().numpy()
    return I, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="msmarco-1m,fever,nq,quora")
    ap.add_argument("--arm", default="vanilla")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--n-queries", type=int, default=1000)
    ap.add_argument("--n-docs", type=int, default=1000,
                    help="documents sampled for the corpus-to-itself reference")
    ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args()

    p = RES / f"oodness_{a.arm}.json"
    # merge: this is a screen, run a few collections at a time as they encode
    out = json.loads(p.read_text()) if p.exists() else {}
    for c in a.corpora.split(","):
        qp = EMB / f"{c}__{a.arm}_queries.npy"
        if not qp.exists():
            print(f"[{c}] not encoded, skipped", flush=True)
            continue
        X = np.load(EMB / f"{c}__{a.arm}_docs.npy", mmap_mode="r")
        Q = np.load(qp, mmap_mode="r")
        rng = np.random.default_rng(a.seed)
        qi = np.sort(rng.choice(len(Q), min(a.n_queries, len(Q)), replace=False))
        di = np.sort(rng.choice(len(X), min(a.n_docs, len(X)), replace=False))

        QI, QS = topk_and_best(np.asarray(Q[qi]), X, a.k)
        # the query's own nearest document, and that document's neighbourhood
        d1 = QI[:, 0]
        DI, _ = topk_and_best(np.asarray(X[d1]), X, a.k + 1)
        drift = np.array([len(set(QI[i]) & set(DI[i][DI[i] != d1[i]][:a.k])) / a.k
                          for i in range(len(qi))])
        # corpus to itself: a document's nearest OTHER document
        SI, _ = topk_and_best(np.asarray(X[di]), X, 2)
        DS = np.array([float(np.asarray(X[di[i]], np.float32)
                             @ np.asarray(X[SI[i][1] if SI[i][0] == di[i]
                                             else SI[i][0]], np.float32))
                       for i in range(len(di))])
        out[c] = {"n_docs": int(X.shape[0]), "n_queries_sampled": int(len(qi)),
                  "drift_mean": float(drift.mean()),
                  "drift_share_zero": float((drift == 0).mean()),
                  "q_to_corpus_sim": float(QS.mean()),
                  "corpus_to_itself_sim": float(DS.mean()),
                  "gap": float(QS.mean() - DS.mean())}
        print(f"  {c:11s} drift {out[c]['drift_mean']:.3f} "
              f"(zero for {out[c]['drift_share_zero']*100:4.1f}%)   "
              f"sim q->corpus {QS.mean():.3f}  corpus->corpus {DS.mean():.3f}  "
              f"gap {out[c]['gap']:+.3f}", flush=True)

    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p.name}")


if __name__ == "__main__":
    main()

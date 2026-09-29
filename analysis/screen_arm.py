"""Screen an arm without building an index.

phi_1 asks how many pieces a query's exact top-k falls into when only direct
edges count. A level-0 HNSW node keeps at most maxM0 = 2M edges, so the most a
builder can do is connect each document to its true top-maxM0. Computing the
components under exactly that rule gives phi-hat_1: the fragmentation the index
would have if its builder were perfect.

That makes it the right screening quantity. It needs no index, so it costs a
corpus scan instead of a build; it is deterministic, so it carries none of the
+-0.124 per-query build noise a real graph does; and it bounds the real thing --
no builder can do better than its own out-degree allows.

The cost side is screened too. An arm can make its target set cohere by choosing
a different, easier target set, so the exhaustive quality of the embedding --
MRR@10 and qrels recall@10 with no index at all -- is reported next to it. A
geometry win paid for with the encoder's ranking is not a win.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import mrr_at_k, ndcg_at_k, qrels_recall_at_k        # noqa: E402
from paths import ROOT                                            # noqa: E402

EMB, RES = ROOT / "emb", ROOT / "results"


def components(adj):
    """Number of connected components of a k x k boolean adjacency."""
    k = adj.shape[0]
    seen, n = np.zeros(k, bool), 0
    for s in range(k):
        if seen[s]:
            continue
        n += 1
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(adj[u] & ~seen):
                seen[v] = True
                stack.append(v)
    return n


@torch.no_grad()
def mutual_ranks(X, T, device="cuda", anchor_batch=512, doc_chunk=200_000):
    """W[q,i,j] = min(rank_i(j), rank_j(i)) over each query's target set.

    rank_u(v) = |{w : sim(u,w) > sim(u,v)}| counts u itself, so it is 1-based
    excluding self. Only the pairs that share a target set are needed, so each
    anchor carries a padded partner list and one corpus scan answers all of its
    thresholds at once: torch.searchsorted turns the comparisons into a log-time
    lookup, and a reverse cumsum over the histogram gives the counts above each
    threshold. fp32 throughout -- these ranks are the reference.
    """
    nq, k = T.shape
    ids, inv = np.unique(T.reshape(-1), return_inverse=True)
    inv = inv.reshape(nq, k)
    na, n = ids.size, X.shape[0]

    part = [set() for _ in range(na)]
    for q in range(nq):
        row = inv[q]
        for i in range(k):
            for j in range(k):
                if i != j:
                    part[int(row[i])].add(int(row[j]))
    P = max(len(s_) for s_ in part)
    pad = np.zeros((na, P), np.int64)
    slot = [dict() for _ in range(na)]
    for i, ss in enumerate(part):
        for j, v in enumerate(sorted(ss)):
            pad[i, j] = ids[v]
            slot[i][v] = j
        if len(ss) < P:
            pad[i, len(ss):] = pad[i, 0]

    A = torch.from_numpy(np.array(X[ids], copy=True)).to(device).float()
    ranks = np.zeros((na, P), np.int64)
    for s0 in range(0, na, anchor_batch):
        e0 = min(s0 + anchor_batch, na)
        B = e0 - s0
        Ab = A[s0:e0]
        Pv = torch.from_numpy(np.array(X[pad[s0:e0].reshape(-1)], copy=True)
                              ).to(device).float().reshape(B, P, -1)
        thr = torch.bmm(Ab.unsqueeze(1), Pv.transpose(1, 2)).squeeze(1)   # (B,P)
        thr_sorted, order = torch.sort(thr, dim=1)
        cnt = torch.zeros((B, P + 1), device=device, dtype=torch.int64)
        for cs in range(0, n, doc_chunk):
            Xc = torch.from_numpy(np.array(X[cs:min(cs + doc_chunk, n)], copy=True)
                                  ).to(device).float()
            sim = Ab @ Xc.T
            pos = torch.searchsorted(thr_sorted.contiguous(), sim.contiguous(),
                                     right=False)
            cnt.scatter_add_(1, pos, torch.ones_like(pos, dtype=torch.int64))
            del Xc, sim, pos
        above = torch.flip(torch.cumsum(torch.flip(cnt, [1]), 1), [1])[:, 1:]
        r = torch.empty_like(above)
        r.scatter_(1, order, above)
        ranks[s0:e0] = r.cpu().numpy()
        del Ab, Pv, thr, thr_sorted, cnt, above, r
        torch.cuda.empty_cache()

    W = np.zeros((nq, k, k), np.int64)
    for q in range(nq):
        row = inv[q]
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                a, b = int(row[i]), int(row[j])
                W[q, i, j] = ranks[a, slot[a][b]]
    return np.minimum(W, W.transpose(0, 2, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("-M", type=int, default=16, help="the index's M; maxM0 = 2M")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--n-queries", type=int, default=1000,
                    help="a fixed random sample of the evaluation split")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    maxM0 = 2 * a.M
    X = np.load(EMB / f"{a.corpus}__{a.arm}_docs.npy", mmap_mode="r")
    gt = np.load(RES / f"gt_{a.corpus}__{a.arm}_k{a.k}.npz")["idx"]
    qs = [json.loads(l) for l in
          (ROOT / "corpora" / f"{a.corpus}_queries.jsonl")
          .read_text(encoding="utf-8").splitlines() if l.strip()]
    rels = [{int(x): int(v) for x, v in q["rel"].items()} for q in qs]

    rng = np.random.default_rng(a.seed)
    nq = min(a.n_queries, len(gt))
    sel = np.sort(rng.choice(len(gt), nq, replace=False))
    T = gt[sel].astype(np.int64)

    print(f"[{a.corpus}/{a.arm}] {nq:,} queries, k={a.k}, maxM0={maxM0}", flush=True)
    W = mutual_ranks(X, T)
    iu = np.triu_indices(a.k, 1)
    pair = W[:, iu[0], iu[1]]
    adj = W <= maxM0
    for q in range(nq):
        np.fill_diagonal(adj[q], False)
    phi1 = np.array([components(adj[q]) for q in range(nq)], float)

    # the cost side: what the embedding ranks with no index at all
    ex_mrr = float(mrr_at_k(T, [rels[i] for i in sel], 10).mean())
    ex_ndcg = float(ndcg_at_k(T, [rels[i] for i in sel], 10).mean())
    ex_gold = float(qrels_recall_at_k(T, [rels[i] for i in sel], a.k).mean())

    out = {"corpus": a.corpus, "arm": a.arm, "M": a.M, "maxM0": maxM0, "k": a.k,
           "n_queries": int(nq), "query_seed": a.seed,
           "phi1_hat": float(phi1.mean()),
           "pr_phi1_hat_eq_1": float((phi1 == 1).mean()),
           "median_mutual_rank": float(np.median(pair)),
           "pair_within_maxM0": float((pair <= maxM0).mean()),
           "exhaustive_mrr10": ex_mrr, "exhaustive_ndcg10": ex_ndcg,
           f"exhaustive_gold{a.k}": ex_gold,
           "note": "phi1_hat: components of the target set under the best graph "
                   "a builder with out-degree maxM0 could make. No index, no "
                   "build noise; a bound on the real phi_1."}
    p = RES / f"screen_{a.corpus}__{a.arm}{a.tag}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"  phi1_hat {out['phi1_hat']:.3f}  Pr[=1] {out['pr_phi1_hat_eq_1']:.3f}  "
          f"median mutual rank {out['median_mutual_rank']:.0f}  "
          f"pairs within maxM0 {out['pair_within_maxM0']:.3f}", flush=True)
    print(f"  exhaustive  MRR@10 {ex_mrr:.4f}  nDCG@10 {ex_ndcg:.4f}  "
          f"gold@{a.k} {ex_gold:.4f}", flush=True)
    print(f"wrote {p.name}", flush=True)


if __name__ == "__main__":
    main()

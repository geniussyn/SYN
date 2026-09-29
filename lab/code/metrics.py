"""Shared metric definitions. One place, so every figure means the same thing."""
import numpy as np

def recall_at_k(ret, gt, k=10):
    """Fraction of each query's exact top-k that the ANN result list contains."""
    out = np.empty(len(ret), dtype=np.float64)
    for i, (r, g) in enumerate(zip(ret, gt)):
        out[i] = len(set(r[:k].tolist()) & set(g[:k].tolist())) / k
    return out

def mrr_at_k(ret, rels, k=10):
    """MRR@k of the returned list against human qrels (score>0 = relevant)."""
    out = np.zeros(len(ret), dtype=np.float64)
    for i, (r, R) in enumerate(zip(ret, rels)):
        pos = {d for d, s in R.items() if s > 0}
        for j, d in enumerate(r[:k]):
            if int(d) in pos:
                out[i] = 1.0 / (j + 1); break
    return out

def ndcg_at_k(ret, rels, k=10):
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    out = np.zeros(len(ret), dtype=np.float64)
    for i, (r, R) in enumerate(zip(ret, rels)):
        g = np.array([R.get(int(d), 0) for d in r[:k]], dtype=np.float64)
        dcg = float((np.maximum(g, 0) * disc[:len(g)]).sum())
        ideal = np.sort(np.array([v for v in R.values() if v > 0], dtype=np.float64))[::-1][:k]
        idcg = float((ideal * disc[:len(ideal)]).sum()) if ideal.size else 0.0
        out[i] = dcg / idcg if idcg > 0 else 0.0
    return out


def qrels_recall_at_k(ret, rels, k):
    """Fraction of a query's RELEVANT documents that appear in the returned list.

    Distinct from recall against the arm's own exact top-k: that says what the
    index realises of its encoder, this says what a reranker downstream is
    actually handed. At the candidate depth a reranking pipeline uses, this is
    the ceiling no reranker can exceed.
    """
    out = np.zeros(len(ret), dtype=np.float64)
    for i, (r, R) in enumerate(zip(ret, rels)):
        pos = {d for d, s in R.items() if s > 0}
        if not pos:
            continue
        out[i] = len(pos & set(int(x) for x in r[:k])) / len(pos)
    return out

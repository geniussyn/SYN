"""The fp32 exhaustive reference.

Document vectors are stored fp16, because that is what the served index stores.
Every quantity SYN derives from them -- the frozen target set S_q, mutual ranks,
boundary bands, the positive law -- is accumulated in **fp32**. A reference has
to be at least as precise as the object it defines: the active-pair threshold
sits at rank L and the boundary band just below it, where neighbours are
separated by far less than a half-precision ulp, so an fp16 accumulation
reorders them and both the pair set and the band become arithmetic noise.
"""
import numpy as np
import torch


def load_doc_matrix(path, n_docs=None):
    """Memmap the document cache the index is built from."""
    X = np.load(str(path), mmap_mode="r")
    if n_docs is not None and X.shape[0] != n_docs:
        raise SystemExit(f"--doc-emb has {X.shape[0]:,} rows, corpus has {n_docs:,}")
    return X


@torch.no_grad()
def topk(Q, X, k, qchunk=512, dchunk=1_000_000, device="cuda", return_sim=True):
    """Exhaustive top-k by inner product, accumulated in fp32.

    Q (nq, d) and X (n, d) may be fp16 or fp32, in memory or memmapped. Returns
    (idx int32, sim float32) in decreasing similarity.
    """
    nq, n = Q.shape[0], X.shape[0]
    k = min(k, n)
    Qg = torch.from_numpy(np.array(Q, copy=True)).to(device).float()
    best_v = torch.full((nq, k), -3.4e38, device=device, dtype=torch.float32)
    best_i = torch.zeros((nq, k), device=device, dtype=torch.int64)
    for s in range(0, n, dchunk):
        e = min(s + dchunk, n)
        Xg = torch.from_numpy(np.array(X[s:e], copy=True)).to(device).float()
        for qs in range(0, nq, qchunk):
            qe = min(qs + qchunk, nq)
            v, i = torch.topk(Qg[qs:qe] @ Xg.T, min(k, e - s), dim=1)
            cv = torch.cat([best_v[qs:qe], v], 1)
            ci = torch.cat([best_i[qs:qe], i + s], 1)
            v2, j = torch.topk(cv, k, dim=1)
            best_v[qs:qe] = v2
            best_i[qs:qe] = ci.gather(1, j)
        del Xg
        torch.cuda.empty_cache()
    idx = best_i.cpu().numpy().astype(np.int32)
    sim = best_v.cpu().numpy() if return_sim else None
    del best_i, best_v, Qg
    torch.cuda.empty_cache()
    return (idx, sim) if return_sim else idx


def neighbourhoods(ids, X, L, **kw):
    """N_L(u): the exact top-L of each u in `ids`, itself removed.

    rank_u(v) = |{w : sim(u,w) > sim(u,v)}| counts u itself, so it is 1-based
    excluding self and `v in N_L(u)` is exactly `rank_u(v) <= L`.
    """
    ids = np.asarray(ids)
    idx = topk(np.ascontiguousarray(X[ids]), X, L + 1, return_sim=False, **kw)
    out = np.empty((ids.size, L), dtype=np.int32)
    for r, self_id in enumerate(ids):
        row = idx[r]
        row = row[row != self_id]
        out[r] = row[:L] if row.size >= L else np.pad(row, (0, L - row.size),
                                                      constant_values=row[-1])
    return out

"""Target-set cohesion: kappa, kappa_tree (Def. 3.2), and fragmentation phi.

kappa uses MUTUAL RANK, not distance -- rank_u(v) = #{w in C : sim(u,w) > sim(u,v)},
which counts u itself and so is 1-based excluding self, matching syn/train.py's
monitor. w(u,v) = min(rank_u(v), rank_v(u)); kappa = median over the k*(k-1)/2
pairs; kappa_tree = the MST bottleneck (max edge of the minimum spanning tree),
i.e. min{M : B_M connected}.
"""
import numpy as np
import torch


def mutual_ranks(X, GT, anchor_batch=1024, doc_chunk=1_000_000, device="cuda"):
    """X (n,d) fp16 (memmap ok), GT (nq,k) int32 -> W (nq, k, k) float32 mutual ranks.

    One corpus scan per anchor batch; torch.searchsorted turns the k-1 threshold
    comparisons into one log-time lookup, so cost is ~one GEMM per anchor.
    """
    nq, k = GT.shape
    n = X.shape[0]
    anchors, inv = np.unique(GT.reshape(-1), return_inverse=True)
    inv = inv.reshape(nq, k)
    na = anchors.size

    # partner list per anchor: every co-member of every GT set the anchor appears in
    part = [set() for _ in range(na)]
    for qi in range(nq):
        row = inv[qi]
        for a in range(k):
            for b in range(k):
                if a != b:
                    part[int(row[a])].add(int(row[b]))
    P = max(len(s) for s in part)
    # pad holds CORPUS ids (anchors[...]); slot is keyed by ANCHOR index.
    pad = np.zeros((na, P), dtype=np.int64)
    slot = [dict() for _ in range(na)]
    for i, ss in enumerate(part):
        for j, v in enumerate(sorted(ss)):
            pad[i, j] = anchors[v]; slot[i][v] = j
        if len(ss) < P:
            pad[i, len(ss):] = pad[i, 0]
    print(f"  kappa: {na:,} anchors, max partners {P}", flush=True)

    ranks = np.zeros((na, P), dtype=np.int64)
    aidx = torch.from_numpy(anchors.astype(np.int64))
    padt = torch.from_numpy(pad)
    for s in range(0, na, anchor_batch):
        e = min(s + anchor_batch, na)
        rows = aidx[s:e].numpy()
        Xa = torch.from_numpy(np.ascontiguousarray(X[rows])).to(device).half()
        pr = padt[s:e].to(device)
        Xp = torch.from_numpy(np.ascontiguousarray(X[pad[s:e].reshape(-1)])).to(device).half()
        Xp = Xp.reshape(e - s, P, -1)
        # right=False below counts s STRICTLY greater than the threshold, so the
        # partner's own self-similarity does not inflate its rank.
        # Same half-GEMM path as `sim` below: a float32 elementwise thr against a
        # half-GEMM sim disagrees by a ULP and shifts ranks among near-ties.
        thr = torch.bmm(Xa.unsqueeze(1), Xp.transpose(1, 2)).squeeze(1).float()  # (B,P)
        thr_sorted, order = torch.sort(thr, dim=1)
        cnt = torch.zeros((e - s, P + 1), device=device, dtype=torch.int64)
        for cs in range(0, n, doc_chunk):
            ce = min(cs + doc_chunk, n)
            Xc = torch.from_numpy(np.ascontiguousarray(X[cs:ce])).to(device).half()
            sim = (Xa @ Xc.T).float()
            pos = torch.searchsorted(thr_sorted.contiguous(), sim.contiguous(), right=False)
            cnt += torch.zeros_like(cnt).scatter_add_(
                1, pos, torch.ones_like(pos, dtype=torch.int64))
            del Xc, sim, pos
        # #elements strictly greater than thr_sorted[:, j] = sum of cnt[:, j+1:]
        gt = torch.flip(torch.cumsum(torch.flip(cnt, [1]), 1), [1])[:, 1:]
        r = torch.empty_like(gt)
        r.scatter_(1, order, gt)
        ranks[s:e] = r.cpu().numpy()
        del Xa, Xp, thr, thr_sorted, cnt, gt, r
        torch.cuda.empty_cache()
        if (s // anchor_batch) % 5 == 0:
            print(f"    anchors {e:,}/{na:,}", flush=True)

    W = np.zeros((nq, k, k), dtype=np.float32)
    for qi in range(nq):
        row = inv[qi]
        for a in range(k):
            ia = int(row[a])
            for b in range(k):
                if a == b: continue
                ib = int(row[b])
                r_ab = ranks[ia, slot[ia][ib]]
                r_ba = ranks[ib, slot[ib][ia]]
                W[qi, a, b] = min(r_ab, r_ba)
    return W


def kappa_from_W(W):
    """kappa = median off-diagonal weight; kappa_tree = MST bottleneck."""
    nq, k, _ = W.shape
    off = ~np.eye(k, dtype=bool)
    kap = np.array([np.median(W[q][off]) for q in range(nq)])
    ktree = np.empty(nq)
    for q in range(nq):
        w = W[q].copy(); np.fill_diagonal(w, np.inf)
        inside = np.zeros(k, dtype=bool); inside[0] = True
        best = w[0].copy(); mx = 0.0
        for _ in range(k - 1):
            best_masked = np.where(inside, np.inf, best)
            j = int(np.argmin(best_masked))
            mx = max(mx, float(best_masked[j]))
            inside[j] = True
            best = np.minimum(best, w[j])
        ktree[q] = mx
    return kap, ktree


def _hop_expand(indptr, indices, frontier):
    deg = indptr[frontier + 1] - indptr[frontier]
    tot = int(deg.sum())
    if tot == 0:
        return np.empty(0, dtype=np.int64)
    g = np.repeat(indptr[frontier], deg) + \
        (np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(deg) - deg, deg))
    return indices[g].astype(np.int64)


def reach_sets(indptr, indices, seeds, h):
    """Nodes reachable from each seed within h hops (seed included)."""
    out = []
    for s in seeds:
        seen = {int(s)}
        frontier = np.array([s], dtype=np.int64)
        for _ in range(h):
            nxt = _hop_expand(indptr, indices, frontier)
            if nxt.size == 0: break
            nxt = np.unique(nxt)
            new = nxt[~np.isin(nxt, np.fromiter(seen, dtype=np.int64, count=len(seen)))]
            if new.size == 0: break
            seen.update(new.tolist())
            frontier = new
        out.append(seen)
    return out


def phi(indptr, indices, GT, h=3, log_every=500):
    """Definition 2.1: components of the graph on GT_k(q) with u<->v when each is
    reachable from the other within h hops of G."""
    nq, k = GT.shape
    out = np.empty(nq, dtype=np.int32)
    for qi in range(nq):
        g = GT[qi].astype(np.int64)
        R = reach_sets(indptr, indices, g, h)
        parent = list(range(k))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for a in range(k):
            for b in range(a + 1, k):
                if int(g[b]) in R[a] and int(g[a]) in R[b]:
                    ra, rb = find(a), find(b)
                    if ra != rb: parent[ra] = rb
        out[qi] = len({find(x) for x in range(k)})
        if log_every and qi % log_every == 0:
            print(f"    phi {qi:,}/{nq:,}", flush=True)
    return out

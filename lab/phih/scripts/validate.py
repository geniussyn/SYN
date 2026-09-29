"""Two checks that the extracted CSR really is the graph hnswlib traverses:
 (1) level-0 out-neighbours are geometrically local (overlap with true kNN);
 (2) a plain ef-search over the extracted CSR, seeded at the stored entry point,
     reproduces hnswlib's recall."""
import os, sys, heapq, struct
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[v] = '24'
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CORPORA, WORK
from dump_graph import read_level0

M, EFC, K = 16, 200, 10

def main(name, nq=200, ef=128, seed=0):
    c = CORPORA[name]
    base, query = c['base'](), c['query']()
    if c['space'] == 'cosine':
        base = base / np.linalg.norm(base, axis=1, keepdims=True).clip(1e-30)
        query = query / np.linalg.norm(query, axis=1, keepdims=True).clip(1e-30)
    n = base.shape[0]
    gt = np.load(f'{WORK}/{name}_gt{K}.npy')
    _, _, _, label, meta = read_level0(f'{WORK}/{name}_M{M}_efc{EFC}.bin')
    ep = int(label[meta['enterpoint']])
    indptr = np.fromfile(f'{WORK}/{name}_fwd_indptr.i64', dtype=np.int64)
    indices = np.fromfile(f'{WORK}/{name}_fwd_indices.i32', dtype=np.int32)

    # (1) locality of the extracted adjacency lists
    rng = np.random.default_rng(seed)
    probe = rng.choice(n, 200, replace=False)
    if c['space'] == 'cosine':
        d = 1.0 - base[probe] @ base.T
    else:
        d = ((base[probe] ** 2).sum(1)[:, None] - 2 * base[probe] @ base.T
             + (base ** 2).sum(1)[None, :])
    rank_of = np.argsort(np.argsort(d, axis=1, kind='stable'), axis=1)
    fr, med = [], []
    for r, u in enumerate(probe):
        nb = indices[indptr[u]:indptr[u + 1]]
        rk = rank_of[r, nb]
        fr.append((rk <= 64).mean()); med.append(np.median(rk))
    print(f'[{name}] neighbour-locality: {np.mean(fr)*100:.1f}% of level-0 out-edges land in '
          f'true top-64; median true rank of a neighbour = {np.median(med):.0f} (of {n})', flush=True)

    # (2) ef-search over the extracted graph only
    def search(q):
        if c['space'] == 'cosine':
            dist = lambda ids: 1.0 - base[ids] @ q
        else:
            dist = lambda ids: ((base[ids] - q) ** 2).sum(1)
        d0 = float(dist(np.array([ep]))[0])
        cand = [(d0, ep)]; top = [(-d0, ep)]
        vis = {ep}
        while cand:
            dc, u = heapq.heappop(cand)
            if dc > -top[0][0] and len(top) >= ef: break
            nb = indices[indptr[u]:indptr[u + 1]]
            nb = np.fromiter((v for v in nb if v not in vis), dtype=np.int64)
            if not len(nb): continue
            vis.update(nb.tolist())
            for dv, v in zip(dist(nb), nb):
                dv = float(dv)
                if len(top) < ef or dv < -top[0][0]:
                    heapq.heappush(cand, (dv, int(v))); heapq.heappush(top, (-dv, int(v)))
                    if len(top) > ef: heapq.heappop(top)
        return [v for _, v in sorted((-a, b) for a, b in top)][:K], len(vis)

    qs = rng.choice(len(query), nq, replace=False)
    hits, nd = [], []
    for qi in qs:
        r, v = search(query[qi])
        hits.append(len(set(r) & set(gt[qi].tolist()))); nd.append(v)
    ref = np.load(f'{WORK}/{name}_recall_ef{ef}.npy')[qs]
    print(f'[{name}] ef={ef} on extracted CSR: recall@{K}={np.mean(hits)/K:.4f} '
          f'(hnswlib, same queries: {ref.mean():.4f})  mean visited={np.mean(nd):.0f}', flush=True)

if __name__ == '__main__':
    main(sys.argv[1])

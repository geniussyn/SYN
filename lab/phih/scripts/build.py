import os, sys, time
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = '24'
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CORPORA, WORK, NTHREAD

M, EFC, K = 16, 200, 10

def exact_gt(base, query, space, k=K, cand=256, chunk=256):
    """Exact top-k, ties broken by ascending document id."""
    n = base.shape[0]
    if space == 'cosine':
        b = base / np.linalg.norm(base, axis=1, keepdims=True).clip(1e-30)
        q = query / np.linalg.norm(query, axis=1, keepdims=True).clip(1e-30)
        bsq = None
    else:
        b, q, bsq = base, query, (base.astype(np.float32) ** 2).sum(1)
    out = np.empty((query.shape[0], k), dtype=np.int64)
    for s in range(0, query.shape[0], chunk):
        e = min(s + chunk, query.shape[0])
        d = -2.0 * (q[s:e] @ b.T)                      # (c, n) float32
        if bsq is not None:
            d += bsq[None, :]
        idx = np.argpartition(d, cand, axis=1)[:, :cand]
        idx.sort(axis=1)                               # id-ascending => stable sort ties by id
        dv = np.take_along_axis(d, idx, axis=1)
        order = np.argsort(dv, axis=1, kind='stable')[:, :k]
        out[s:e] = np.take_along_axis(idx, order, axis=1)
        del d
    return out

def main(name):
    import hnswlib
    c = CORPORA[name]
    t = time.time(); base = c['base'](); query = c['query']()
    n, dim = base.shape
    print(f'[{name}] base {base.shape} query {query.shape} space={c["space"]} load {time.time()-t:.1f}s', flush=True)

    gt_p = f'{WORK}/{name}_gt{K}.npy'
    if not os.path.exists(gt_p):
        t = time.time(); gt = exact_gt(base, query, c['space'])
        np.save(gt_p, gt); gt.astype(np.int32).tofile(f'{WORK}/{name}_gt{K}.i32'); print(f'[{name}] exact GT {time.time()-t:.1f}s', flush=True)
    else:
        gt = np.load(gt_p)
    gt.astype(np.int32).tofile(f'{WORK}/{name}_gt{K}.i32')

    idx_p = f'{WORK}/{name}_M{M}_efc{EFC}.bin'
    if not os.path.exists(idx_p):
        p = hnswlib.Index(space=c['space'], dim=dim)
        p.init_index(max_elements=n, ef_construction=EFC, M=M, random_seed=100)
        p.set_num_threads(NTHREAD)
        t = time.time()
        p.add_items(base, np.arange(n, dtype=np.int64), num_threads=NTHREAD)
        print(f'[{name}] build {time.time()-t:.1f}s', flush=True)
        p.save_index(idx_p)
    else:
        p = hnswlib.Index(space=c['space'], dim=dim)
        p.load_index(idx_p, max_elements=n)
    print(f'[{name}] index ready -> {idx_p}', flush=True)

    p.set_num_threads(NTHREAD)
    for ef in (32, 64, 128, 256):
        p.set_ef(ef)
        t = time.time()
        lab, _ = p.knn_query(query, k=K, num_threads=NTHREAD)
        el = time.time() - t
        hits = np.array([len(np.intersect1d(lab[i], gt[i])) for i in range(len(gt))])
        np.save(f'{WORK}/{name}_recall_ef{ef}.npy', hits / K)
        print(f'[{name}] ef={ef:4d} recall@{K}={hits.mean()/K:.4f} '
              f'hard-decile={np.sort(hits)[:len(hits)//10].mean()/K:.4f} {el:.1f}s', flush=True)

if __name__ == '__main__':
    main(sys.argv[1])

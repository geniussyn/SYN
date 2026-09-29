"""SpaceV-100M: is the shipped GT exact, L2, and against THIS base?"""
import time
import cfg, numpy as np, faiss

faiss.omp_set_num_threads(cfg.NTHREAD)
t0 = time.time()
base, query, space, gt = cfg.load_spacev()
n, d = base.shape
print(f'base {n:,}x{d} int8, queries {query.shape}, gt {gt.shape}, '
      f'loaded in {time.time()-t0:.0f}s', flush=True)
print(f'gt id range [{gt.min()}, {gt.max()}]  (n={n})', flush=True)

rng = np.random.default_rng(0)
qi = np.sort(rng.choice(query.shape[0], 200, replace=False))
xq = np.ascontiguousarray(query[qi])
K = 10
bd = np.full((len(qi), K), np.inf, np.float32)
bi = np.full((len(qi), K), -1, np.int64)
step = 2_000_000
t0 = time.time()
for i0 in range(0, n, step):
    xb = base[i0:i0 + step]
    D, I = faiss.knn(xq, xb, K, metric=faiss.METRIC_L2)
    bd = np.concatenate([bd, D.astype(np.float32)], 1)
    bi = np.concatenate([bi, I.astype(np.int64) + i0], 1)
    o = np.argsort(bd, axis=1)[:, :K]
    bd, bi = np.take_along_axis(bd, o, 1), np.take_along_axis(bi, o, 1)
print(f'brute force over {n:,} points for {len(qi)} queries: {time.time()-t0:.0f}s', flush=True)

inter = np.mean([len(set(bi[j]) & set(gt[qi[j], :K])) for j in range(len(qi))])
# distance of the shipped 10th neighbour, recomputed here
dk = np.array([((query[qi[j]] - base[gt[qi[j], K-1]]) ** 2).sum() for j in range(len(qi))])
print(f'mean |mine n shipped|@10 = {inter:.3f}/10')
print(f'shipped d10 vs exact d10: max rel gap = '
      f'{np.abs(dk - bd[:, K-1]).max() / max(bd[:, K-1].max(), 1e-9):.3e}')
bad = int((dk > bd[:, K - 1] * (1 + 1e-6) + 1e-6).sum())
print(f'queries where the shipped 10th neighbour is farther than the true 10th: {bad}/{len(qi)}')

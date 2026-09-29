import os, sys, time
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[v] = '24'
import numpy as np
from common import CORPORA, WORK, NTHREAD
from build import exact_gt
import hnswlib
M, EFC, K = 16, 200, 100
import sys
for name in (sys.argv[1:] or ['sift','glove']):
    c = CORPORA[name]; base, query = c['base'](), c['query']()
    n, dim = base.shape
    p = f'{WORK}/{name}_gt{K}.npy'
    if not os.path.exists(p):
        t = time.time(); np.save(p, exact_gt(base, query, c['space'], k=K, cand=512, chunk=128))
        print(f'[{name}] exact GT@{K} {time.time()-t:.1f}s', flush=True)
    gt = np.load(p)
    idx = hnswlib.Index(space=c['space'], dim=dim)
    idx.load_index(f'{WORK}/{name}_M{M}_efc{EFC}.bin', max_elements=n)
    idx.set_num_threads(NTHREAD)
    for ef in (128, 200, 256, 512, 1024):
        idx.set_ef(ef)
        lab, _ = idx.knn_query(query, k=K, num_threads=NTHREAD)
        hits = np.array([len(np.intersect1d(lab[i], gt[i])) for i in range(len(gt))])
        np.save(f'{WORK}/{name}_recall{K}_ef{ef}.npy', hits / K)
        print(f'[{name}] k={K} ef={ef:5d} recall={hits.mean()/K:.4f} sd={(hits/K).std():.4f} '
              f'p10={np.percentile(hits/K,10):.3f} p90={np.percentile(hits/K,90):.3f}', flush=True)
    gt.astype(np.int32).tofile(f'{WORK}/{name}_gt{K}.i32')

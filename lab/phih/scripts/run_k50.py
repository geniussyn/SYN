"""Put phi_1 and the effort axis on the Steiner-hardness paper's own setting:
k=50, target recall (Acc) = 0.98, y = least NDC."""
import os, sys, subprocess
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='10'
import numpy as np
from common import CORPORA, WORK, NTHREAD
from build import exact_gt
from run_effort import prep
import hnswlib

K, M, EFC, EF = 50, 16, 200, 128
LADDER = ['16','32','64','128','256','512','1024','2048','4096','8192','16384','32768']

for name in sys.argv[1:]:
    c = CORPORA[name]
    gp = f'{WORK}/{name}_gt{K}.npy'
    if not os.path.exists(gp):
        base, query = c['base'](), c['query']()
        gt = exact_gt(base, query, c['space'], k=K, cand=256, chunk=128)
        np.save(gp, gt); gt.astype(np.int32).tofile(f'{WORK}/{name}_gt{K}.i32')
        n, dim = base.shape
        p = hnswlib.Index(space=c['space'], dim=dim)
        p.load_index(f'{WORK}/{name}_M{M}_efc{EFC}.bin', max_elements=n)
        p.set_num_threads(NTHREAD); p.set_ef(EF)
        lab, _ = p.knn_query(query, k=K, num_threads=NTHREAD)
        hits = np.array([len(np.intersect1d(lab[i], gt[i])) for i in range(len(gt))])
        np.save(f'{WORK}/{name}_recall{K}_ef{EF}.npy', hits / K)
        print(f'[{name}] k={K} ef={EF} recall={hits.mean()/K:.4f}', flush=True)
        del base, query, p
    if not os.path.exists(f'{WORK}/{name}_phi{K}.i32'):
        subprocess.run(['./phih', f'{WORK}/{name}', f'{WORK}/{name}_gt{K}.i32', str(K),
                        f'{WORK}/{name}_phi{K}.i32', str(NTHREAD)], check=True)
    if not os.path.exists(f'{WORK}/{name}_effort{K}.bin'):
        bp, qp, dim, ep, metric = prep(name)
        subprocess.run(['./effort', f'{WORK}/{name}', bp, qp, str(dim),
                        f'{WORK}/{name}_gt{K}.i32', str(K), str(ep), metric,
                        f'{WORK}/{name}_effort{K}.bin', str(NTHREAD)] + LADDER, check=True)

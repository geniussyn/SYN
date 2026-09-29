"""Wall-clock cost of the prerequisites each measure needs, GloVe, k=100, 10 threads."""
import os, time
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np
from common import CORPORA
from build import exact_gt
from hardness import dists          # exact top-200 pass -> LID, RC@k, d_2k/d_k, d_k

c = CORPORA['glove']
base, query = c['base'](), c['query']()
t = time.time(); exact_gt(base, query, c['space'], k=100, cand=256, chunk=128)
print(f'GT100 {time.time()-t:.1f}', flush=True)
t = time.time(); dists('glove')
print(f'DISTPASS {time.time()-t:.1f}', flush=True)

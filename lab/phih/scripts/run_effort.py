import os, sys, subprocess, struct
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='24'
import numpy as np
from common import CORPORA, WORK
from dump_graph import read_level0

import os
SCRATCH = os.environ.get('PHIH_TMP', '/tmp/phih') + '/eff'
os.makedirs(SCRATCH, exist_ok=True)
M, EFC = 16, 200

def prep(name):
    c = CORPORA[name]; base, query = c['base'](), c['query']()
    if c['space'] == 'cosine':
        base = base/np.linalg.norm(base,axis=1,keepdims=True).clip(1e-30)
        query = query/np.linalg.norm(query,axis=1,keepdims=True).clip(1e-30)
    bp, qp = f'{SCRATCH}/{name}_b.f32', f'{SCRATCH}/{name}_q.f32'
    if not os.path.exists(bp): base.astype(np.float32).tofile(bp)
    if not os.path.exists(qp): query.astype(np.float32).tofile(qp)
    _, _, _, label, meta = read_level0(f'{WORK}/{name}_M{M}_efc{EFC}.bin')
    return bp, qp, base.shape[1], int(label[meta['enterpoint']]), \
           'l2' if c['space'] == 'l2' else 'cos'

def load(path):
    a = np.fromfile(path, dtype=np.int64)
    NQ, NE, K = a[:3]
    efs = a[3:3+NE]
    ndc = a[3+NE:3+NE+NQ*NE].reshape(NQ, NE)
    hit = a[3+NE+NQ*NE:].reshape(NQ, NE)
    return efs, ndc, hit

if __name__ == '__main__':
  for name in sys.argv[1:]:
      bp, qp, dim, ep, metric = prep(name)
      for K in (10, 100):
          out = f'{WORK}/{name}_effort{K}.bin'
          if not os.path.exists(out):
              subprocess.run(['./effort', f'{WORK}/{name}', bp, qp, str(dim),
                              f'{WORK}/{name}_gt{K}.i32', str(K), str(ep), metric, out, '24'],
                             check=True)
          efs, ndc, hit = load(out)
          i128 = int(np.where(efs == 128)[0][0])
          ref = np.load(f'{WORK}/{name}_recall_ef128.npy' if K == 10
                        else f'{WORK}/{name}_recall{K}_ef128.npy')
          print(f'[{name}] k={K}: ef=128 recall {(hit[:, i128]/K).mean():.4f} '
                f'(hnswlib {ref.mean():.4f})  mean NDC {ndc[:, i128].mean():.0f}  '
                f'ef ladder {efs.min()}..{efs.max()}', flush=True)

"""Classical per-query difficulty measures, computed exactly from the base set."""
import os, sys, time
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='24'
import numpy as np
from common import CORPORA, WORK

K = 100
CAND = 200          # need 2k distances for the expansion ratio

def dists(name, chunk=128):
    """Sorted distances from each query to its CAND nearest base points, plus the
    global mean distance (exact, for relative contrast)."""
    c = CORPORA[name]; base, query = c['base'](), c['query']()
    if c['space'] == 'cosine':
        b = base/np.linalg.norm(base,axis=1,keepdims=True).clip(1e-30)
        q = query/np.linalg.norm(query,axis=1,keepdims=True).clip(1e-30)
        bsq = None
    else:
        b, q, bsq = base, query, (base.astype(np.float32)**2).sum(1)
    nq = q.shape[0]
    D = np.empty((nq, CAND), dtype=np.float64)
    MU = np.empty(nq, dtype=np.float64)
    for s in range(0, nq, chunk):
        e = min(s+chunk, nq)
        if bsq is not None:
            d = -2.0*(q[s:e] @ b.T)
            d += bsq[None,:] + (q[s:e]**2).sum(1)[:,None]
            np.maximum(d, 0, out=d); np.sqrt(d, out=d)          # true L2
        else:
            d = 1.0 - (q[s:e] @ b.T)                            # cosine distance
            np.maximum(d, 0, out=d)
        MU[s:e] = d.mean(1)
        idx = np.argpartition(d, CAND, axis=1)[:, :CAND]
        dv = np.take_along_axis(d, idx, axis=1)
        dv.sort(axis=1)
        D[s:e] = dv
        del d
    return D, MU

import sys
for name in (sys.argv[1:] or ['sift','glove']):
    p = f'{WORK}/{name}_hardness.npz'
    if os.path.exists(p): continue
    t = time.time(); D, MU = dists(name)
    eps = 1e-12
    dk  = D[:, K-1]
    d1  = D[:, 0]
    ndup = int((d1 <= eps).sum())          # exact duplicates of the query in the base set
    D = np.maximum(D, eps)                 # keep log() finite for those
    # LID, MLE estimator (Amsaleg et al. 2015) on the K nearest neighbours
    lid = -1.0 / (np.log(np.maximum(D[:, :K], eps) / np.maximum(dk, eps)[:, None]).mean(1) - eps)
    # Relative contrast (He et al. 2012): mean distance / distance to the NN
    rc1 = MU / np.maximum(d1, eps)
    rck = MU / np.maximum(dk, eps)
    # Expansion ratio d_2k / d_k  (Karger-Ruhl style local expansion)
    exp2 = D[:, 2*K-1] / np.maximum(dk, eps)
    np.savez(p, ndup=ndup, lid=lid, rc1=rc1, rck=rck, exp2=exp2, dk=dk, d1=d1, mu=MU)
    print(f'[{name}] LID {lid.mean():.2f}  RC@1 {rc1.mean():.2f}  RC@k {rck.mean():.2f}  '
          f'exp {exp2.mean():.3f}  dup-queries {ndup}  {time.time()-t:.1f}s', flush=True)

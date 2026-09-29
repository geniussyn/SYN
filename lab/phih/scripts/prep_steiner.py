"""Materialise our corpora + our HNSW level-0 graph in the layout that
Steiner-hardness's get_delta0_point expects, so their ME algorithm runs on
exactly the graph phi_1 is read from (method 1 = hnsw)."""
import os, sys, time
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='10'
import numpy as np
from common import _root, CORPORA, WORK

SB = os.environ.get('PHIH_STEINER') or _root('PHIH_STEINER', 'the Steiner-hardness checkout (VLDB\'25)')
M, EFC = 16, 200
# get_delta0_point picks the GT filename by dataset name: glove needs a
# 100,000-deep list for purpose<10; unregistered names use no suffix.
GTSPEC = {'glove': (100000, '_100000'), 'msmarco': (50000, ''), 'youtube1m': (50000, '')}

def fvecs_write(path, X, chunk=50000):
    with open(path, 'wb') as f:
        for s in range(0, len(X), chunk):
            b = np.ascontiguousarray(X[s:s+chunk], dtype=np.float32)
            d = np.empty((len(b), b.shape[1] + 1), dtype=np.float32)
            d[:, 0] = np.frombuffer(np.int32(b.shape[1]).tobytes(), dtype=np.float32)[0]
            d[:, 1:] = b
            d.tofile(f)

def gt_big(base, query, space, kk, out, chunk=64):
    """Exact top-kk, ties by ascending id, written straight to .ivecs."""
    if space == 'cosine':
        b = base/np.linalg.norm(base,axis=1,keepdims=True).clip(1e-30)
        q = query/np.linalg.norm(query,axis=1,keepdims=True).clip(1e-30); bsq = None
    else:
        b, q, bsq = base, query, (base.astype(np.float32)**2).sum(1)
    with open(out, 'wb') as f:
        for s in range(0, len(q), chunk):
            e = min(s+chunk, len(q))
            d = -2.0*(q[s:e] @ b.T)
            if bsq is not None: d += bsq[None, :]
            idx = np.argpartition(d, kk, axis=1)[:, :kk]
            idx.sort(axis=1)
            order = np.argsort(np.take_along_axis(d, idx, axis=1), axis=1, kind='stable')
            top = np.take_along_axis(idx, order, axis=1).astype(np.int32)
            row = np.empty((e-s, kk+1), dtype=np.int32); row[:, 0] = kk; row[:, 1:] = top
            row.tofile(f); del d

def graph_ibin(name, pre):
    ip = np.fromfile(f'{WORK}/{name}_fwd_indptr.i64', dtype=np.int64)
    ix = np.fromfile(f'{WORK}/{name}_fwd_indices.i32', dtype=np.int32)
    n = len(ip) - 1
    deg = np.diff(ip); D = int(deg.max())
    G = np.full((n, D), -1, dtype=np.int32)
    col = np.arange(D)[None, :] < deg[:, None]
    G[col] = ix
    with open(f'{pre}_ef{EFC}_M{M}_hnsw.ibin_plain', 'wb') as f:
        np.array([n, D], dtype=np.int32).tofile(f); G.tofile(f)
    # reverse graph: per node, uint32 ids terminated by 0xFFFFFFFF
    rp = np.fromfile(f'{WORK}/{name}_rev_indptr.i64', dtype=np.int64)
    ri = np.fromfile(f'{WORK}/{name}_rev_indices.i32', dtype=np.int32)
    rdeg = np.diff(rp)
    out = np.empty(len(ri) + n, dtype=np.uint32)
    pos = np.cumsum(rdeg + 1) - 1                       # slots for the terminators
    mask = np.ones(len(out), dtype=bool); mask[pos] = False
    out[mask] = ri.astype(np.uint32); out[pos] = np.uint32(0xFFFFFFFF)
    out.tofile(f'{pre}_ef{EFC}_M{M}_hnsw.ibin_plain_reversed_std')
    return n, D

for name in sys.argv[1:]:
    d = f'{SB}/{name}'; os.makedirs(d, exist_ok=True); pre = f'{d}/{name}'
    c = CORPORA[name]
    if not os.path.exists(f'{pre}_ef{EFC}_M{M}_hnsw.ibin_plain'):
        t = time.time(); n, D = graph_ibin(name, pre)
        print(f'[{name}] graph ibin n={n} maxdeg={D}  {time.time()-t:.0f}s', flush=True)
    if not os.path.exists(f'{pre}_base.fvecs'):
        t = time.time(); fvecs_write(f'{pre}_base.fvecs', c['base']())
        print(f'[{name}] base.fvecs {time.time()-t:.0f}s', flush=True)
    gtn, sfx = GTSPEC[name]
    gtp = f'{pre}_groundtruth{sfx}.ivecs'
    if not os.path.exists(gtp):
        t = time.time()
        gt_big(c['base'](), c['query'](), c['space'], gtn, gtp)
        print(f'[{name}] {os.path.basename(gtp)} (depth {gtn}) {time.time()-t:.0f}s', flush=True)

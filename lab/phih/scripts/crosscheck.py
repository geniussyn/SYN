"""Independent phi_h: plain forward h-step BFS (no bidirectional decomposition,
no bitmasks) via scipy sparse frontier expansion.  Must match phih.cpp exactly."""
import sys, numpy as np, scipy.sparse as sp
from common import WORK

def phi_bruteforce(A, S, hmax=4):
    k = len(S); n = A.shape[0]
    reach = np.zeros((k, k), dtype=bool)   # reach[i,j] filled per h below
    out = []
    R = np.zeros((k, k, hmax), dtype=bool)
    for i, u in enumerate(S):
        vis = np.zeros(n, dtype=bool); vis[u] = True
        front = np.zeros(n, dtype=bool); front[u] = True
        for h in range(1, hmax + 1):
            nxt = (sp.csr_matrix(front.astype(np.int8)) @ A).toarray().ravel() > 0
            nxt &= ~vis
            vis |= nxt
            front = nxt
            R[i, :, h - 1] = vis[S]
    res = []
    for h in range(hmax):
        par = list(range(k))
        def find(x):
            while par[x] != x: par[x] = par[par[x]]; x = par[x]
            return x
        for i in range(k):
            for j in range(i + 1, k):
                if R[i, j, h] and R[j, i, h]:
                    a, b = find(i), find(j)
                    if a != b: par[a] = b
        res.append(sum(1 for i in range(k) if find(i) == i))
    return res

name = sys.argv[1]; NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 20
K = int(sys.argv[3]) if len(sys.argv) > 3 else 10
SUF = '' if K == 10 else str(K)
indptr = np.fromfile(f'{WORK}/{name}_fwd_indptr.i64', dtype=np.int64)
indices = np.fromfile(f'{WORK}/{name}_fwd_indices.i32', dtype=np.int32)
n = len(indptr) - 1
A = sp.csr_matrix((np.ones(len(indices), np.int8), indices.astype(np.int64), indptr), shape=(n, n))
gt = np.load(f'{WORK}/{name}_gt{K}.npy')
phi = np.fromfile(f'{WORK}/{name}_phi{SUF}.i32', dtype=np.int32).reshape(-1, 4)
rec = np.load(f'{WORK}/{name}_recall_ef128.npy' if K==10 else f'{WORK}/{name}_recall{K}_ef128.npy')
rng = np.random.default_rng(7)
# half random, half from the hard decile (where fragmentation is largest)
hard = np.argsort(rec)[:len(rec) // 10]
qs = np.concatenate([rng.choice(len(gt), NQ // 2, replace=False),
                     rng.choice(hard, NQ - NQ // 2, replace=False)])
bad = 0
for q in qs:
    b = phi_bruteforce(A, gt[q].tolist())
    if b != list(phi[q]):
        bad += 1; print(f'  MISMATCH q={q} brute={b} fast={list(phi[q])}')
print(f'[{name}] cross-check {len(qs)} queries (incl. hard decile): '
      f'{len(qs)-bad}/{len(qs)} exact match')

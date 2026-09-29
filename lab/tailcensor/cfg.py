import os

# corpora root; no default, so a wrong path refuses rather than
# silently reading the wrong file
DATA = os.environ['TC_DATA']
# lab rule: shared 8-user server, never exceed 24 threads
NTHREAD = 24
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = str(NTHREAD)
os.environ.setdefault('HDF5_USE_FILE_LOCKING','FALSE')
import numpy as np

OUT = os.environ.get('TC_OUT', './out')
os.makedirs(OUT, exist_ok=True)

K        = 10
M        = 16
EFC      = 200
SEED_A   = 100      # evaluated index
SEED_B   = 2026     # independent index, used only to rank query hardness
EFS      = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
REPS     = 9

def fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    a = a.reshape(-1, d + 1)[:, 1:]
    return np.ascontiguousarray(a.view(np.float32))

def _norm_inplace(x, step=200_000):
    """unit-normalise rows in place (chunked: the youtube base is 64 GB)"""
    for i in range(0, x.shape[0], step):
        c = x[i:i+step]
        n = np.linalg.norm(c, axis=1, keepdims=True)
        n[n == 0] = 1.0
        c /= n
    return x

def load_gist():
    b = fvecs(f'{DATA}/gist/gist_base.fvecs')
    q = fvecs(f'{DATA}/gist/gist_query.fvecs')
    return b, q, 'l2'

def load_youtube():
    import h5py
    p = f'{DATA}/youtube-15M-angular.hdf5'
    with h5py.File(p, 'r') as f:
        n = f['train'].shape[0]
        d = f['train'].shape[1]
        b = np.empty((n, d), dtype=np.float32)
        step = 500_000
        for i in range(0, n, step):
            j = min(n, i + step)
            b[i:j] = f['train'][i:j]
        q = np.asarray(f['test'][:], dtype=np.float32)
    # angular -> unit-normalised + inner product space
    return _norm_inplace(b), _norm_inplace(q), 'ip'

NQ_SPACEV = 10_000          # of the 29,316 shipped queries (evenly spaced)
SPACEV_BASE = f'{DATA}/spacev100m_raw/spacev100m_base.i8bin'
SPACEV_H5 = f'{DATA}/msspacev-100M-i8-euclidean.hdf5'


class F32View:
    """int8 base held once; every slice/gather comes out as float32.

    (the 100M x 100 base is 10 GB as int8 and 40 GB as float32 -- hnswlib keeps
    its own float32 copy anyway, so never materialise a second one)"""

    def __init__(self, a):
        self.a = a
        self.shape = a.shape

    @property
    def dtype(self):
        return f'{self.a.dtype} view->float32'

    def __getitem__(self, idx):
        return np.ascontiguousarray(self.a[idx], dtype=np.float32)


def load_spacev():
    """SpaceV-100M, 100d int8, L2.  GT (k=100) ships with the hdf5; the hdf5's
    own `train` is an unreadable external link, so the base comes from the
    i8bin next to it."""
    import h5py
    n, d = np.fromfile(SPACEV_BASE, dtype=np.int32, count=2)
    b = np.fromfile(SPACEV_BASE, dtype=np.int8, offset=8).reshape(int(n), int(d))
    with h5py.File(SPACEV_H5, 'r') as f:
        q = np.asarray(f['test'][:])
        nb = np.asarray(f['neighbors'][:], dtype=np.int64)
    sel = np.unique(np.linspace(0, q.shape[0] - 1, NQ_SPACEV).astype(np.int64))
    return (F32View(b), np.ascontiguousarray(q[sel], dtype=np.float32), 'l2',
            np.ascontiguousarray(nb[sel]))


CORPORA = {'gist': load_gist, 'youtube': load_youtube, 'spacev': load_spacev}

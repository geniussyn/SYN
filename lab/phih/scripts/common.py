"""Paths and the thread cap for the phi_h experiments.

These ran on a different machine from the rest of the paper, on public ANN
benchmark corpora rather than on the BEIR collections, so nothing here can be
found by looking around the repository. Both roots are taken from the
environment and neither has a default: a wrong guess would silently read the
wrong corpus, which is worse than refusing to start.

    PHIH_WORK      scratch: indexes, ground truth, phi and NDC dumps
    PHIH_DATA      where the .hdf5 / .fvecs corpora live

REPRODUCE.md lists the corpora and where each was obtained.
"""
import os, numpy as np


def _root(var, what):
    v = os.environ.get(var)
    if not v:
        raise SystemExit(
            f"{var} is not set. It is the directory holding {what}; see "
            "phih/REPRODUCE.md. These experiments do not run from the "
            "repository alone -- they need the ANN benchmark corpora.")
    return v.rstrip("/")


WORK = _root("PHIH_WORK", "the phi_h scratch artefacts")
DATA = _root("PHIH_DATA", "the ANN benchmark corpora")
# the lab server this ran on was shared and capped at ten threads; the timings
# in table_cost.tex are single-threaded and unaffected by it
NTHREAD = int(os.environ.get("PHIH_THREADS", "10"))

def fvecs(path, dtype=np.float32):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    a = a.reshape(-1, d + 1)[:, 1:]
    return np.ascontiguousarray(a.view(dtype))

CORPORA = {
    'sift': dict(
        space='l2', dim=128,
        base=lambda: fvecs(f'{DATA}/sift/sift_base.fvecs'),
        query=lambda: fvecs(f'{DATA}/sift/sift_query.fvecs'),
    ),
    'glove': dict(
        space='cosine', dim=100,
        base=lambda: _h5('glove', 'train'),
        query=lambda: _h5('glove', 'test'),
    ),
    'gist': dict(
        space='l2', dim=960,
        base=lambda: fvecs(f'{DATA}/gist/gist_base.fvecs'),
        query=lambda: fvecs(f'{DATA}/gist/gist_query.fvecs'),
    ),
    'msmarco': dict(
        space='cosine', dim=1536,
        base=lambda: _h5('msmarco', 'train'),
        query=lambda: _h5('msmarco', 'test'),
    ),
    'youtube1m': dict(
        space='cosine', dim=1024,
        base=lambda: np.load(f'{WORK}/youtube1m_base.npy'),
        query=lambda: np.load(f'{WORK}/youtube1m_query.npy'),
    ),
    'landmark': dict(
        space='cosine', dim=768,
        base=lambda: _h5('landmark', 'train'),
        query=lambda: _h5('landmark', 'test'),
    ),
}

# file names as published by ann-benchmarks; DATA is wherever they were put
H5 = {'glove':    f'{DATA}/glove-100-angular.hdf5',
      'landmark': f'{DATA}/landmark-nomic-768-angular.hdf5',
      'msmarco':  f'{DATA}/msmarco-v1-openai-ada2-1M-ip.hdf5'}

def _h5(which, key):
    os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
    import h5py
    with h5py.File(H5[which], 'r') as f:
        return np.ascontiguousarray(f[key][:], dtype=np.float32)

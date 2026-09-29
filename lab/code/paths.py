"""Repository root, resolved once.

The measurement scripts were written against a fixed working directory. They take
it from SYN_ROOT if it is set, and otherwise from this file's location, so a clone
runs without editing anything.

Expected layout under the root:

    code/      this package
    corpora/   <corpus>_docs.txt, _queries.jsonl, _manifest.json, subset_*.npz
    emb/       <corpus>__<arm>_{docs,queries}.npy and _manifest.json
    results/   everything the measurements write
    figs/      everything the figure scripts write
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SYN_ROOT", Path(__file__).resolve().parent.parent))
CODE = ROOT / "code"
CORPORA = ROOT / "corpora"
# Embeddings are the one large input that is purely regenerable, and on a
# quota-limited volume they are the first thing that has to move. SYN_EMB_DIR
# relocates them without touching anything else; unset, they sit under the root
# as before.
EMB = Path(os.environ.get("SYN_EMB_DIR", ROOT / "emb"))
RESULTS = ROOT / "results"
FIGS = ROOT / "figs"

# Index files are large and short-lived; keep them off the repository filesystem.
INDEX_DIR = Path(os.environ.get("SYN_INDEX_DIR", "/dev/shm/synidx"))


def cell_path(corpus, arm, M=16, efc=200, results=None):
    """A measured cell, under whichever naming produced it.

    The current protocol derives the stratum per arm and writes
    `cell_<c>__<arm>__M<M>__own_efc<efc>.json`; the campaign before it held the
    stratum fixed on the baseline and wrote `cell_<c>__<arm>__M<M>.json`. The two
    are not comparable and are never merged -- this only finds whichever one
    exists, newest naming first, so a figure script does not have to know which
    campaign it is reading.
    """
    res = Path(results) if results else RESULTS
    for name in (f"cell_{corpus}__{arm}__M{M}__own_efc{efc}.json",
                 f"cell_{corpus}__{arm}__M{M}.json"):
        q = res / name
        if q.exists():
            return q
    return res / f"cell_{corpus}__{arm}__M{M}__own_efc{efc}.json"


def perquery_path(corpus, arm, M=16, efc=200, results=None):
    """The per-query arrays of a measured cell; same resolution as cell_path."""
    res = Path(results) if results else RESULTS
    for name in (f"perquery_{corpus}__{arm}__M{M}__own_efc{efc}.npz",
                 f"perquery_{corpus}__{arm}__M{M}.npz"):
        q = res / name
        if q.exists():
            return q
    return res / f"perquery_{corpus}__{arm}__M{M}__own_efc{efc}.npz"

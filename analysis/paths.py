"""Repository root, resolved once.

The measurement scripts were written against a fixed working directory. They take
it from SYN_ROOT if it is set, and otherwise from this file's location, so a clone
runs without editing anything.

Expected layout under the root:

    analysis/  this package
    corpora/   <corpus>_docs.txt, _queries.jsonl, _manifest.json, subset_*.npz
    emb/       <corpus>__<arm>_{docs,queries}.npy and _manifest.json
    results/   everything the measurements write
    figures/   everything the figure scripts write
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SYN_ROOT", Path(__file__).resolve().parent.parent))
CODE = ROOT / "analysis"
CORPORA = ROOT / "corpora"
# Embeddings are the one large input that is purely regenerable, and on a
# quota-limited volume they are the first thing that has to move. SYN_EMB_DIR
# relocates them without touching anything else; unset, they sit under the root
# as before.
EMB = Path(os.environ.get("SYN_EMB_DIR", ROOT / "emb"))
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

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


def cell_or_strata(corpus, arm, M=16, efc=200, results=None):
    """A cell-shaped dict for an arm that was never measured by run_cell.py.

    The headline arms have `cell_*.json`. An arm added later for comparison may
    only have `strata_cell.py`'s output, which times the SAME ef grid on every
    stratum and so already carries the two columns a cell has -- the whole set
    and the decile -- under different names (`strata.all` / `strata.P90`, with
    `ms` rather than `ms_per_query`). Synthesising the cell from it is reading
    the same measurement, not a weaker substitute; requiring a cell instead would
    mean timing the identical grid a second time.

    Probe rows are dropped: the target search adds points at an ef it already
    measured, and a duplicate ef would shadow the grid row.
    """
    import json
    res = Path(results) if results else RESULTS
    p = cell_path(corpus, arm, M, efc, results=res)
    if p.exists():
        return json.loads(p.read_text())
    sp = res / f"strata_{corpus}__{arm}__M{M}__strata_efc{efc}.json"
    if not sp.exists():
        raise FileNotFoundError(
            f"no cell and no strata file for {corpus}/{arm} "
            f"(looked for {p.name} and {sp.name})")
    s = json.loads(sp.read_text())
    st = s.get("strata") or {}
    allg = {r["ef"]: r for r in st.get("all", {}).get("grid", [])
            if "probe_for" not in r}
    hard = {r["ef"]: r for r in st.get("P90", {}).get("grid", [])
            if "probe_for" not in r}
    grid = []
    for ef in sorted(allg):
        row = {"ef": ef, "ms_per_query": allg[ef]["ms"]}
        if ef in hard:
            row["ms_per_query_hard"] = hard[ef]["ms"]
        grid.append(row)
    return {**{k: v for k, v in s.items() if k != "strata"},
            "grid": grid, "synthesised_from": sp.name}

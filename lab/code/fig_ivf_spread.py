"""The same geometry, seen by a partition index.

phi counts components under graph edges and cannot be asked of IVF, which has no
edges. The quantities that play its part are where the answers sit relative to
the partition, and both are properties of the embedding and the quantizer alone
-- no search is run to obtain either.

    left    how many distinct inverted lists the exact top-10 falls into. The
            graph figure's phi_1 and this are the same question put to two data
            structures: can the search reach the rest once it has found one.

    right   nprobe*, the rank of the deepest list holding one of the ten in the
            query's own ordering of the centroids. This is not a proxy for the
            budget IVF needs -- it IS the nprobe at which that query's recall@10
            reaches 1.0, so its distribution is the distribution of the cost,
            per query, before any search happens.

Drawn like the other distributions: probability mass as translucent bars
enclosing area 1, each arm's median as a dotted rule. nprobe* spans three decades
and is binned on a log axis, so the bars there are per-decade density.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import housestyle as H                                          # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402
from paths import ROOT                                          # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figs" / "final"
COLLS = [("msmarco-1m", "MS MARCO 1M"), ("fever", "FEVER 1M"),
         ("nq", "NQ 1M"), ("quora", "Quora 523K")]
ARMS = [("vanilla", "Vanilla", H.RED), ("syn", "SYN", H.BLUE)]
CELLS = list(range(1, 11))
NLIST = 4096


def main(pop="hard", fname="fig_ivf_spread"):
    p = RES / f"ivf_spread_{pop}.json"
    if not p.exists():
        return False
    d = json.loads(p.read_text())
    have = [(c, lab) for c, lab in COLLS
            if all(f"{c}__{a}" in d for a, _, _ in ARMS)]
    if not have:
        return False

    H.setup(13)
    fig, axes = plt.subplots(2, len(have), figsize=(4.6 * len(have), 8.2),
                             squeeze=False)
    edges = np.logspace(0, np.log10(NLIST), 25)
    for j, (c, clab) in enumerate(have):
        for arm, lab, col in ARMS:
            v = d[f"{c}__{arm}"]
            n = np.array(v["cells_counts"], float)
            axes[0][j].bar(CELLS, n / max(n.sum(), 1), width=1.0, color=col,
                           alpha=0.45, edgecolor=col, lw=1.2, zorder=2,
                           label=f"{lab}  (mean {v['cells_mean']:.2f})")
            axes[0][j].axvline(v["cells_mean"], color=col, lw=1.4,
                               ls=(0, (1, 2)), zorder=3)
            npr = np.asarray(v["nprobe"], float)
            axes[1][j].hist(npr, bins=edges, density=True, color=col, alpha=0.45,
                            edgecolor=col, linewidth=1.2, zorder=2,
                            label=f"{lab}  (median {v['nprobe_median']:.0f})")
            axes[1][j].axvline(v["nprobe_median"], color=col, lw=1.4,
                               ls=(0, (1, 2)), zorder=3)
        axes[0][j].set_title(clab, pad=8)
        axes[0][j].set_xticks(CELLS)
        axes[0][j].set_xlim(0.4, 10.6)
        axes[0][j].set_xlabel("lists holding the top-10")
        axes[1][j].set_xscale("log")
        axes[1][j].set_xlim(1, NLIST)
        axes[1][j].set_xlabel("nprobe$^*$")
        for r in (0, 1):
            axes[r][j].legend(loc="upper center", fontsize=9.5)
    axes[0][0].set_ylabel("density")
    axes[1][0].set_ylabel("density")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"{fname}.{e}")
    plt.close(fig)
    print(f"wrote figs/final/{fname}.pdf / .png  ({len(have)} collection(s))")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", default="hard")
    a = ap.parse_args()
    if not main(a.pop):
        print("no ivf_spread results yet")

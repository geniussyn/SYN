"""Two distributions, one figure: what the graph does to a query, and what the
query gets back.

The top row is phi_1, the number of connected components a query's exact top-10
falls into under the index's direct edges. The bottom row is the recall that
search actually returns at a fixed, tight budget. They are the same claim at
either end of the mechanism -- fragmentation is what the intervention changes,
recall is what changes because of it -- and putting them on one page lets the two
be read against each other rather than in separate figures.

Both rows are probability mass drawn as translucent bars at the values the
measurement can take, scaled so each encloses area 1, with each arm's mean as a
dotted rule. Nothing is smoothed: recall@10 over k=10 has eleven possible values
and phi_1 has ten, and a curve through them would invent a continuity that is not
there -- and would bury the spike at recall 0, which is the part of the
distribution the argument is about.

phi_1 is recomputed from the level-0 graph each cell dumped, so it agrees with
`phi1` in the cell JSON by construction. Nothing is rebuilt.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import housestyle as H                                          # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402
from kappa import phi                                           # noqa: E402
from paths import ROOT                                          # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
COLLS = [("msmarco-1m", "MS MARCO 1M"), ("fever", "FEVER 1M"),
         ("nq", "NQ 1M"), ("quora", "Quora 523K")]
ARMS = [("vanilla", "Vanilla", H.RED), ("syn", "SYN", H.BLUE)]
PHI_LEVELS = list(range(1, 11))
REC_LEVELS = np.arange(0, 11) / 10.0
EF_SHOW = 64          # a tight, realistic budget


def phi_dist(corpus, arm, M, efc, pop, h, cache):
    key = f"{corpus}__{arm}"
    if key in cache:
        return cache[key]
    g = RES / f"graph_{corpus}__{arm}__M{M}__own_efc{efc}.npz"
    gt = RES / f"gt_{corpus}__{arm}_k10.npz"
    ps = RES / f"pops_{corpus}__{arm}__M{M}__own_efc{efc}.npz"
    if not (g.exists() and gt.exists() and (pop == "all" or ps.exists())):
        return None
    z = np.load(g)
    T = np.load(gt)["idx"]
    if pop != "all":
        T = T[np.load(ps)[pop]]
    v = phi(z["indptr"], z["indices"], T, h=h, log_every=0)
    cache[key] = {"n": int(len(v)), "mean": float(v.mean()),
                  "counts": np.bincount(v.astype(int), minlength=11)[1:11].tolist()}
    return cache[key]


def rec_dist(corpus, arm, M, efc, pop):
    p = RES / f"perquery_{corpus}__{arm}__M{M}__own_efc{efc}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    efs = [int(e) for e in z["ef"]]
    if EF_SHOW not in efs:
        return None
    r = z["recall"][efs.index(EF_SHOW)]
    if pop != "all":
        r = r[z[pop]] if pop in z.files else r
    c = np.bincount(np.rint(np.asarray(r, np.float64) * 10).astype(int),
                    minlength=11)[:11]
    return {"n": int(len(r)), "mean": float(r.mean()), "counts": c.tolist()}


def main(M=16, efc=200, pop="hard", h=1, fname="fig_dist"):
    H.setup(13)
    cpath = RES / f"phi{h}_dist_M{M}_efc{efc}_{pop}.json"
    cache = json.loads(cpath.read_text()) if cpath.exists() else {}
    fig, axes = plt.subplots(2, len(COLLS), figsize=(3.3 * len(COLLS), 4.4),
                             squeeze=False)
    drew, means = 0, {}
    for j, (c, clab) in enumerate(COLLS):
        for arm, lab, col in ARMS:
            d = phi_dist(c, arm, M, efc, pop, h, cache)
            if d:
                drew += 1
                n = np.array(d["counts"], float)
                axes[0][j].bar(PHI_LEVELS, n / max(n.sum(), 1), width=1.0,
                               color=col, alpha=0.45, edgecolor=col, lw=1.2,
                               zorder=2, label=lab)
                means[(0, j)] = (means.get((0, j), "")
                                 + f"{lab} {d['mean']:.2f}\n")
                axes[0][j].axvline(d["mean"], color=col, lw=1.4,
                                   ls=(0, (1, 2)), zorder=3)
            r = rec_dist(c, arm, M, efc, pop)
            if r:
                drew += 1
                n = np.array(r["counts"], float)
                # spacing 0.1, so the mass is divided by it to enclose area 1
                axes[1][j].bar(REC_LEVELS, n / max(n.sum(), 1) / 0.1, width=0.1,
                               color=col, alpha=0.45, edgecolor=col, lw=1.2,
                               zorder=2, label=lab)
                means[(1, j)] = (means.get((1, j), "")
                                 + f"{lab} {r['mean']:.3f}\n")
                axes[1][j].axvline(r["mean"], color=col, lw=1.4,
                                   ls=(0, (1, 2)), zorder=3)
        axes[0][j].set_title(clab, pad=8)
        axes[0][j].set_xticks(PHI_LEVELS)
        axes[0][j].set_xlim(0.4, 10.6)
        axes[0][j].set_xlabel(f"$\\varphi_{h}$")
        axes[1][j].set_xticks(REC_LEVELS)
        axes[1][j].set_xticklabels([f"{v:.1f}" for v in REC_LEVELS], fontsize=10)
        axes[1][j].set_xlim(-0.08, 1.08)
        axes[1][j].set_xlabel(f"Recall@10   (ef = {EF_SHOW})")
        # The arm names are the same in all eight panels; printing them eight
        # times costs the space the distributions are drawn in. The shared
        # legend below carries them, and what stays on the panel is the pair of
        # means, which differs per panel and is the number being compared.
        for r_ in (0, 1):
            txt = (means.get((r_, j)) or "").rstrip()
            if txt:
                axes[r_][j].text(0.97, 0.95, txt, transform=axes[r_][j].transAxes,
                                 ha="right", va="top", fontsize=8.2,
                                 linespacing=1.35)
    if not drew:
        plt.close(fig)
        return False
    axes[0][0].set_ylabel("density")
    axes[1][0].set_ylabel("density")
    handles = [Line2D([], [], color=col, lw=6, alpha=0.55, label=lab)
               for _, lab, col in ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, -0.004))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"{fname}.{e}")
    plt.close(fig)
    cpath.write_text(json.dumps(cache, indent=2))
    print(f"wrote figures/{fname}.pdf / .png  ({drew} distributions)")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--pop", default="hard", choices=("hard", "sample", "all"))
    ap.add_argument("--hops", type=int, default=1)
    a = ap.parse_args()
    if not main(a.M, a.efc, a.pop, a.hops):
        print("no cells for this setting")

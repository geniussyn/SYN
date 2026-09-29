"""What each knob trades, on an axis the target set cannot move.

The four knobs with a continuous range are swept here rather than shown as bars,
and they are read with screen_arm.py rather than with a timed search, for a
reason the bar figure cannot escape: recall@10 is measured against each arm's
own exact top-10, so an arm that moves its target set to an easier one improves
that number without repairing anything. phi-hat_1 has no such freedom. It is the
number of components the target set falls into under the best graph a builder
with out-degree maxM0 could make, so it needs no index, carries no build noise,
and bounds the real phi_1 from below.

Two curves per panel, on twin axes:

    phi-hat_1               left, lower is better -- the fragmentation itself
    exhaustive MRR@10       right, higher is better -- what the ranking cost

A knob that lowers phi-hat_1 while holding MRR is a win; one that lowers both is
buying geometry with the encoder's ranking, which is the trade the paper has to
show rather than assert. The untrained arm is a rule on each axis.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import housestyle as H                                           # noqa: E402
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402
from paths import ROOT                                           # noqa: E402

RES, OUT = ROOT / "results", ROOT / "figs" / "final"
C = "msmarco-1m"
BASE, SHIPPED = "vanilla", "syn"

# panel -> (x label, log x, [(arm tag, x value)])  -- the shipped point is in
# every panel at its own setting, so the four curves share one anchor
PANELS = [
    ("$\\lambda_{pull}$", True,
     [("abl-nopull", 0.0), ("abl-lam0.3", 0.3), (SHIPPED, 1.0),
      ("abl-lam2.0", 2.0), ("abl-lam3.0", 3.0)]),
    ("$\\tau_p$", True,
     [("abl-tp0.005", 0.005), (SHIPPED, 0.02), ("abl-tp0.05", 0.05),
      ("abl-tp0.1", 0.1)]),
    ("$L$", True,
     [("abl-L32", 32), ("abl-L130", 130), (SHIPPED, 1100)]),
    ("training queries", True,
     [("abl-n50", 50), ("abl-n100", 100), ("abl-n250", 250), (SHIPPED, 500)]),
]
PHI, MRR = H.BLUE, H.RED


def read(arm):
    p = RES / f"screen_{C}__{arm}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d["phi1_hat"], d["exhaustive_mrr10"]


def main(fname="fig_sweep_ablation"):
    base = read(BASE)
    drawn = 0
    H.setup(13)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.3 * len(PANELS), 3.7),
                             squeeze=False)
    for j, (xlab, logx, pts) in enumerate(PANELS):
        ax = axes[0][j]
        have = [(x, read(a)) for a, x in pts]
        have = [(x, v) for x, v in have if v]
        if len(have) < 2:
            ax.text(0.5, 0.5, "not measured", ha="center", va="center",
                    transform=ax.transAxes, color="0.5", fontsize=11)
            ax.set_xlabel(xlab)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        drawn += 1
        have.sort()
        xs = [x for x, _ in have]
        ax.plot(xs, [v[0] for _, v in have], color=PHI, marker="o",
                markersize=6.4, lw=2.0, markerfacecolor=PHI,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        if base:
            ax.axhline(base[0], color=PHI, lw=1.2, ls=(0, (1, 3)), zorder=2)
        ax.set_ylabel("$\\hat{\\varphi}_1$" if j == 0 else None, color=PHI)
        ax.tick_params(axis="y", colors=PHI)

        tw = ax.twinx()
        tw.plot(xs, [v[1] for _, v in have], color=MRR, marker="s",
                markersize=6.0, lw=2.0, ls="--", markerfacecolor=MRR,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        if base:
            tw.axhline(base[1], color=MRR, lw=1.2, ls=(0, (1, 3)), zorder=2)
        tw.set_ylabel("exhaustive MRR@10" if j == len(PANELS) - 1 else None,
                      color=MRR)
        tw.tick_params(axis="y", colors=MRR)
        tw.grid(False)

        if logx and min(x for x in xs if x > 0) > 0:
            pos = [x for x in xs if x > 0]
            if len(pos) == len(xs):
                ax.set_xscale("log")
        ax.set_xlabel(xlab)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=10)
        ax.minorticks_off()
    if not drawn:
        plt.close(fig)
        return False

    fig.legend(handles=[
        Line2D([], [], color=PHI, lw=2.2, marker="o", markersize=6.4,
               markerfacecolor=PHI, markeredgecolor=PHI,
               label="$\\hat{\\varphi}_1$  (lower is better)"),
        Line2D([], [], color=MRR, lw=2.2, ls="--", marker="s", markersize=6.0,
               markerfacecolor=MRR, markeredgecolor=MRR,
               label="exhaustive MRR@10  (higher is better)"),
        Line2D([], [], color="0.45", lw=1.2, ls=(0, (1, 3)),
               label="untrained")],
        loc="upper center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{e}")
    plt.close(fig)
    print(f"wrote figs/final/{fname}.pdf / .png  ({drawn}/{len(PANELS)} panels)")
    for xlab, _, pts in PANELS:
        row = [(x, read(a)) for a, x in pts]
        got = [f"{x:g}:{v[0]:.3f}/{v[1]:.4f}" for x, v in row if v]
        miss = [a for a, x in pts if not read(a)]
        print(f"  {xlab:18s} " + "  ".join(got) +
              (f"    missing: {', '.join(miss)}" if miss else ""))
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_sweep_ablation")
    a = ap.parse_args()
    if not main(a.out):
        print("no screened arms yet")

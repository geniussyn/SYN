"""What the build parameter buys, and what SYN lets you stop paying for it.

Section 5.5 sweeps M for what it does to search time. The question a system
builder actually asks is the other way round: at the latency budget I have, how
much graph do I have to build to hit my recall -- and the answer is what SYN
changes.

Left panel, per collection: recall@10 on the hard decile at a fixed time budget,
against M. Two budgets, 1 ms and 5 ms, because the answer should not depend on
picking one. Colour is the budget, line style the arm.

The reading is a horizontal one. On MS MARCO at 1 ms the untrained arm climbs
0.653 -> 0.740 -> 0.764 as M goes 8 -> 16 -> 32, and SYN starts at 0.957 with
the smallest graph. Doubling M twice does not carry the untrained arm to where
SYN already is at M = 8; the same holds at 5 ms and on FEVER. So the M sweep and
the repair are not two ways to buy the same thing, and more degree does not
substitute for unfragmenting the target set.

Right panel: the same fact in minutes. The best the untrained arm reaches at any
M in the sweep, beside the cheapest SYN build that matches or beats it -- 7.5
against 3.0 minutes on MS MARCO, 10.7 against 3.5 on FEVER. The axis is build
time rather than the percentage saved, because a percentage hides its own
magnitude: 59% of nine minutes and 59% of nine hours are different claims, and
this one is minutes, which is worth seeing before anyone budgets for it. Both
pairs are floored by the sweep, not by the method: M = 8 is the smallest degree
measured, so the saving is a lower bound and a smaller graph was never tried.

Index size follows M the same way and is identical between the arms, being a
function of n, dim and M alone: 3.89, 3.95 and 4.07 GB at M = 8, 16 and 32. The
saving is therefore in build time and in memory at once. Encoding is unchanged
in both arms -- 2249 against 2235 docs/s -- because SYN is a LoRA merged into
the encoder, which changes the weights and not the shape.
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

RES, OUT = ROOT / "results", ROOT / "figures"
COLLS = [("msmarco-1m", "MS MARCO 1M"), ("fever", "FEVER 1M")]
# Colour is the arm, the way it is in every other figure that draws these two:
# vanilla red, SYN blue. The budget is the marker, because it is the dimension
# a reader scans within a colour rather than across.
ARMS = [("vanilla", "Vanilla", H.RED), ("syn", "SYN", H.BLUE)]
BUDGETS = [(1.0, "1 ms", "o"), (5.0, "5 ms", "^")]
MS_SWEEP = [8, 16, 32]
EFC = 200


def cell(corpus, arm, M):
    """(time, recall) on the hard decile, plus the build's wall clock."""
    p = RES / f"cell_{corpus}__{arm}__M{M}__own_efc{EFC}.json"
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    g = [(r["ms_per_query_hard"], r["recall_hard"]) for r in d["grid"]
         if "recall_hard" in r]
    # the deep grid carries the budgets the ordinary sweep runs past
    g += [(r["ms_per_query_hard"], r["recall_hard"])
          for r in d.get("grid_deep_hard", [])]
    bt = d.get("build_time_s")
    return sorted(set(g)), (bt / 60.0 if bt else None)


def reach(g, ms):
    """The best recall the arm reaches without exceeding the budget."""
    under = [y for x, y in g if x <= ms]
    return max(under) if under else None


def main(fname="fig_build_cost"):
    H.setup(13)
    fig, axes = plt.subplots(1, len(COLLS) + 1,
                             figsize=(4.6 * (len(COLLS) + 1), 3.8),
                             squeeze=False)
    drew, saving = 0, []
    for j, (c, clab) in enumerate(COLLS):
        ax = axes[0][j]
        for ms, blab, mk in BUDGETS:
            for a, _, colour in ARMS:
                xs, ys = [], []
                for M in MS_SWEEP:
                    g, _ = cell(c, a, M)
                    if not g:
                        continue
                    r = reach(g, ms)
                    if r is not None:
                        xs.append(M)
                        ys.append(r)
                if len(xs) < 2:
                    continue
                drew += 1
                ax.plot(xs, ys, color=colour, marker=mk,
                        markersize=(6.4 if mk == "o" else 7.4), lw=1.9,
                        markerfacecolor=colour, markeredgecolor="white",
                        markeredgewidth=0.8,
                        zorder=3 if a == "syn" else 2)
        ax.set_xscale("log")
        ax.set_xticks(MS_SWEEP)
        ax.set_xticklabels([str(m) for m in MS_SWEEP])
        ax.minorticks_off()
        ax.set_xlabel("$M$")
        ax.set_title(clab, pad=7)
        if j == 0:
            ax.set_ylabel("Recall@10\n(hard decile)")

        # the saving, computed from the same cells the panel is drawn from
        for ms, blab, _ in BUDGETS:
            best = []
            for M in MS_SWEEP:
                g, b = cell(c, "vanilla", M)
                if g and b:
                    r = reach(g, ms)
                    if r is not None:
                        best.append((r, M, b))
            if not best:
                continue
            rv, Mv, bv = max(best)
            for M in MS_SWEEP:
                g, b = cell(c, "syn", M)
                if g and b and reach(g, ms) is not None and reach(g, ms) >= rv:
                    saving.append((clab, blab, Mv, bv, M, b, 1 - b / bv))
                    break
    if not drew:
        plt.close(fig)
        return False

    # Right panel: the build itself, in minutes. Plotting the saving as a
    # percentage hid the thing the percentage is of -- a reader cannot tell
    # whether 59% is nine minutes or nine hours, and at this size it is minutes,
    # which is worth knowing before anyone calls the method expensive. The two
    # budgets select the same pair on both collections (SYN at M=8 is already
    # past untrained at M=32), so the bars are one pair per collection and the
    # budget is no longer a dimension here.
    ax = axes[0][-1]
    pairs = []
    for clab, blab, Mv, bv, Ms, bs, sv in saving:
        if not any(p[0] == clab for p in pairs):
            pairs.append((clab, Mv, bv, Ms, bs, sv))
    if pairs:
        # The percentage used to float above the pair, where it landed on the
        # taller bar's own label. It belongs to the gap between the bars, so it
        # is drawn in the gap: a rule at the untrained height carried across the
        # SYN bar, and the drop measured against it.
        # The label sits to the right of each pair, so the pairs need room
        # between them: at unit spacing the first pair's percentage ran under
        # the second pair's bar.
        w, gap = 0.32, 1.5
        for i0, (clab, Mv, bv, Ms, bs, sv) in enumerate(pairs):
            i = i0 * gap
            ax.bar(i - w / 2, bv, width=w, color=H.RED, edgecolor="white",
                   linewidth=1.1, zorder=3)
            ax.bar(i + w / 2, bs, width=w, color=H.BLUE, edgecolor="white",
                   linewidth=1.1, zorder=3)
            ax.text(i - w / 2, bv + 0.22, f"$M${Mv} · {bv:.1f}", ha="center",
                    va="bottom", fontsize=9.5, color=H.RED)
            ax.text(i + w / 2, bs + 0.22, f"$M${Ms} · {bs:.1f}", ha="center",
                    va="bottom", fontsize=9.5, color=H.BLUE)
            ax.plot([i - w / 2, i + w / 2 + 0.34], [bv, bv], color="0.45",
                    lw=1.0, ls=(0, (2, 2)), zorder=2)
            ax.annotate("", xy=(i + w / 2 + 0.30, bs),
                        xytext=(i + w / 2 + 0.30, bv),
                        arrowprops=dict(arrowstyle="<->", color="0.35",
                                        lw=1.2, shrinkA=0, shrinkB=0),
                        zorder=4)
            ax.text(i + w / 2 + 0.37, (bv + bs) / 2, f"$-${100 * sv:.0f}%",
                    ha="left", va="center", fontsize=11.5, color="0.2",
                    zorder=5)
        ax.set_xticks([i * gap for i in range(len(pairs))])
        ax.set_xticklabels([p[0] for p in pairs], fontsize=10.5)
        ax.set_xlim(-0.55, (len(pairs) - 1) * gap + 0.95)
        ax.set_ylim(0, max(max(p[2], p[4]) for p in pairs) * 1.26)
        ax.set_ylabel("index build time (min)")
        ax.set_title("Same recall, cheaper graph", pad=7)
        ax.grid(axis="x", visible=False)

    handles = [Line2D([], [], color=col, lw=2.4, label=nm)
               for _, nm, col in ARMS]
    handles += [Line2D([], [], color="0.35", ls="none", marker=mk,
                       markersize=(6.4 if mk == "o" else 7.4),
                       markerfacecolor="0.35", markeredgecolor="white",
                       markeredgewidth=0.8, label=f"budget {lab}")
                for _, lab, mk in BUDGETS]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=10.5, bbox_to_anchor=(0.5, 1.015))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{e}")
    plt.close(fig)
    print(f"wrote figures/{fname}.pdf / .png  ({drew} curves)")
    for clab, blab, Mv, bv, Ms, bs, s in saving:
        print(f"  {clab:<12} {blab:<5} vanilla best at M={Mv} ({bv:.1f} min) "
              f"-> SYN at M={Ms} ({bs:.1f} min)  saving {s:.1%}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_build_cost")
    a = ap.parse_args()
    if not main(a.out):
        print("no build cells measured yet")

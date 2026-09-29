"""What the two build knobs cost, on two collections.

Rows are the knob -- graph degree M at a fixed construction budget, then the
construction budget efC at a fixed degree -- and columns are the collection. In
every panel the x axis is measured search time rather than ef, because ef is a
budget dial and a sparse graph spends it differently: comparing two degrees at
equal ef charges them different amounts of time for the same position.

The statement the rows make together is that the two knobs are not
interchangeable. SYN's curve barely moves with M until a floor and then falls off
it, and lowering efC costs more than it looks like it should, because a pool of
16 candidates does not survive hnswlib's pruning and the measured out-degree
collapses with it. The degree each cell actually built is in the legend, so the
second row can be read against the first at a matched degree rather than at a
matched knob.

Build seconds are not drawn. They are a property of the builder, not of the
comparison, and they are reported in the build-cost table.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import housestyle as H                                          # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.lines import Line2D                             # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator         # noqa: E402
from paths import ROOT                                          # noqa: E402

RES, OUT = ROOT / "results", ROOT / "figs" / "final"
COLLS = [("msmarco-1m", "MS MARCO 1M"), ("fever", "FEVER")]
ARMS = [("vanilla", "Vanilla", "--", "none"), ("syn", "SYN", "-", None)]
# row -> (label, the setting that varies, the one held fixed, colours+markers)
SWEEPS = [
    ("M", "graph degree M", [(8, H.RED, "s"), (16, H.BLUE, "^"),
                             (32, H.GREEN, "D")], 200),
    ("efc", "construction budget efC", [(100, H.RED, "s"), (200, H.BLUE, "^"),
                                        (300, H.GREEN, "D")], 16),
]
POP = "hard"


def cell(corpus, arm, M, efc):
    p = RES / f"cell_{corpus}__{arm}__M{M}__own_efc{efc}.json"
    return json.loads(p.read_text()) if p.exists() else None


def curve(d):
    """(ms, recall) on the arm's own hard decile, grid plus deep tail."""
    g = [(r["ms_per_query_hard"], r["recall_hard"]) for r in d["grid"]
         if "ms_per_query_hard" in r]
    g += [(r["ms_per_query_hard"], r["recall_hard"])
          for r in (d.get("grid_deep_hard") or [])]
    return sorted(g)


def main(fname="fig_sweep"):
    H.setup(13)
    fig, axes = plt.subplots(len(SWEEPS), len(COLLS),
                             figsize=(5.6 * len(COLLS), 4.6 * len(SWEEPS)),
                             sharey=True, squeeze=False)
    drew = 0
    deg = {}
    for r, (knob, klab, values, fixed) in enumerate(SWEEPS):
        for j, (c, clab) in enumerate(COLLS):
            ax = axes[r][j]
            for v, col, mk in values:
                M, efc = (v, fixed) if knob == "M" else (fixed, v)
                for arm, _, ls, fill in ARMS:
                    d = cell(c, arm, M, efc)
                    if not d:
                        continue
                    g = curve(d)
                    if not g:
                        continue
                    drew += 1
                    deg.setdefault((knob, v), []).append(d.get("measured_out_degree"))
                    ax.plot([x for x, _ in g], [y for _, y in g],
                            color=col, ls=ls, marker=mk, markersize=5.2,
                            markevery=2,
                            markerfacecolor=(col if fill is None else "none"),
                            markeredgecolor=("white" if fill is None else col),
                            markeredgewidth=(0.7 if fill is None else 1.2),
                            lw=1.8, zorder=3 if arm == "syn" else 2)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v_, _: f"{v_:g}"))
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0),
                                                  numticks=12))
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda v_, _: ""))
            ax.set_ylim(0, 1.02)
            if r == 0:
                ax.set_title(clab, pad=7)
            if j == 0:
                ax.set_ylabel(f"Recall@10 (hard decile)\n{klab}")
    if not drew:
        plt.close(fig)
        return False

    def lab(knob, v, fixed):
        ds = [x for x in deg.get((knob, v), []) if x]
        d_ = f", degree {sum(ds)/len(ds):.1f}" if ds else ""
        return (f"M={v} (efC={fixed}{d_})" if knob == "M"
                else f"efC={v} (M={fixed}{d_})")

    handles = []
    for knob, _, values, fixed in SWEEPS:
        handles += [Line2D([], [], color=col, lw=2.2, marker=mk, markersize=6,
                           markerfacecolor=col, markeredgecolor=col,
                           label=lab(knob, v, fixed)) for v, col, mk in values]
    handles += [Line2D([], [], color="0.35", ls=ls, marker="o", markersize=6,
                       markerfacecolor=("0.35" if fill is None else "none"),
                       markeredgecolor="0.35", label=nm)
                for _, nm, ls, fill in ARMS]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 1.0))
    fig.supxlabel("Search time per Query (ms)", y=0.015)
    fig.tight_layout(rect=(0, 0.03, 1, 0.90), w_pad=1.0)
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{e}")
    plt.close(fig)
    print(f"wrote figs/final/{fname}.pdf / .png  ({drew} curves)")
    return True


if __name__ == "__main__":
    if not main():
        print("no cells yet")

"""Two repairs, one plot: NGFix fixes the graph, SYN fixes what it is built from.

Colour carries the embedding and line style carries the graph repair, so the
figure is read twice. Down a colour: what the graph repair buys on a fixed
embedding -- the question HNSW-NGFix* is about. Across colours at one style:
what the embedding buys under a fixed repair. If the two were substitutes the
curves would meet; if they compose, the SYN band moves and keeps its own
internal spread.

Every curve comes from the authors' own binary at their own defaults, on one
bottom graph per arm:

    none          their HNSW, M=16 MEX=48 efC=500, no repair
    HNSW-NGFix    their method, both repair stages

The two stages were also measured on their own -- NGFix is the neighbouring
graph defect repair, RFix the reachability repair -- and they are in the JSON
under those keys. They are not drawn: RFix alone moves nothing anywhere
(0.99-1.02x on all eight arms) and NGFix alone accounts for the whole of the
combined effect, so three nearly coincident curves per arm cost the panel its
legibility and answered a question about the baseline's internals rather than
about the baseline. The stage numbers belong in a sentence, not in the figure.

The early-termination baseline that used to share these axes is not drawn. It
was measured and is in the lab, but it is out of this submission, and a figure
is not the place to keep something the text does not argue.

Two collections by default, which is what the main text takes; --colls all
draws the other two for the appendix.

MILLISECONDS ARE THE BASELINES' HARNESSES, not cpp/hnsw_bench. Nothing here is
comparable with the other figures, and nothing needs to be.
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

RES, OUT = ROOT / "results", ROOT / "figures"
ARMS = [("vanilla", "Vanilla", H.RED), ("syn", "SYN", H.BLUE)]
# key -> (label, line style, marker, size)
STAGES = [("bottom", "no repair", "--", "o", 6.2),
          ("both", "HNSW-NGFix", "-", "X", 7.4)]
POPS = [("hard", "hard decile"), ("test", "all query")]
# Both repairs are argued at high recall -- a declarative target of 0.90 and
# above, and a graph fix whose whole effect is in the last tenth -- so the axis
# starts where the argument does. Below this the curves are ordered and nothing
# more; the full range is in the family figure.
#
# The floor is per ROW as well as per column, because the two rows are two
# different populations and their recall lives in different places. The hard
# decile is where the low budgets still fail, and it starts at 0.70. Over all
# queries the same budgets are already past 0.85 everywhere -- the lowest point
# in that row is FEVER's 0.532 and the next is NQ's 0.645, both from the single
# cheapest efSearch -- so a 0.70 floor there is mostly blank paper under the
# curves.
#
# Quora is cropped harder again in both rows: it is the collection every method
# already handles, its worst hard-decile curve stops at 0.752 and its worst
# all-query curve at 0.910, and the declarative targets it is being asked about
# all sit in the top twentieth. Panels are therefore NOT on a shared scale and
# each carries its own ticks.
YFLOOR = {("hard", None): 0.70, ("hard", "quora"): 0.90,
          ("test", None): 0.85, ("test", "quora"): 0.95}


def floor_for(pop, corpus):
    return YFLOOR.get((pop, corpus), YFLOOR[(pop, None)])


def yticks(floor):
    step = 0.1 if floor <= 0.75 else (0.05 if floor <= 0.88 else 0.025)
    n = int(round((1.0 - floor) / step))
    return [round(floor + i * step, 3) for i in range(n + 1)]
ALL_COLLS = [("msmarco-1m", "MS MARCO 1M"), ("fever", "FEVER 1M"),
             ("nq", "NQ 1M"), ("quora", "Quora 523K")]
MAIN = ("msmarco-1m", "fever")


def main(fname="fig_ngfix", colls=MAIN):
    COLLS = [c for c in ALL_COLLS if c[0] in colls]
    H.setup(13)
    nr, nc = len(POPS), len(COLLS)
    fig, axes = plt.subplots(nr, nc, figsize=(4.3 * nc, 3.5 * nr),
                             squeeze=False)
    drew, note = 0, {}
    for r, (pop, plab) in enumerate(POPS):
        for cix, (corpus, clab) in enumerate(COLLS):
            ax = axes[r][cix]
            for arm, _, col in ARMS:
                p = RES / f"ngfix_{corpus}__{arm}.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                note.update(d["params"])
                note["n_history"] = d.get("n_history")
                for key, _, ls, mk, ms in STAGES:
                    g = (d.get("grids") or {}).get(key, {}).get(pop)
                    if not g:
                        continue
                    drew += 1
                    ax.plot([x["ms"] for x in g], [x["recall"] for x in g],
                            color=col, ls=ls, marker=mk, markersize=ms,
                            markevery=2, markerfacecolor=col,
                            markeredgecolor="white", markeredgewidth=0.7, lw=1.8,
                            zorder=3 if arm == "syn" else 2)
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0),
                                                  numticks=10))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
            floor = floor_for(pop, corpus)
            ax.set_ylim(floor, 1.0 + (1.0 - floor) * 0.017)
            ax.set_yticks(yticks(floor))
            if r == 0:
                ax.set_title(clab, pad=7)
            ax.set_ylabel(f"Recall@10\n({plab})" if cix == 0 else None,
                          labelpad=6)
    if not drew:
        plt.close(fig)
        return False

    handles = ([Line2D([], [], color=c, lw=2.6, label=n) for _, n, c in ARMS]
               + [Line2D([], [], color="0.35", ls=ls, marker=mk, markersize=ms,
                         markerfacecolor="0.35", markeredgecolor="0.35", label=n)
                  for _, n, ls, mk, ms in STAGES]
)
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, 1.0))
    fig.supxlabel("Search time per Query (ms)", y=0.012)
    fig.tight_layout(rect=(0, 0.025, 1, 0.915))
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{e}")
    plt.close(fig)
    print(f"wrote figures/{fname}.pdf / .png  ({drew} curves, {nr}x{nc})")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--colls", default="main", choices=("main", "all"),
                    help="main: the two collections the text argues over. "
                         "all: the appendix version.")
    a = ap.parse_args()
    colls = MAIN if a.colls == "main" else [c for c, _ in ALL_COLLS]
    out = "fig_ngfix" if a.colls == "main" else "figA_ngfix_all"
    if not main(out, colls):
        print("no ngfix results yet")

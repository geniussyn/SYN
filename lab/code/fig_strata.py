"""HNSW by difficulty percentile: the average query, the decile, the 5th, the 1st.

Five collections. MS MARCO is the one the adapter was trained on; FEVER, NQ,
Quora and MS COCO are zero-shot.

The strata are nested and cut from the same seed-777 selector, so one index and
one set of returned neighbours answers all four -- but each is timed on its own
queries, because a harder stratum costs more per query and a curve that borrowed
the full set's x axis would shift left by up to a fifth of its own scale.

Colour carries the arm and line style carries the stratum, so the figure is read
down a colour: the vertical gap between two curves of the same colour is what
difficulty costs that embedding, and the horizontal gap between the two colours
at one stratum is the speedup there. The average query hides the claim -- both
arms are near their ceiling at a fraction of a millisecond -- and the percentiles
are where it lives.

Recall is measured against each arm's own fp32 exact top-10, so a curve reaching
1.0 means the index returned that arm's own neighbours, not that the embedding
ranks better.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import housestyle as H                                           # noqa: E402
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator          # noqa: E402
from paths import ROOT                                           # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "figs" / "final"
# Four collections, then the same 1M English passages with the query language
# moved. The last two are not more collections: the documents, the judgements
# and the adapter are identical to the first panel and only the text of the
# query changes, which is why they are labelled by the drift of their queries
# rather than by a corpus name. MS COCO was measured as a fifth collection and
# is not in the paper; its strata files are in results/excluded/.
# the label is the name; the corpus size is appended from the measurement, so a
# panel cannot claim a size the run it draws did not have
COLLS = [("msmarco-1m", "MS MARCO"), ("fever", "FEVER"), ("nq", "NQ"),
         ("quora", "Quora"),
         ("msmarco-fr", "mMARCO French"),
         ("msmarco-ar", "mMARCO Arabic"),
         # one scale point: the same collection at its full size, same adapter
         ("msmarco-8m", "MS MARCO")]
# colour = arm
ARMS = [("vanilla", "Vanilla", H.RED), ("syn", "SYN", H.BLUE)]
# line style and marker = stratum, widest first; the percentile names are the
# worst 10 / 5 / 1 per cent by the selector's mean ANN recall
STRATA = [("all", "Avg", "-", "o", 6.2),
          ("P90", "decile (worst 10%)", "--", "s", 6.4),
          ("P95", "5th pct (worst 5%)", ":", "^", 7.6),
          ("P99", "1st pct (worst 1%)", "-.", "X", 7.4)]
# one marker every other point: the ef grid is dense enough that drawing all of
# them turns eight curves into a smear
EVERY = 2
TARGETS = (0.90, 0.95, 0.99)
# Where each panel starts. 0.30 is where the four strata stop being separable on
# the three hard collections. Quora is the collection everything already handles
# -- even its worst percentile never drops below 0.465 -- so it is cropped to the
# band its curves actually occupy rather than given a floor it never reaches.
# Panels are therefore on different scales and each carries its own ticks.
YFLOOR_DEFAULT = 0.30
YFLOOR = {"quora": 0.80}


def yticks(floor):
    step = 0.1 if floor <= 0.45 else 0.05
    n = int(round((1.0 - floor) / step))
    return [round(floor + i * step, 3) for i in range(n + 1)]


def load(corpus, arm, M=16, efc=200):
    p = RES / f"strata_{corpus}__{arm}__M{M}__strata_efc{efc}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main(M=16, efc=200):
    H.setup(13)
    # NOT sharey: Quora is cropped to its own band
    have = [c for c, _ in COLLS
            if (RES / f"strata_{c}__vanilla__M16__strata_efc200.json").exists()]
    nc = 4 if len(have) > 6 else 3
    nr = -(-max(len(have), 1) // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(6.4 * nc, 5.3 * nr), squeeze=False)
    drew = False
    sizes = {}
    for c, _ in COLLS:
        d = load(c, "vanilla", M, efc) or load(c, "syn", M, efc)
        if d:
            n = d["n_docs"]
            sizes[c] = (f"{n/1e6:g}M" if n >= 1_000_000
                        else f"{n/1e3:.0f}K")
    for ax, (c, lab) in zip(axes.ravel(), COLLS):
        for R in TARGETS:
            ax.axhline(R, color="0.55", lw=0.7, ls=(0, (1, 3)), zorder=1)
        for arm, _, col in ARMS:
            d = load(c, arm, M, efc)
            if not d:
                continue
            for key, _, ls, mk, ms in STRATA:
                s = (d.get("strata") or {}).get(key)
                if not s:
                    continue
                g = sorted((r for r in s["grid"] if "probe_for" not in r),
                           key=lambda r: r["ef"])
                if not g:
                    continue
                drew = True
                ax.plot([r["ms"] for r in g], [r["recall"] for r in g],
                        color=col, ls=ls, marker=mk, markersize=ms,
                        markevery=EVERY, markerfacecolor=col,
                        markeredgecolor="white", markeredgewidth=0.8, lw=1.9,
                        zorder=3 if arm == "syn" else 2)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0), numticks=12))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        # the 2x/5x minors are worth labelling only where the decades are far
        # apart on the page; past 100 ms they collide with the decade beside them
        ax.xaxis.set_minor_formatter(
            FuncFormatter(lambda v, _: f"{v:g}" if v < 100 else ""))
        floor = YFLOOR.get(c, YFLOOR_DEFAULT)
        ax.set_ylim(floor, 1.0 + (1.0 - floor) * 0.007)
        ax.set_yticks(yticks(floor))
        sz = sizes.get(c)
        ax.set_title(f"{lab} {sz}" if sz else lab, pad=7)
    if not drew:
        plt.close(fig)
        return False
    # one label for the grid: at 2x2 the per-row label is longer than the panel
    # is tall and the two run into each other
    fig.supylabel("Recall@10", x=0.012)
    handles = ([Line2D([], [], color=col, lw=2.6, label=nm)
                for _, nm, col in ARMS]
               + [Line2D([], [], color="0.35", ls=ls, marker=mk, markersize=ms,
                         markerfacecolor="0.35", markeredgecolor="0.35",
                         label=nm) for _, nm, ls, mk, ms in STRATA])
    # with a spare cell the legend goes in it; with a full grid it goes above
    spare = axes.ravel()[len(COLLS):]
    if len(spare):
        lax = spare[0]
        lax.set_axis_off()
        lax.legend(handles=handles, loc="center", fontsize=13, frameon=True)
        for a_ in spare[1:]:
            a_.set_axis_off()
    else:
        fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=12,
                   frameon=True, bbox_to_anchor=(0.5, 1.0))
    fig.supxlabel("Search time per Query (ms)", y=0.02)
    fig.tight_layout(rect=(0.022, 0.03, 1, 0.945))
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"fig_strata_M{M}.{e}")
    plt.close(fig)
    print(f"wrote figs/final/fig_strata_M{M}.pdf / .png")
    return True


if __name__ == "__main__":
    if not main():
        print("no strata files yet")

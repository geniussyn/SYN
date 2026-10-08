"""Four index families, four collections, in the house style.

Colour and marker shape both carry the family -- square, triangle, diamond,
plus, all angular and none a rotation of another, so they stay apart at print
size and in greyscale -- while the arm is carried by line style and marker
fill: Vanilla dashed and hollow, SYN solid and filled. Two redundant channels
per family means the panel survives a monochrome printout, and neither channel
is spent on the arm, which the fill already separates. That keeps the
comparison the figure is making -- within a family, across the two embeddings
-- in one panel, and makes the second reading free: the SYN band sits above
every Vanilla curve, so the repair is not a property of one builder.

Columns are collections, rows are the population the recall is measured over:
the arm's own hard decile above, every query below. Reading down a column is
what the tail costs.


The axes are cropped to where the figure has something to say. Vertically they
start at 0.4, because below that the three non-HNSW families lie on top of each
other on every collection and no scale separates identical data. Horizontally
each panel stops once every curve in it has reached 0.95, which removes a decade
of flat saturated tail and gives that width back to the part being read. What
falls outside is not lost: the measured 0.90 / 0.95 / 0.99 targets are a table,
and the full range is in the per-stratum figure.

Every panel is drawn at the same setting, the small one: M=8 for HNSW, R=8 for
NSG and Vamana, nlist=4096 for IVF. A graph family's degree is a memory
budget, so curves are not comparable across it. The settings are recorded in
results/family_*__own_M8.json and belong in the caption, not on the plot. The
claim they serve is not "SYN wins at a matched degree" but "SYN built small beats
the baseline built normally", which the R=25 runs measure separately.

HNSW is read from its own family run rather than from the headline cell, so
all four panels have the same provenance. MS MARCO is the only collection the
adapter was trained on -- the other three are zero-shot.
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

RES, OUT = ROOT / "results", ROOT / "figures"

COLLS = [("msmarco-1m", "MS MARCO"), ("fever", "FEVER"),
         ("nq", "NQ"), ("quora", "Quora")]
# corpus sizes, read from the measurements at draw time
N_DOCS = {}
# family key -> (label, colour, marker)
FAMILIES = [
    ("hnswlib",   "HNSW",             H.RED,    "s"),
    ("faiss-nsg", "NSG",              H.BLUE,   "^"),
    ("faiss-ivf", "IVF",              H.GREEN,  "D"),
    ("vamana",    "Vamana (DiskANN)", H.ORANGE, "P"),
]
# arm -> (label, line style, marker fill)
ARMS = [("vanilla", "Vanilla", "--", "none"),
        ("syn", "SYN", "-", None)]
MS = {"s": 6.4, "^": 7.6, "D": 5.8, "P": 8.0}   # equal visual weight
# one marker every other point: the grid is dense enough that drawing all of
# them turns eight curves into a smear
EVERY = 2
TAG = "__own_M8"
# Every panel starts at 0.80. Below it eight curves climb through the same
# narrow band of milliseconds and lie on top of one another -- the ordering is
# visible but nothing else is -- while above it they fan out over two or three
# decades, which is where the whole comparison lives: the time each family needs
# for the last fifth of the recall. Cropping to it is what makes the panel
# readable, and it costs nothing that is argued anywhere: the sub-0.80 points
# are still in results/family_*__own_M8.json, and no claim in the paper is made
# below 0.90. The scale is shared by every panel, so the columns can be read
# against each other.
# ...except the all-query row on the harder collections. The all-query grid was
# measured to efSearch 512 and no further -- the hard-decile grid runs to 65k --
# and on FEVER the untrained arms have not arrived by then: Vamana tops out at
# 0.788 and NSG at 0.809, so at a 0.80 floor one baseline vanishes entirely and
# another is a single marker. That reads as a missing measurement when it is a
# truncated one. The row drops to 0.70, where all four baselines show the shape
# of their approach, and the two rows are then not on one scale -- which is why
# each row's floor is drawn on its own axis rather than inherited.
YFLOOR = {"hard": 0.80, "all": 0.70}
# The right edge: where every curve in the panel has reached this. Raised with
# the floor -- a panel that begins at 0.80 has to show the approach to 1.0 or it
# is cropped to a sliver -- and 0.99 is the deepest declarative target measured.
CROP_AT = 0.99


def series(fam, corpus, arm, tag, pop):
    """(ms, recall) for one family and arm on one collection's stratum."""
    p = RES / f"family_{fam}_{corpus}__{arm}{tag}.json"
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    par = d.get("params") or {}
    note = ", ".join(f"{k}={v}" for k, v in par.items() if k in ("R", "M", "nlist"))
    if pop == "hard" and d.get("grid_hard"):
        g = [(r["ms_per_query"], r["recall"]) for r in d["grid_hard"]
             if "recall" in r]
    else:
        g = [(r["ms_per_query"], r[f"recall_{pop}"]) for r in d["grid"]]
    return sorted(g), note


POPS = [("hard", "hard decile"), ("all", "all query")]


def yticks(floor):
    """Ticks from `floor` to 1, at a spacing the span can carry."""
    step = 0.2 if floor <= 0.45 else (0.1 if floor <= 0.75 else 0.05)
    n = int(round((1.0 - floor) / step))
    return [round(floor + i * step, 3) for i in range(n + 1)]


def main(fname="fig_families", tag=TAG):
    H.setup(13)
    nr, nc = len(POPS), len(COLLS)
    # sharey by ROW, not across the whole grid. Within a row the four columns
    # must be on one scale or they cannot be read against each other; between
    # the rows they must not be, because the all-query row starts lower -- its
    # grid stops at efSearch 512 and the untrained arms have not arrived by
    # then. Sharing all eight panels silently gives every panel the last floor
    # that was set, which is how the hard-decile row ended up drawn at 0.70.
    fig, axes = plt.subplots(nr, nc, figsize=(4.3 * nc, 3.5 * nr),
                             sharey="row", squeeze=False)
    note, drew = {}, 0
    for c, _ in COLLS:
        for fam, *_ in FAMILIES:
            p = RES / f"family_{fam}_{c}__vanilla{tag}.json"
            if p.exists():
                nd = (json.loads(p.read_text()) or {}).get("n_docs")
                if nd:
                    N_DOCS[c] = (f"{nd/1e6:g}M" if nd >= 1_000_000
                                 else f"{nd/1e3:.0f}K")
                break
    for r, (pop, plab) in enumerate(POPS):
        floor = YFLOOR[pop]
        for cix, (c, clab) in enumerate(COLLS):
            ax = axes[r][cix]
            xmax, xmin = 0.0, float("inf")
            for fam, lab, col, mk in FAMILIES:
                for arm, _, ls, fill in ARMS:
                    g, n = series(fam, c, arm, tag, pop)
                    if not g:
                        continue
                    note.setdefault(fam, n)
                    drew += 1
                    # the width this curve needs: where it first reaches
                    # CROP_AT, or its deepest point if it never does
                    hit = [x for x, y in g if y >= CROP_AT]
                    xmax = max(xmax, hit[0] if hit else g[-1][0])
                    # and the left edge: the points under the floor are not
                    # drawn, so autoscaling on them would leave the panel
                    # opening on empty space
                    vis = [x for x, y in g if y >= floor]
                    if vis:
                        xmin = min(xmin, vis[0])
                    ax.plot([x for x, _ in g], [y for _, y in g],
                            color=col, ls=ls, marker=mk, markersize=H.ms(MS[mk]),
                            markevery=EVERY,
                            markerfacecolor=(col if fill is None else "none"),
                            markeredgecolor=("white" if fill is None else col),
                            markeredgewidth=(0.8 if fill is None else 1.3),
                            lw=H.LW, zorder=3 if arm == "syn" else 2)
            ax.set_xscale("log")
            # decades only: a panel spanning four of them cannot label the
            # 2x and 5x marks without the labels running into each other
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0),
                                                  numticks=12))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
            if xmax > 0:
                ax.set_xlim(left=(xmin * 0.85 if xmin < float("inf") else None),
                            right=xmax * 1.3)
            ax.set_ylim(floor, 1.0 + (1.0 - floor) * 0.025)
            ax.set_yticks(yticks(floor))
            if r == 0:
                n = N_DOCS.get(c)
                ax.set_title(f"{clab} {n}" if n else clab, pad=7)
            ax.set_ylabel(f"Recall@10\n({plab})" if cix == 0 else None,
                          labelpad=6)
    if not drew:
        plt.close(fig)
        return False

    fam_h = [Line2D([], [], color=c, lw=2.4, marker=m, markersize=MS[m] + 1.4,
                    markerfacecolor=c, markeredgecolor=c, label=l)
             for k, l, c, m in FAMILIES if k in note]
    arm_h = [Line2D([], [], color="0.35", ls=ls, marker="o", markersize=7.0,
                    markerfacecolor=("0.35" if fill is None else "none"),
                    markeredgecolor="0.35", label=nm)
             for _, nm, ls, fill in ARMS]
    fig.legend(handles=fam_h + arm_h, loc="upper center", ncol=6, fontsize=11,
               bbox_to_anchor=(0.5, 1.0))
    # the protocol -- one pinned core, and where each panel stops -- belongs in
    # the caption, not on the axis
    fig.supxlabel("Search time per Query (ms)", y=0.012)
    fig.tight_layout(rect=(0, 0.025, 1, 0.945))
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{e}")
    plt.close(fig)
    print(f"wrote figures/{fname}.pdf / .png  ({drew} curves, {nr}x{nc})")
    return True


if __name__ == "__main__":
    main()

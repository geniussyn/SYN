"""Five encoders, one question: is the repair a property of this embedding or of
embeddings.

Every arm is the same pipeline -- mine 500 MS MARCO training queries in that
encoder's own space, fit one LoRA adapter, apply it unchanged -- and every curve
is HNSW at the same build setting. What changes between curves is only which
encoder produced the space.

The encoders are ordered by width, because width is the alternative explanation
worth ruling out. A 2560-dimensional space has two and a half times the room of
the reference encoder to hold ten documents apart without their being each
other's neighbours, so if fragmentation were an artefact of a cramped 1024d
embedding the gap between the two arms should close as the legend goes right.
It does not, and it does not widen either: the three 1024d encoders alone span
12.7x, 7.3x and 6.5x at recall 0.95, a spread as large as the one across the
whole width range, and neither the narrowest nor the widest encoder sits at an
end of the ordering. Width does not order the effect, which is the claim the
figure supports -- not that width makes it larger. A 4096d arm (Qwen3-8B) was
planned and dropped: its training and re-encodes were half the campaign's
remaining time, and the 4B arm already spans the range.

The encoder is carried by colour and the arm by line style, on shared axes,
because the comparison the figure exists to make is within an encoder -- is the
solid curve left of the dashed one of the same colour. Shared axes also put the
absolute cost on display: a 2560d distance costs two and a half times a 1024d
one, so the wide encoders sit right for a reason that has nothing to do with the
graph. Read the gap within a colour, not the position of a colour.

FEVER was measured for all five encoders and the same ordering holds there; it
is cut from the figure for space, and its numbers are in the paper's text.

Recall is each arm against its OWN exact top-10, the contract the rest of the
paper uses. Milliseconds are one pinned core, one thread, min of three.
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
# (arm suffix, label, width) ordered by width; "" is the paper's encoder
# encoder -> label, width, colour, marker; ordered by width
ENCODERS = [("-mini", "MiniLM-L6", 384,  H.GREEN,  "^"),
            ("",      "BGE-M3",    1024, H.RED,    "s"),
            ("-e5l",  "E5-large",  1024, H.BLUE,   "o"),
            ("-gte",  "GTE-large", 1024, H.ORANGE, "D"),
            ("-qw4",  "Qwen3-4B",  2560, "#6A3D9A", "P")]
COLLS = [("msmarco-1m", "MS MARCO 1M")]
# arm -> label, line style, marker fill
ARMS = [("vanilla", "Vanilla", "--", "none"),
        ("syn", "SYN", "-", None)]
YFLOOR = 0.80
CROP_AT = 0.99
EVERY = 2


def series(corpus, arm, pop):
    p = RES / f"family_hnswlib_{corpus}__{arm}__own_M8.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if pop == "hard" and d.get("grid_hard"):
        g = [(r["ms_per_query"], r["recall"]) for r in d["grid_hard"]
             if "recall" in r]
    else:
        g = [(r["ms_per_query"], r[f"recall_{pop}"]) for r in d["grid"]]
    return sorted(g) or None


def main(fname="fig_encoders", pop="hard"):
    have = [e for e in ENCODERS
            if all(series(c, f"{a}{e[0]}", pop) for c, _ in COLLS
                   for a, *_ in ARMS)]
    if not have:
        return False
    H.setup(13)
    fig, axes = plt.subplots(1, len(COLLS), figsize=(6.6 * len(COLLS), 3.5),
                             sharey=True, squeeze=False)
    drew = 0
    for j, (c, clab) in enumerate(COLLS):
        ax = axes[0][j]
        xmax, xmin = 0.0, float("inf")
        for suf, elab, dim, col, mk in have:
            for a, _, ls, fill in ARMS:
                g = series(c, f"{a}{suf}", pop)
                if not g:
                    continue
                drew += 1
                hit = [x for x, y in g if y >= CROP_AT]
                xmax = max(xmax, hit[0] if hit else g[-1][0])
                vis = [x for x, y in g if y >= YFLOOR]
                if vis:
                    xmin = min(xmin, vis[0])
                ax.plot([x for x, _ in g], [y for _, y in g],
                        color=col, ls=ls, marker=mk, markersize=5.6,
                        markevery=EVERY,
                        markerfacecolor=(col if fill is None else "none"),
                        markeredgecolor=("white" if fill is None else col),
                        markeredgewidth=(0.7 if fill is None else 1.2),
                        lw=1.7, zorder=3 if a == "syn" else 2)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0),
                                              numticks=12))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        if xmax > 0:
            ax.set_xlim(left=(xmin * 0.85 if xmin < float("inf") else None),
                        right=xmax * 1.3)
        ax.set_ylim(YFLOOR, 1.0 + (1.0 - YFLOOR) * 0.025)
        ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
        ax.set_title(clab, pad=7)
        if j == 0:
            ax.set_ylabel(f"Recall@10\n({pop} decile)", labelpad=6)
    if not drew:
        plt.close(fig)
        return False

    handles = ([Line2D([], [], color=col, lw=2.2, marker=mk, markersize=6.2,
                       markerfacecolor=col, markeredgecolor=col,
                       label=f"{elab} ({dim}d)")
                for _, elab, dim, col, mk in have]
               + [Line2D([], [], color="0.35", ls=ls, marker="o", markersize=6.0,
                         markerfacecolor=("0.35" if fill is None else "none"),
                         markeredgecolor="0.35", label=nm)
                  for _, nm, ls, fill in ARMS])
    fig.legend(handles=handles, loc="upper center",
               ncol=min(len(handles), 4), fontsize=10.5,
               bbox_to_anchor=(0.5, 1.015))
    xlab = "Search time per Query (ms)"
    if len(COLLS) == 1:
        # tight_layout does not measure supxlabel, so on a single panel it
        # leaves the strip below the axis empty and the label floating in it.
        axes[0][0].set_xlabel(xlab, labelpad=6)
        fig.tight_layout(rect=(0, 0, 1, 0.87))
    else:
        fig.supxlabel(xlab, y=0.02)
        fig.tight_layout(rect=(0, 0.06, 1, 0.87))
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{e}")
    plt.close(fig)
    print(f"wrote figures/{fname}.pdf / .png  "
          f"({len(have)} encoder(s), {drew} curves, 1x{len(COLLS)})")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", default="hard", choices=("hard", "all"))
    a = ap.parse_args()
    if not main(pop=a.pop):
        print("no encoder runs complete yet")

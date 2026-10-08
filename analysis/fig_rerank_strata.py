"""The reranked pipeline, read down the difficulty scale, in three metrics.

Rows are the three ways the answer is scored after reranking; columns are the
collections. Inside a panel, colour is the arm and line style is the population
-- every query, the worst decile, the worst 5%, the worst 1% -- the same grammar
as the retrieval figure, so the two can be read together: what SYN buys the
search, and what survives a cross-encoder.

The three rows are not three claims. MRR@10 and nDCG@10 are scored against the
fixed qrels and are therefore comparable across arms by construction; gold
recall@10 is the share of judged documents the pipeline returns at all, and is
the one that says whether a cheaper candidate list dropped an answer the
reranker never got to see. They move together where the claim holds and come
apart where it does not, which is worth a reader's attention and is why they are
separated rather than overlaid.

Each panel is scaled to its own data. The metrics live in different places --
FEVER's MRR runs 0.79-0.91 where NQ's runs 0.63-0.65 -- and a shared axis would
flatten every panel to show the one with the widest span. What is read here is
the HORIZONTAL gap between two curves of the same style, which no vertical
scaling changes.

The horizontal gap between two curves of the same style is the speedup at
matched end-to-end quality. It widens down the scale on the collections where
the tail is fragmented and closes on the ones where it is not, which is the
whole claim, stated once in the units a pipeline is actually billed in.

Nothing here is re-run: the metrics are slices of the all-query pass and the
search milliseconds come from strata_cell.py's own timing of each stratum.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import housestyle as H                                          # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.lines import Line2D                             # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator         # noqa: E402
from paths import ROOT                                          # noqa: E402

RES, OUT = ROOT / "results", ROOT / "figures"
ALL_COLLS = [("msmarco-1m", "MS MARCO 1M"), ("fever", "FEVER 1M"),
             ("nq", "NQ 1M"), ("quora", "Quora 523K")]
ARMS = [("vanilla", "Vanilla", H.RED), ("syn", "SYN", H.BLUE)]
# suffix -> (label, line style, marker, size); "" is the decile
# The 5th percentile is measured and ships -- rerank_pipeline_P95_*.json is in
# results/ and the numbers are quoted in the text -- but it is not drawn. Four
# strata and two arms is eight curves per panel, and the 5th percentile runs
# between the decile and the 1st percentile for its whole length, so it adds a
# crossing to every panel and separates nothing. Three strata say the same
# thing: the gap widens as the stratum narrows.
STRATA = [("_all", "all query", "-", "o", 5.0),
          ("", "hard decile", "--", "s", 5.2),
          ("_P99", "1st pct", "-.", "X", 6.0)]
STRATA_MEASURED = STRATA + [("_P95", "5th pct", ":", "^", 6.0)]
# rows, in the order the paper reads them
ALL_METRICS = [("mrr10_post", "MRR@10"),
               ("ndcg10_post", "nDCG@10"),
               ("gold10_post", "gold recall@10")]
# The main text takes one collection and two of the three metrics: MRR is scored
# against the fixed qrels and so compares the arms directly, and gold recall asks
# the separate question of whether a cheaper candidate list dropped an answer the
# reranker never got to see. nDCG moves with MRR everywhere and is in the
# appendix with the other three collections.
MAIN_COLLS = ("fever",)
MAIN_METRICS = ("mrr10_post", "gold10_post")


def main(fname="fig_rerank_strata", colls=MAIN_COLLS, metrics=MAIN_METRICS):
    COLLS = [c for c in ALL_COLLS if c[0] in colls]
    METRICS = [m for m in ALL_METRICS if m[0] in metrics]
    H.setup(13)
    # one collection reads better with the metrics side by side than stacked
    flip = len(COLLS) == 1 and len(METRICS) > 1
    nr, nc = (1, len(METRICS)) if flip else (len(METRICS), len(COLLS))
    fig, axes = plt.subplots(nr, nc, figsize=(4.7 * nc, 3.9 * nr),
                             squeeze=False)
    drew = 0
    for i, (mkey, mlab) in enumerate(METRICS):
      for j, (c, clab) in enumerate(COLLS):
        ax = axes[0][i] if flip else axes[i][j]
        lo, hi = float("inf"), float("-inf")
        for arm, _, col in ARMS:
            for suf, _, ls, mk, ms in STRATA:
                p = RES / f"rerank_pipeline{suf}_{c}.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                rows = [r for r in sorted(d["arms"][arm]["rows"],
                                          key=lambda r: r["R"])
                        if r.get("ms_total") and mkey in r]
                if not rows:
                    continue
                drew += 1
                lo = min(lo, min(r[mkey] for r in rows))
                hi = max(hi, max(r[mkey] for r in rows))
                ax.plot([r["ms_total"] for r in rows],
                        [r[mkey] for r in rows],
                        color=col, ls=ls, marker=mk, markersize=H.ms(ms), lw=H.LW,
                        markerfacecolor=col, markeredgecolor="white",
                        markeredgewidth=0.7,
                        zorder=3 if arm == "syn" else 2)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0),
                                              numticks=12))
        # a single collection spans well under a decade, so the decade ticks
        # alone leave the axis with one label on it
        ax.xaxis.set_minor_formatter(
            FuncFormatter((lambda v, _: f"{v:g}") if flip
                          else (lambda v, _: "")))
        if lo < hi:
            pad = (hi - lo) * 0.10
            ax.set_ylim(lo - pad, hi + pad)
        if flip:
            # the title carries the metric, so a y label would say it twice
            ax.set_title(f"{clab} · {mlab}", pad=7)
        else:
            if i == 0:
                ax.set_title(clab, pad=7)
            if j == 0:
                ax.set_ylabel(mlab, labelpad=6)
    if not drew:
        plt.close(fig)
        return False
    handles = ([Line2D([], [], color=col, lw=2.4, label=nm)
                for _, nm, col in ARMS]
               + [Line2D([], [], color="0.35", ls=ls, marker=mk, markersize=ms,
                         markerfacecolor="0.35", markeredgecolor="0.35",
                         label=nm) for _, nm, ls, mk, ms in STRATA])
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=11,
               bbox_to_anchor=(0.5, 1.0))
    fig.supxlabel("Search + rerank time per Query (ms)", y=0.012)
    fig.tight_layout(rect=(0, 0.025, 1, 0.945), w_pad=1.0)
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{e}")
    plt.close(fig)
    print(f"wrote figures/{fname}.pdf / .png  ({drew} curves)")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="main", choices=("main", "appendix"))
    a = ap.parse_args()
    if a.scope == "main":
        ok = main("fig_rerank_strata", MAIN_COLLS, MAIN_METRICS)
    else:
        ok = main("figA_rerank_all", [c for c, _ in ALL_COLLS],
                  [m for m, _ in ALL_METRICS])
    if not ok:
        print("no rerank strata files yet")

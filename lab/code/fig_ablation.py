"""Which part of the objective the numbers come from.

The paper reports this as a table, not a figure: three arms and two targets is
six numbers, and six numbers in a bar chart is a table drawn badly. The figure
is still written, because it is the fastest way to see whether a new arm landed
somewhere sensible, but `ablation_table.tex` is what the paper uses.

Each bar is one change from the shipped configuration, everything else held: the
same 500 mined MS MARCO queries, the same 500 training steps from the same seed,
the same encoder, and HNSW built and searched the same way. The quantity is the
search time each arm needs to reach recall 0.90 and 0.95 on its OWN hard decile
against its OWN exact top-10, expressed as a speedup over the untrained arm, so
a bar of 1 means the change threw the whole effect away.

The arms, in the order they answer things:

    no pull            the contrastive term alone. Everything the paper claims
                       rests on this bar being short.
    plain gradient     the same loss, the gradient unrestricted. Separates the
                       objective from the query-radial restriction on it.
Two arms, not five. The knobs with a continuous range -- lambda_pull, the two
temperatures, L, and how many queries are mined -- are a sweep, not an ablation,
and they are in the sweep figure, which reads them on an axis that cannot be
gamed by moving the target set. These two are the ones that switch a mechanism
off, so a bar is the right shape for them.

A uniform-positive arm was measured and is not drawn. It is in the lab as
abl-uniform, and it does not support the component it tests: drawing positives
flat instead of by softmax over the band gave both a larger speedup and better
qrels quality than the shipped configuration. It is out of the figure at the
authors' direction. If it stays out, score-aware positive sampling should be
described as a sampling choice rather than as a component the results rest on.

Read the bars against the full arm, not against each other: they are not nested
and two of them change different things.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import housestyle as H                                           # noqa: E402
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
from paths import ROOT                                           # noqa: E402

RES, OUT = ROOT / "results", ROOT / "figs" / "final"
C = "msmarco-1m"
BASE = "vanilla"
# Every arm here is built at the campaign's own setting, M=16 efC=200. The
# untrained and shipped arms carry the "__own" tag from the family campaign and
# the ablation arms "__own_M16"; both are the same build. run_family's -M
# defaults to 16 and the tag is only a label, so an earlier version of this
# figure divided M=16 arms by an M=8 reference. The loader below reads the
# degree out of each file and refuses to mix.
TAGS = ("__own", "__own_M16")
WANT_M = 16
# arm -> label, in the order drawn
ARMS = [("syn", "full SYN"),
        ("abl-nopull", "no pull\n($\\lambda_{pull}=0$)"),
        ("syn-plain", "plain gradient\n(no query-radial)"),
]
TARGETS = [(0.90, H.BLUE), (0.95, H.RED)]


def curve(arm, pop="hard"):
    p = next((RES / f"family_hnswlib_{C}__{arm}{t}.json" for t in TAGS
              if (RES / f"family_hnswlib_{C}__{arm}{t}.json").exists()), None)
    if p is None:
        return None
    d = json.loads(p.read_text())
    M = (d.get("params") or {}).get("M")
    if M != WANT_M:
        print(f"  !! {p.name} is M={M}, not {WANT_M}; refusing to mix degrees")
        return None
    if pop == "hard" and d.get("grid_hard"):
        return [(r["ms_per_query"], r["recall"]) for r in d["grid_hard"]
                if "recall" in r]
    return [(r["ms_per_query"], r[f"recall_{pop}"]) for r in d["grid"]]


def ms_at(g, r):
    ok = [x for x, y in g if y >= r]
    return min(ok) if ok else None


def main(fname="fig_ablation", pop="hard"):
    ref = curve(BASE, pop)
    if not ref:
        return False
    rows = [(a, lab, curve(a, pop)) for a, lab in ARMS]
    rows = [r for r in rows if r[2]]
    if not rows:
        return False

    H.setup(13)
    fig, ax = plt.subplots(figsize=(1.55 * len(rows) + 2.2, 4.6))
    x = np.arange(len(rows))
    w = 0.38
    for k, (R, col) in enumerate(TARGETS):
        base = ms_at(ref, R)
        vals, labs = [], []
        for _, _, g in rows:
            m = ms_at(g, R)
            vals.append(base / m if (base and m) else np.nan)
            labs.append("" if (base and m) else "n/r")
        ax.bar(x + (k - 0.5) * w, vals, w, color=col, alpha=0.85,
               edgecolor=col, linewidth=1.2, zorder=2,
               label=f"recall {R:g}")
        for xi, v, lb in zip(x + (k - 0.5) * w, vals, labs):
            if np.isnan(v):
                ax.text(xi, 1.05, lb, ha="center", fontsize=9, color="0.4")
            else:
                ax.text(xi, v + 0.12, f"{v:.1f}×", ha="center", fontsize=10.5)
    ax.axhline(1.0, color="0.35", lw=1.2, ls=(0, (3, 3)), zorder=1)
    ax.text(len(rows) - 0.45, 1.06, "untrained", fontsize=9.5, color="0.35",
            ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab, _ in rows], fontsize=11)
    ax.set_ylabel("speedup over the untrained arm\n(hard decile, matched recall)")
    # headroom for the value labels, which the legend was clipping
    top = max((v for v in ax.get_yticks()), default=1)
    ymax = max([b.get_height() for b in ax.patches] or [1])
    ax.set_ylim(0, ymax * 1.22)
    ax.legend(fontsize=11, loc="upper center", ncol=2, frameon=True,
              bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{e}")
    plt.close(fig)

    # the paper's own table body, so the six numbers are not retyped
    tex = ["% hard decile, HNSW M=16 efC=200, each arm against its own exact top-10",
           "% t@r is the per-query search time at which the arm first reaches recall r"]
    b90, b95 = ms_at(ref, 0.90), ms_at(ref, 0.95)
    tex.append(f"    untrained & {b90:.3f} & 1.00 & {b95:.3f} & 1.00 \\\\")
    for a, lab, g in rows:
        if a == BASE:
            continue
        m90, m95 = ms_at(g, 0.90), ms_at(g, 0.95)
        nm = lab.replace("\n", " ")
        tex.append(f"    {nm:22s} & {m90:.3f} & {b90/m90:.2f} & "
                   f"{m95:.3f} & {b95/m95:.2f} \\\\")
    (RES / "ablation_table.tex").write_text("\n".join(tex) + "\n")
    print(f"wrote figs/final/{fname}.pdf / .png and results/ablation_table.tex "
          f"({len(rows)} arms)")
    for a, lab, g in rows:
        line = f"  {a:14s}"
        for R, _ in TARGETS:
            b, m = ms_at(ref, R), ms_at(g, R)
            line += (f"  R{R:g} {b/m:6.2f}x ({m:7.3f} ms)" if (b and m)
                     else f"  R{R:g}      n/r")
        print(line)
    return True


if __name__ == "__main__":
    if not main():
        print("no ablation arms measured yet")

"""Redraw Figure 3 from VALUES.txt, without the corpora.

paper_fig3.py computes the correlations and then draws them, so it needs the
indexes, the ground truth, the phi dumps and the Steiner-hardness build -- none
of which fit in an artifact. Every number it prints is in VALUES.txt, so this
reads that file and draws the same figure. The two agree by construction: the
values are the ones paper_fig3.py wrote.

Use paper_fig3.py to recompute, this to redraw.

    python3 replot_fig3.py ../VALUES.txt ../fig3_hardness_measures
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
import sigmod_style                                              # noqa: E402
from matplotlib.patches import Patch                             # noqa: E402

sigmod_style.apply()
matplotlib.rcParams["hatch.linewidth"] = 0.45

TEXTWIDTH, K, EF, ACC = 7.0, 100, 128, 0.98
CORP = ["GloVe", "MS MARCO\nada-002", "YouTube-1M"]
MEAS = [("$\\varphi_1$", "#FFFFFF", "///"), ("Steiner ME", "#FFFFFF", "ooo"),
        ("LID", "#DCDCDC", "\\\\\\"), ("$d_{2k}/d_k$", "#DCDCDC", "xxx"),
        ("RC@$k$", "#9A9A9A", "")]
PAIRS = [("$\\varphi_1|$LID", "#FFFFFF", "..."), ("LID$|\\varphi_1$", "#BFBFBF", "..."),
         ("$\\varphi_1|$ME", "#FFFFFF", "++"), ("ME$|\\varphi_1$", "#BFBFBF", "++")]
INK, INK2 = "black", "#4D4D4D"


def parse(path):
    """The three '### (x)' blocks of VALUES.txt, as {row label: [three floats]}."""
    out, cur = {}, None
    for line in Path(path).read_text().splitlines():
        m = re.match(r"###\s*\((\w)\)", line)
        if m:
            cur = m.group(1)
            out[cur] = {}
            continue
        if cur is None or not line.startswith("  "):
            continue
        nums = re.findall(r"[-+]?\d*\.\d+", line)
        if len(nums) == 3:
            out[cur][line[:15].strip()] = [float(v) for v in nums]
    return out


def frame(ax, hi):
    ax.set_ylim(0, hi)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", ".5", "1"])
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(CORP, fontsize=7)
    ax.grid(axis="y", color="#DDDDDD", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
        ax.spines[s].set_linewidth(0.5)
    ax.tick_params(direction="out", length=2, width=0.5, colors=INK2,
                   labelsize=7, pad=1.5)
    for t in ax.get_xticklabels():
        t.set_color(INK)


def bars(ax, rows, specs, w, signed=False):
    x = np.arange(3)
    off = (len(specs) - 1) / 2.0
    for i, (lab, face, hat) in enumerate(specs):
        v = np.array(rows[i])
        ax.bar(x + (i - off) * w, np.abs(v), w * 0.88, facecolor=face, hatch=hat,
               edgecolor="black", linewidth=0.5, zorder=3)
        for xi, vi in zip(x + (i - off) * w, v):
            txt = (f"{vi:+.2f}".replace("0.", ".") if signed
                   else f"{vi:.2f}".lstrip("0"))
            ax.text(xi, abs(vi) + 0.015, txt, ha="center", va="bottom",
                    fontsize=5.4, color=INK, zorder=4)


def main(values, out):
    V = parse(values)
    order = [m[0] for m in MEAS]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.15), dpi=600)
    fig.patch.set_facecolor("white")

    for ax, key, ylab, title in (
            (axes[0], "a", f"$|\\rho|$ with recall@{K}", f"(a) fixed budget (ef={EF})"),
            (axes[1], "b", "$|\\rho|$ with least NDC", f"(b) effort to recall {ACC}")):
        bars(ax, [V[key][k] for k in order], MEAS, 0.165)
        frame(ax, 1.16)
        ax.set_ylabel(ylab, fontsize=7.5, labelpad=2)
        ax.set_title(title, fontsize=8, pad=5)
        ax.set_xlim(-0.55, 2.55)

    ax = axes[2]
    keys = list(V["c"].keys())
    bars(ax, [V["c"][k] for k in keys], PAIRS, 0.2, signed=True)
    frame(ax, 1.40)
    ax.set_xlim(-0.55, 2.45)
    ax.set_ylabel("partial $|\\rho|$", fontsize=7.5, labelpad=2)
    ax.set_title("(c) partial, fixed budget", fontsize=8, pad=5)
    ax.legend(handles=[Patch(facecolor=f, hatch=h, edgecolor="black",
                             linewidth=0.5, label=l) for l, f, h in PAIRS],
              loc="upper center", ncol=2, frameon=False, fontsize=6.0,
              handlelength=1.2, columnspacing=0.8, handletextpad=0.3,
              labelspacing=0.2, borderpad=0.1)

    fig.legend(handles=[Patch(facecolor=f, hatch=h, edgecolor="black",
                              linewidth=0.5, label=l) for l, f, h in MEAS],
               loc="lower center", ncol=5, frameon=False, fontsize=7,
               bbox_to_anchor=(0.34, -0.04), handlelength=1.3,
               columnspacing=1.1, handletextpad=0.3)
    fig.text(0.5, -0.055,
             "ada-002 = OpenAI text-embedding-ada-002 on ann-benchmarks MS MARCO v1; "
             "not the BEIR subset under BGE-M3 used elsewhere in this paper.",
             ha="center", fontsize=5.8, color="#4D4D4D")
    fig.tight_layout(rect=(0, 0.075, 1, 1.0))
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", facecolor="white", bbox_inches="tight",
                    pad_inches=0.01)
    print(f"wrote {out}.pdf / .png from {values}")


if __name__ == "__main__":
    a = sys.argv[1:] or ["../VALUES.txt", "../fig3_hardness_measures"]
    main(a[0], a[1] if len(a) > 1 else "../fig3_hardness_measures")

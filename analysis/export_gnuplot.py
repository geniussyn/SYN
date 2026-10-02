"""Figures 1-3 as plain data and gnuplot, for redrawing outside this tree.

Each figure gets its numbers as TSV and a .gp that reproduces the layout. The
TSV is the deliverable: it is what the paper's panel is drawn from, in the units
the axis uses, so a different plotting stack does not have to re-derive
anything. The gnuplot is a reference rendering, not the one in the paper -- the
paper's panels come from matplotlib and differ in typography.

  fig1  tail censoring. Two corpora, three query sets each, as (recall, QPS).
        The point of the panel is the vertical gap at fixed recall, so QPS is
        the y axis on a log scale; the second block is the derived gain curve,
        which is what the all-query set would have to run at to report the
        censored set's mean recall.
  fig2  mutual rank. Three 11x11 matrices, one per panel, written as a matrix
        TSV that gnuplot's `image` reads directly. Row and column 0 are the
        query. The diagonal is a document against itself and is written as NaN
        rather than 0, because 0 would plot as the strongest colour.
  fig3  hardness measures. Three blocks of grouped bars, |rho| of each measure
        against the quantity named in the block, over three corpora.

Run from the lab root; writes into export/gnuplot/.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT                                           # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "export" / "gnuplot"
TAILC = ROOT / "studies" / "tail-censoring" / "out"
PHIH = ROOT / "studies" / "hardness"

CORP = [("spacev", "SpaceV100M, 100d, L2"),
        ("youtube", "YouTube15M, 1024d, cosine")]
SUBSETS = [("all", "all queries"),
           ("drop1_ind", "hardest 1% removed"),
           ("drop10_ind", "hardest 10% removed")]


# --------------------------------------------------------------------- fig 1 --
def fig1():
    wrote = []
    for name, title in CORP:
        p = TAILC / f"{name}_curves.json"
        if not p.exists():
            print(f"  !! {p.name} missing")
            continue
        d = json.loads(p.read_text())
        by = {}
        for r in d["rows"]:
            by.setdefault(r["subset"], []).append(
                (r["ef"], r["recall"], r["qps"]))
        f = OUT / f"fig1_{name}.tsv"
        with f.open("w") as fh:
            fh.write(f"# {title}; n={d['n']}, dim={d['dim']}, "
                     f"nq={d['nq']}, M={d['M']}, efC={d['efC']}, k={d['k']}\n")
            fh.write("# subset\tef\trecall\tqps\n")
            for key, lab in SUBSETS:
                for ef, rec, qps in sorted(by.get(key, [])):
                    fh.write(f"{key}\t{ef}\t{rec:.6f}\t{qps:.3f}\n")
                fh.write("\n\n")           # gnuplot index separator
        wrote.append(f)

        # the derived panel: QPS ratio at matched mean recall
        g = OUT / f"fig1_{name}_gain.tsv"
        a = np.array(sorted(by["all"]))
        with g.open("w") as fh:
            fh.write(f"# {title}: QPS the all-query set reaches, over the QPS "
                     f"it would need to report the censored set's mean recall\n")
            fh.write("# subset\trecall_all\tgain\n")
            for key, lab in SUBSETS[1:]:
                b = np.array(sorted(by[key]))
                ok = (b[:, 1] > a[0, 1]) & (b[:, 1] <= a[-1, 1])
                need = np.exp(np.interp(b[ok, 1], a[:, 1], np.log(a[:, 2])))
                for r, q, nd in zip(a[ok, 1], a[ok, 2], need):
                    fh.write(f"{key}\t{r:.6f}\t{q / nd:.6f}\n")
                fh.write("\n\n")
        wrote.append(g)

    (OUT / "fig1_tail_censor.gp").write_text(FIG1_GP)
    return wrote


# --------------------------------------------------------------------- fig 2 --
def fig2():
    p = RES / "mutual_rank_panels_all.json"
    if not p.exists():
        print("  !! mutual_rank_panels_all.json missing")
        return []
    d = json.loads(p.read_text())
    wrote, meta = [], []
    for i, panel in enumerate(d["panels"], 1):
        m = np.array(panel["matrix"], dtype=float)
        f = OUT / f"fig2_panel{i}.tsv"
        with f.open("w") as fh:
            fh.write(f"# {panel['query']!r}  qid={panel['qid']}  "
                     f"arm={panel['arm']}  {panel.get('kind','')}\n")
            fh.write(f"# {d['quantity']}\n")
            fh.write("# row 0 and column 0 are the query; "
                     "nan on the diagonal is a document against itself\n")
            for row in m:
                fh.write("\t".join("nan" if not np.isfinite(v) else f"{v:g}"
                                   for v in row) + "\n")
        wrote.append(f)
        meta.append(f"panel{i}: {panel['arm']:<8} {panel['query']}")
    cs = d["colour_scale"]
    (OUT / "fig2_mutual_rank.gp").write_text(
        FIG2_GP.format(vmin=cs["vmin"], vmax=cs["vmax"],
                       rule=cs["L_rule_at"], n=len(d["panels"])))
    (OUT / "fig2_panels.txt").write_text("\n".join(meta) + "\n")
    return wrote


# --------------------------------------------------------------------- fig 3 --
BLOCKS = {"a": "|rho| with recall@100 at ef=128",
          "b": "|rho| with least NDC to 0.98",
          "c": "partial |rho|, fixed budget"}


def fig3():
    p = PHIH / "VALUES.txt"
    if not p.exists():
        print("  !! phih/VALUES.txt missing")
        return []
    blocks, cur = {}, None
    for line in p.read_text().splitlines():
        m = re.match(r"###\s*\((\w)\)", line)
        if m:
            cur = m.group(1)
            blocks[cur] = {}
            continue
        if cur is None or not line.startswith("  "):
            continue
        nums = re.findall(r"[-+]?\d*\.\d+", line)
        if len(nums) == 3:
            blocks[cur][line[:15].strip()] = [float(v) for v in nums]
    corpora = ["glove", "msmarco", "youtube1m"]
    f = OUT / "fig3_hardness_measures.tsv"
    with f.open("w") as fh:
        fh.write("# absolute Spearman correlation of each per-query hardness "
                 "measure with the quantity named by the block\n")
        fh.write(f"# corpora, in column order: {', '.join(corpora)}\n")
        fh.write("# block\tmeasure\t" + "\t".join(corpora) + "\n")
        for b in ("a", "b", "c"):
            if b not in blocks:
                continue
            fh.write(f"\n# ({b}) {BLOCKS[b]}\n")
            for measure, vals in blocks[b].items():
                clean = measure.replace("$", "").replace("\\varphi", "phi")
                # short, plottable tick labels: rotated names get clipped by a
                # multiplot's explicit margins, and these read fine horizontally
                clean = {"phi_1": "phi1", "Steiner ME": "Steiner",
                         "d_{2k}/d_k": "d2k/dk", "RC@$k$": "RC@k",
                         "RC@k": "RC@k"}.get(clean, clean)
                fh.write(f"{b}\t{clean}\t" +
                         "\t".join(f"{abs(v):.3f}" for v in vals) + "\n")
            fh.write("\n\n")
    (OUT / "fig3_hardness_measures.gp").write_text(FIG3_GP)
    return [f]


FIG1_GP = """# Figure 1 -- tail censoring. gnuplot >= 5.2
# Removing the hardest queries from the evaluation set moves the reported
# curve up without changing the index. The gap at fixed recall is the claim.
# Each file holds three index blocks, in the order of SUB below.
set terminal pdfcairo size 7.0in,2.8in font "Helvetica,10" noenhanced
set output "fig1_tail_censor.pdf"
array SUB[3] = ["all queries", "hardest 1% removed", "hardest 10% removed"]
array CORP[2] = ["spacev", "youtube"]
array NAME[2] = ["SpaceV100M  100d, L2", "YouTube15M  1024d, cosine"]
array PT[3] = [7, 5, 9]
array LC[3] = ["#000000", "#6b6b6b", "#9e9e9e"]
set multiplot layout 1,2 margins 0.09,0.98,0.17,0.90 spacing 0.09
set logscale y
set key left bottom spacing 1.15 samplen 2 width -2
set grid lw 0.35 lc rgb "#dbdbdb"
set xlabel "mean recall@10" offset 0,0.4
set ylabel "queries per second" offset 1.5,0
do for [c=1:2] {
  set title NAME[c] font "Helvetica,10"
  plot for [i=1:3] sprintf("fig1_%s.tsv", CORP[c]) index i-1 using 3:4 \\
       with linespoints lw 1.4 pt PT[i] ps 0.5 lc rgb LC[i] title SUB[i]
}
unset multiplot
"""

FIG2_GP = """# Figure 2 -- mutual rank. gnuplot >= 5.2
# Each panel is an 11x11 matrix: the query and its exact top-10, coloured by
# min(rank_i(j), rank_j(i)) over the whole corpus. Low is reachable.
# The diagonal is NaN -- a document against itself -- and is left blank.
set terminal pdfcairo size 7.6in,2.7in font "Helvetica,10" noenhanced
set output "fig2_mutual_rank.pdf"
array TITLE[{n}] = ["Vanilla, easy query", "Vanilla, fragmented", "SYN, the same query"]
set logscale cb
set cbrange [{vmin}:{vmax}]
# viridis, reversed: dark is a distant pair
set palette defined (0 "#fde725", 0.5 "#21918c", 1 "#440154")
set cbtics 1,10,10000
set size ratio 1
unset key
# row and column 0 are the query; the ten documents follow
set xrange [-0.5:10.5]
set yrange [-0.5:10.5]
set tics scale 0.4 out
set xtics 0,2,10
set ytics 0,2,10
set multiplot layout 1,{n} margins 0.06,0.90,0.12,0.88 spacing 0.07
do for [i=1:{n}] {{
  set title TITLE[i] font "Helvetica,10"
  # one colour bar is enough; naming it on every panel runs the label into
  # the next panel's tics
  if (i == {n}) {{ set cblabel "mutual rank" offset 1.2,0 }} else {{ unset cblabel }}
  plot sprintf("fig2_panel%d.tsv", i) matrix with image
}}
unset multiplot
"""

FIG3_GP = """# Figure 3 -- hardness measures. gnuplot >= 5.2
# |Spearman rho| of each per-query hardness measure against the quantity the
# block names, on three corpora. phi_1 is the paper's measure.
# Columns: 1 block, 2 measure, 3-5 glove / msmarco / youtube1m.
# noenhanced throughout: the measure names carry underscores that would
# otherwise be read as subscripts.
set terminal pdfcairo size 7.2in,2.7in font "Helvetica,10" noenhanced
set output "fig3_hardness_measures.pdf"
array CORP[3] = ["GloVe", "MS MARCO", "YouTube1M"]
array TITLE[3] = ["(a) vs recall@100", "(b) vs least NDC", "(c) partial"]
array FACE[3] = ["#4878a8", "#c04a4a", "#5aa05a"]
set style data histograms
set style histogram clustered gap 1
set style fill solid 0.85 border lc rgb "#ffffff"
set boxwidth 0.9
set yrange [0:1.05]
set ytics 0,0.5,1
set grid y lw 0.35 lc rgb "#dbdbdb"
set key at screen 0.5,0.07 center horizontal maxrows 1 samplen 1.2 noenhanced
set xtics font "Helvetica,8" noenhanced
set ylabel "|rho|" offset 1.5,0
set multiplot layout 1,3 margins 0.07,0.98,0.26,0.90 spacing 0.07
do for [b=1:3] {
  set title TITLE[b] font "Helvetica,10"
  plot for [col=3:5] "fig3_hardness_measures.tsv" index b-1 \\
       using col:xtic(2) title CORP[col-2] lc rgb FACE[col-2]
}
unset multiplot
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for name, fn in (("fig1", fig1), ("fig2", fig2), ("fig3", fig3)):
        print(f"{name}:")
        got = fn()
        for f in got:
            print(f"  {f.relative_to(ROOT)}  ({f.stat().st_size:,} B)")
        files += got
    gps = sorted(OUT.glob("*.gp"))
    print(f"\n{len(files)} data file(s), {len(gps)} gnuplot script(s) "
          f"in {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()

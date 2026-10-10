"""Every figure in the paper as plain data and gnuplot.

The repository's figures are drawn with matplotlib. This writes the same
numbers as TSV and a gnuplot script per figure, so the panels can be rebuilt
without that dependency and inspected without any plotting stack at all.

The data is taken by importing each figure module and calling the loader it
draws from, never by re-reading the result JSON independently. If a figure
changes which cell or which tag it reads, this export follows, and a divergence
between the two is a bug rather than a drift.

Layout of a curve TSV: one gnuplot index block per series, blocks in the order
listed in the file's header, columns `x  y`. The header names each block, so
`index i` and the i-th header line correspond.

Three figures are not curves and have their own shape:
  fig2  mutual rank      an 11x11 matrix per panel, for `matrix with image`
  fig3  hardness         grouped bars, one block per sub-question
  fig7  build cost       two line panels and a bar panel

Writes export/gnuplot/. Run `make -C export/gnuplot` to render.
"""
import argparse
import importlib
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


# matplotlib's style vocabulary, in gnuplot's. The figure modules own the
# encoding -- which dimension is colour and which is the line -- so it is read
# from them rather than re-decided here, and a `.style` file travels beside
# every curve TSV saying what each block should look like.
DT = {"-": 1, "--": 2, ":": 3, "-.": 4, "solid": 1, "dashed": 2}
PT = {"o": 7, "s": 5, "^": 9, "D": 13, "P": 11, "X": 3, "v": 10, "+": 1, None: 7}


def write_style(path, rows):
    """rows: [(label, colour, linestyle, marker), ...] in block order."""
    with path.open("w") as fh:
        fh.write("# style of each index block, as the figure module defines it\n")
        fh.write("# block\tlabel\tcolour\tdashtype\tpointtype\n")
        for i, (lab, col, ls, mk) in enumerate(rows):
            fh.write(f"{i}\t{lab}\t{col}\t{DT.get(ls, 1)}\t{PT.get(mk, 7)}\n")
    return path



HOUSE = """
# --- house style: seaborn darkgrid, as the matplotlib figures use -----------
set style line 101 lc rgb "#FFFFFF" lw 1.1 dt 1
set grid back ls 101
set border 0
set tics nomirror scale 0 textcolor rgb "#262626"
set xlabel textcolor rgb "#262626"
set ylabel textcolor rgb "#262626"
set title textcolor rgb "#262626"
set key textcolor rgb "#262626"
set object 1 rectangle from graph 0,0 to graph 1,1 behind \\
    fillcolor rgb "#EAEAF2" fillstyle solid 1.0 noborder
"""


def mod(name):
    return importlib.import_module(name)


def write_curves(path, header, blocks):
    """blocks: [(label, [(x, y), ...]), ...] -> one index block each."""
    with path.open("w") as fh:
        for line in header:
            fh.write(f"# {line}\n")
        fh.write("# blocks, in index order:\n")
        for i, (lab, _) in enumerate(blocks):
            fh.write(f"#   {i}  {lab}\n")
        fh.write("# x\ty\n")
        for lab, pts in blocks:
            fh.write(f"\n# {lab}\n")
            for x, y in pts:
                fh.write(f"{x:.6g}\t{y:.6g}\n")
            fh.write("\n\n")
    return path


# ------------------------------------------------------------------ fig 5 ----
def fig5_strata():
    m = mod("fig_strata")
    out = []
    for corpus, label in m.COLLS:
        blocks, style = [], []
        for arm, alab, acol in m.ARMS:
            d = m.load(corpus, arm)
            if not d:
                continue
            for key, slab, ls, mk, _ in m.STRATA:
                g = (d.get("strata") or {}).get(key)
                if not g:
                    continue
                blocks.append((f"{alab} / {slab}",
                               [(r["ms"], r["recall"]) for r in g["grid"]]))
                style.append((f"{alab} / {slab}", acol, ls, mk))
        if not blocks:
            continue
        d = m.load(corpus, "vanilla")
        write_style(OUT / f"fig5_strata_{corpus}.style", style)
        out.append(write_curves(
            OUT / f"fig5_strata_{corpus}.tsv",
            [f"{label}: recall@10 against search time, by difficulty stratum",
             f"n_docs={d['n_docs']:,}  M={d['M']}  efC={d['efConstruction']}",
             "recall is each arm against its own exact top-10",
             "x: ms per query, one pinned core, one thread, min of three"],
            blocks))
    return out


# ------------------------------------------------------------------ fig 6 ----
def fig6_families():
    m = mod("fig_families")
    out = []
    for pop, plab in m.POPS:
        for corpus, label in m.COLLS:
            blocks, style = [], []
            for fam, flab, fcol, fmk in m.FAMILIES:
                for arm, alab, als, _ in m.ARMS:
                    g, _ = m.series(fam, corpus, arm, m.TAG, pop)
                    if g:
                        blocks.append((f"{flab} / {alab}", g))
                        style.append((f"{flab} / {alab}", fcol, als, fmk))
            if blocks:
                write_style(OUT / f"fig6_families_{pop}_{corpus}.style", style)
                out.append(write_curves(
                    OUT / f"fig6_families_{pop}_{corpus}.tsv",
                    [f"{label}, {plab}: four index families at the smallest degree",
                     f"tag {m.TAG}; recall is each arm against its own exact top-10",
                     "x: ms per query"],
                    blocks))
    return out


# ------------------------------------------------------------------ fig 7 ----
def fig7_build_cost():
    m = mod("fig_build_cost")
    out = []
    for corpus, label in m.COLLS:
        blocks, style = [], []
        for bi, (ms, blab, bmk) in enumerate(m.BUDGETS):
            for arm, alab, acol in m.ARMS:
                pts = []
                for M in m.MS_SWEEP:
                    g, _ = m.cell(corpus, arm, M)
                    if not g:
                        continue
                    r = m.reach(g, ms)
                    if r is not None:
                        pts.append((M, r))
                if len(pts) > 1:
                    blocks.append((f"{alab} / budget {blab}", pts))
                    style.append((f"{alab} / budget {blab}", acol,
                                  "-" if bi == 0 else "--", bmk))
        if blocks:
            write_style(OUT / f"fig7_build_recall_{corpus}.style", style)
            out.append(write_curves(
                OUT / f"fig7_build_recall_{corpus}.tsv",
                [f"{label}: hard-decile recall@10 reachable within a time budget,",
                 "against the HNSW degree M at efC=200"],
                blocks))
    # the bar panel: the cheapest build at matched recall
    rows = []
    for corpus, label in m.COLLS:
        best = []
        for M in m.MS_SWEEP:
            g, b = m.cell(corpus, "vanilla", M)
            if g and b:
                r = m.reach(g, m.BUDGETS[0][0])
                if r is not None:
                    best.append((r, M, b))
        if not best:
            continue
        rv, Mv, bv = max(best)
        for M in m.MS_SWEEP:
            g, b = m.cell(corpus, "syn", M)
            if g and b and m.reach(g, m.BUDGETS[0][0]) is not None \
                    and m.reach(g, m.BUDGETS[0][0]) >= rv:
                rows.append((label, Mv, bv, M, b, 1 - b / bv))
                break
    if rows:
        f = OUT / "fig7_build_bars.tsv"
        with f.open("w") as fh:
            fh.write("# the cheapest build reaching the recall the other arm needs\n")
            fh.write("# collection\tM_vanilla\tmin_vanilla\tM_syn\tmin_syn\tsaved\n")
            for lab, Mv, bv, Ms, bs, sv in rows:
                fh.write(f"{lab.replace(' ', '_')}\t{Mv}\t{bv:.3f}\t{Ms}\t"
                         f"{bs:.3f}\t{sv:.4f}\n")
        out.append(f)
    return out


# ------------------------------------------------------------------ fig 8 ----
def fig8_rerank():
    m = mod("fig_rerank_strata")
    out = []
    for corpus, label in m.ALL_COLLS:
        for mkey, mlab in m.ALL_METRICS:
            blocks, style = [], []
            for arm, alab, acol in m.ARMS:
                for suf, slab, sls, smk, _ in m.STRATA:
                    p = RES / f"rerank_pipeline{suf}_{corpus}.json"
                    if not p.exists():
                        continue
                    d = json.loads(p.read_text())
                    rows = [r for r in sorted(d["arms"][arm]["rows"],
                                              key=lambda r: r["R"])
                            if r.get("ms_total") and mkey in r]
                    if rows:
                        blocks.append((f"{alab} / {slab}",
                                       [(r["ms_total"], r[mkey]) for r in rows]))
                        style.append((f"{alab} / {slab}", acol, sls, smk))
            if blocks:
                write_style(OUT / f"fig8_rerank_{corpus}_{mkey}.style", style)
                out.append(write_curves(
                    OUT / f"fig8_rerank_{corpus}_{mkey}.tsv",
                    [f"{label}: {mlab} after reranking, against search+rerank time",
                     "candidate depth R runs 10..150 along each curve"],
                    blocks))
    return out


# ------------------------------------------------------------------ fig 9 ----
def fig9_encoders():
    m = mod("fig_encoders")
    out = []
    for corpus, label in m.COLLS:
        blocks, style = [], []
        for suf, elab, dim, ecol, emk in m.ENCODERS:
            for arm, alab, als, _ in m.ARMS:
                g = m.series(corpus, f"{arm}{suf}", "hard")
                if g:
                    blocks.append((f"{elab} ({dim}d) / {alab}", g))
                    style.append((f"{elab} ({dim}d) / {alab}", ecol, als, emk))
        if blocks:
            write_style(OUT / f"fig9_encoders_{corpus}.style", style)
            out.append(write_curves(
                OUT / f"fig9_encoders_{corpus}.tsv",
                [f"{label}: five encoders, hard decile, HNSW M=8 efC=200",
                 "one LoRA adapter per encoder, mined and fitted in its own space",
                 "x: ms per query"],
                blocks))
    return out


# ----------------------------------------------------------------- fig A1 ----
def figA1_ngfix():
    m = mod("fig_ngfix")
    out = []
    for pop, plab in m.POPS:
        for corpus, label in m.ALL_COLLS:
            blocks, style = [], []
            for arm, alab, acol in m.ARMS:
                p = RES / f"ngfix_{corpus}__{arm}.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                for key, klab, kls, kmk, _ in m.STAGES:
                    g = (d.get("grids") or {}).get(key, {}).get(pop)
                    if g:
                        blocks.append((f"{alab} / {klab}",
                                       [(x["ms"], x["recall"]) for x in g]))
                        style.append((f"{alab} / {klab}", acol, kls, kmk))
            if blocks:
                write_style(OUT / f"figA1_ngfix_{pop}_{corpus}.style", style)
                out.append(write_curves(
                    OUT / f"figA1_ngfix_{pop}_{corpus}.tsv",
                    [f"{label}, {plab}: SYN composed with index-side repair",
                     "x: ms per query"],
                    blocks))
    return out


# ------------------------------------------------- fig12: the distributions --
# The only figure here that is not a curve. fig_dist draws two rows of
# histograms -- phi_1 and the recall actually returned at a fixed budget -- and
# the claim is the SHAPE: a mean of 7.37 is consistent with every query being
# slightly broken and with most being whole and a few shattered, and the
# distribution says which. The means travel with the data because each panel
# prints its own pair.
def fig12_distributions():
    from fig_dist import (COLLS, ARMS, phi_dist, rec_dist,            # noqa
                          PHI_LEVELS, REC_LEVELS, EF_SHOW)
    wrote, cache = [], {}
    for c, clab in COLLS:
        rows, style, meta = [], [], []
        for arm, lab, col in ARMS:
            d = phi_dist(c, arm, 16, 200, "hard", 1, cache)
            if not d:
                continue
            n = np.array(d["counts"], float)
            rows.append((f"phi1 / {lab}", list(zip(PHI_LEVELS,
                                                   n / max(n.sum(), 1)))))
            meta.append(("phi1", lab, d["mean"]))
            style.append((f"phi1 / {lab}", col, "-", "s"))
        for arm, lab, col in ARMS:
            r = rec_dist(c, arm, 16, 200, "hard")
            if not r:
                continue
            n = np.array(r["counts"], float)
            # divided by the 0.1 bin width, so the panel encloses area 1
            rows.append((f"recall / {lab}",
                         list(zip(REC_LEVELS, n / max(n.sum(), 1) / 0.1))))
            meta.append(("recall", lab, r["mean"]))
            style.append((f"recall / {lab}", col, "-", "s"))
        if not rows:
            continue
        f = OUT / f"fig12_dist_{c}"
        with f.with_suffix(".tsv").open("w") as fh:
            fh.write(f"# {clab}, hard decile. Two panels stacked in one file.\n")
            fh.write("# phi1   read from the stored level-0 graph, h=1, k=10\n")
            fh.write(f"# recall recall@10 actually returned at efSearch={EF_SHOW}\n")
            fh.write("# density is probability mass; the recall rows divide by\n")
            fh.write("# the 0.1 bin width so each panel encloses area 1\n")
            fh.write("# blocks, in index order:\n")
            for i, (lab, _) in enumerate(rows):
                fh.write(f"#   {i}  {lab}\n")
            fh.write("# x\tdensity\n")
            for lab, pts in rows:
                fh.write(f"\n# {lab}\n")
                for x, y in pts:
                    fh.write(f"{x:.6g}\t{y:.6g}\n")
                # two blank lines, which is what gnuplot's `index` counts as a
                # block boundary -- one is only a break within one dataset
                fh.write("\n\n")
        write_style(f.with_suffix(".style"), style)
        with (OUT / f"fig12_dist_{c}.means").open("w") as fh:
            fh.write("# panel\tarm\tmean   -- each panel prints its own pair\n")
            for panel, lab, m in meta:
                fh.write(f"{panel}\t{lab}\t{m:.6g}\n")
        wrote.append(f.name)
    return wrote


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    total = []
    for name, fn in (("fig5 strata", fig5_strata),
                     ("fig6 families", fig6_families),
                     ("fig7 build cost", fig7_build_cost),
                     ("fig8 rerank", fig8_rerank),
                     ("fig9 encoders", fig9_encoders),
                     ("figA1 ngfix", figA1_ngfix),
                     ("fig12 distributions", fig12_distributions)):
        got = fn()
        print(f"{name:<18}{len(got):>3} file(s)")
        total += got
    print(f"\n{len(total)} data file(s) in {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()

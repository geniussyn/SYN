"""Appendix table: the four continuous knobs, swept.

Read index-free with screen_arm.py rather than with a timed search, for a
reason a recall column cannot escape: recall@10 is measured against each arm's
own exact top-10, so an arm that moves its target set to an easier one improves
that number without repairing anything -- which is what the L arms did. There
is no index behind this table and no search time in it.

phi-hat_1 is the number of components the target set falls into under the best
graph a builder with out-degree maxM0 could make. It needs no index, carries no
build noise, and bounds the real phi_1 from below. Beside it, the exhaustive
MRR@10 against the fixed qrels, which is what the geometry cost.

The table is read down the two numeric columns together. Across the fourteen
trained arms they correlate at +0.74: every knob that drives fragmentation down
also costs ranking, and the shipped setting is a point on that frontier rather
than the minimum of either axis.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT                                           # noqa: E402

RES = ROOT / "results"
C = "msmarco-1m"

# (block title, [(arm, setting as printed)]) -- the shipped value is in every
# block at its own setting, so the four share one anchor
BLOCKS = [
    ("$\\lambda_{\\mathrm{pull}}$",
     [("abl-nopull", "$0$"), ("abl-lam0.3", "$0.3$"), ("syn", "$1.0$"),
      ("abl-lam2.0", "$2.0$"), ("abl-lam3.0", "$3.0$")]),
    ("$\\tau_p$",
     [("abl-tp0.005", "$0.005$"), ("syn", "$0.02$"),
      ("abl-tp0.05", "$0.05$"), ("abl-tp0.1", "$0.1$")]),
    # L_lo is printed beside L because the column is not a clean sweep without
    # it. syn/pull.py sets the band floor to 0.75*L when --pull-low is absent,
    # which is what every arm here used except the shipped one: it passes
    # --pull-low 1000 against L = 1,100, a ratio of 0.909. L and its band move
    # together down the column and the anchor moves differently again, so a
    # difference cannot be attributed to L alone. The L1100b75 arm is the
    # shipped L under the rule's band, and the pair separates the two.
    ("$L$ \\ (band floor $L_{lo}$)",
     [("abl-L32", "$32$ \\,/\\,$24$"),
      ("abl-L130", "$130$ \\,/\\,$98$"),
      ("abl-L300", "$300$ \\,/\\,$225$"),
      ("abl-L600", "$600$ \\,/\\,$450$"),
      ("abl-L1100b75", "$1{,}100$ \\,/\\,$825$"),
      ("syn", "$1{,}100$ \\,/\\,$1{,}000$")]),
    ("training queries",
     [("abl-n50", "$50$"), ("abl-n100", "$100$"), ("abl-n250", "$250$"),
      ("syn", "$500$")]),
]
SHIPPED = "syn"


def read(arm):
    p = RES / f"screen_{C}__{arm}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return (d["phi1_hat"], d["pr_phi1_hat_eq_1"], d["exhaustive_mrr10"],
            d["exhaustive_gold10"])


def main(out="hparam_sweep_table.tex"):
    base = read("vanilla")
    if not base:
        print("the untrained reference has not been screened")
        return False

    lines = [f"% {C}, 1,000 queries, maxM0 = 32, k = 10",
             "% phi-hat_1 is index-free; MRR and gold are exhaustive, vs fixed qrels",
             "% the shipped setting is marked; it appears once per block",
             f"    untrained & & {base[0]:.3f} & {base[1]:.3f} & "
             f"{base[2]:.4f} & {base[3]:.4f} \\\\",
             "    \\midrule"]
    seen, xs, ys = set(), [], []
    for title, pts in BLOCKS:
        lines.append(f"    \\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for arm, label in pts:
            v = read(arm)
            if not v:
                print(f"  !! {arm} not screened")
                continue
            mark = "$^{\\dagger}$" if arm == SHIPPED else ""
            lines.append(f"    \\quad {label}{mark} & & {v[0]:.3f} & {v[1]:.3f} "
                         f"& {v[2]:.4f} & {v[3]:.4f} \\\\")
            if arm not in seen:
                seen.add(arm)
                xs.append(v[0])
                ys.append(v[2])
        lines.append("    \\addlinespace[2pt]")
    (RES / out).write_text("\n".join(lines) + "\n")

    r = float(np.corrcoef(xs, ys)[0, 1])
    print(f"wrote results/{out}")
    print(f"  {len(seen)} distinct trained arms")
    print(f"  phi-hat_1 range {min(xs):.3f}-{max(xs):.3f}, "
          f"untrained {base[0]:.3f}")
    print(f"  corr(phi-hat_1, MRR@10) = {r:+.3f}")
    print()
    print(f"  {'setting':<22}{'phi1^':>8}{'Pr[=1]':>8}{'MRR@10':>9}{'gold@10':>9}")
    print(f"  {'untrained':<22}{base[0]:>8.3f}{base[1]:>8.3f}"
          f"{base[2]:>9.4f}{base[3]:>9.4f}")
    for title, pts in BLOCKS:
        print(f"  {title.replace(chr(92),'').replace('$',''):<22}")
        for arm, label in pts:
            v = read(arm)
            if v:
                lab = label.replace("$", "").replace("{,}", ",")
                star = " *" if arm == SHIPPED else ""
                print(f"    {lab+star:<20}{v[0]:>8.3f}{v[1]:>8.3f}"
                      f"{v[2]:>9.4f}{v[3]:>9.4f}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hparam_sweep_table.tex")
    main(ap.parse_args().out)

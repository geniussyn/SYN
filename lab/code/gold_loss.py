"""How often ANN serving loses a document the retriever ranked first, and why.

The per-query examples in Table 1 are anecdotes until this is stated over the
population. Three questions, in order:

  1. Among queries whose judged document is FIRST in the exhaustive ranking --
     the cases where the retriever is unambiguously right -- how often does the
     ANN search return nothing relevant at all?
  2. Is that loss graceful? A search that merely demotes the answer is a
     different failure from one that drops it.
  3. Does it track fragmentation? phi_1 is a property of the query and the
     graph, computed with no search, so if the loss rate rises with it then the
     mechanism and the symptom are the same thing.

Nothing here is re-measured. The per-query recall and MRR come from each cell's
own arrays and phi_1 from the level-0 graph that cell dumped.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kappa import phi                                            # noqa: E402
from paths import ROOT                                           # noqa: E402

RES = ROOT / "results"


def load(corpus, arm, M, efc, ef):
    qs = [json.loads(l) for l in
          (ROOT / "corpora" / f"{corpus}_queries.jsonl")
          .read_text(encoding="utf-8").splitlines() if l.strip()]
    gt = np.load(RES / f"gt_{corpus}__{arm}_k10.npz")["idx"]
    z = np.load(RES / f"perquery_{corpus}__{arm}__M{M}__own_efc{efc}.npz")
    j = [int(e) for e in z["ef"]].index(ef)
    rec, mrr = z["recall"][j], z["mrr"][j]

    # the exhaustive ranking's own MRR: index-free, so it isolates the retriever
    ex = np.zeros(len(qs))
    for i, q in enumerate(qs):
        rel = {int(k) for k, v in q["rel"].items() if int(v) > 0}
        for r, d in enumerate(gt[i][:10]):
            if int(d) in rel:
                ex[i] = 1.0 / (r + 1)
                break

    g = RES / f"graph_{corpus}__{arm}__M{M}__own_efc{efc}.npz"
    f1 = None
    if g.exists():
        zg = np.load(g)
        f1 = phi(zg["indptr"], zg["indices"], gt, h=1, log_every=0)
    return rec, mrr, ex, f1


def report(corpus, arms, M, efc, ef):
    print(f"\n{corpus}  M={M} efC={efc} ef={ef}")
    out = {}
    for arm in arms:
        rec, mrr, ex, f1 = load(corpus, arm, M, efc, ef)
        first = ex == 1.0                       # gold is exhaustive rank 1
        lost = first & (mrr == 0.0)
        demoted = first & (mrr > 0.0) & (mrr < 1.0)
        n, nl = int(first.sum()), int(lost.sum())
        print(f"\n  [{arm}]  gold at exhaustive rank 1: {n} queries")
        print(f"    ANN returns nothing relevant : {nl:5d}  ({nl/n:6.2%})")
        print(f"    ANN demotes it but keeps it  : {int(demoted.sum()):5d}  "
              f"({demoted.sum()/n:6.2%})")
        # the loss is not explained by recall alone
        for lo, hi in ((0.9, 1.01), (0.7, 0.9), (0.0, 0.7)):
            m = first & (rec >= lo) & (rec < hi)
            if m.sum():
                print(f"    among recall in [{lo:.1f},{hi:.1f}): "
                      f"{int((m & lost).sum()):4d} / {int(m.sum()):4d} lost "
                      f"({(m & lost).sum()/m.sum():6.2%})")
        if f1 is not None:
            print(f"    by fragmentation of the exact top-10:")
            for lo, hi, lab in ((1, 2, "phi_1 = 1"), (2, 5, "phi_1 2-4"),
                                (5, 11, "phi_1 >= 5")):
                m = first & (f1 >= lo) & (f1 < hi)
                if m.sum():
                    print(f"      {lab:12s} {int((m & lost).sum()):5d} / "
                          f"{int(m.sum()):5d} lost ({(m & lost).sum()/m.sum():6.2%})")
            a = first & (f1 >= 5)
            b = first & (f1 == 1)
            if a.sum() and b.sum() and (b & lost).sum():
                rr = ((a & lost).sum()/a.sum()) / ((b & lost).sum()/b.sum())
                print(f"      risk ratio phi>=5 vs phi=1: {rr:.1f}x")
        out[arm] = {"n_gold_first": n, "n_lost": nl, "rate": nl/n,
                    "n_demoted": int(demoted.sum())}
    if len(out) == 2:
        a, b = arms
        print(f"\n  loss rate {out[a]['rate']:.2%} -> {out[b]['rate']:.2%}  "
              f"({out[a]['rate']/max(out[b]['rate'],1e-9):.1f}x fewer)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="msmarco-1m")
    ap.add_argument("--arms", default="vanilla,syn")
    ap.add_argument("-M", type=int, default=8)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--ef", type=int, default=64)
    a = ap.parse_args()
    r = report(a.corpus, a.arms.split(","), a.M, a.efc, a.ef)
    p = RES / f"gold_loss_{a.corpus}_M{a.M}_ef{a.ef}.json"
    p.write_text(json.dumps(r, indent=2))
    print(f"\nwrote {p.name}")

"""Why one query's answers hang together and another's do not, and what SYN does.

The quantity is the mutual rank of a pair: how far down each document sits in the
other's ordering of the whole corpus, taken as the smaller of the two. It is the
same number the pull loss thresholds on, and it needs no index, so it separates
"the encoder did not put these near each other" from "the graph failed to connect
them".

The query is the first row and column. Its entries are 1..10 by construction --
the query says all ten are its nearest documents. Reading a row across then asks
whether the documents say the same about each other. On a fragmented query they
do not and the panel is a star: one bright cross through the query, darkness
elsewhere. On a coherent one the whole block is bright.

Three panels: an easy query, a fragmented one, and the fragmented one again on
the trained embedding. Each arm is drawn on ITS OWN exact top-10, the metric
contract the rest of the paper uses. What happened to the BASELINE's ten under
the trained embedding is printed as well, because that isolates the repair from a
change of target set.

Queries are named by id, and a corpus subset that does not contain one falls back
on the rule the pair was chosen under rather than failing.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import setup                                        # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.colors import LogNorm                          # noqa: E402
from paths import ROOT                                         # noqa: E402

LAB = ROOT
RES, EMB, FIG = LAB / "results", LAB / "emb", LAB / "figs"
L_PULL = 1100
DEEP = 400_000          # ranks past this are reported as this

# The easy side must come from OUTSIDE the hard decile or the contrast is between
# two hard queries and says nothing about the population. 1015641 is genuinely
# easy: selector score 1.00, recall 1.00 at ef=64 in all three builds.
#
# The hard side is the case the paper is about, which is NOT a query whose top-10
# is junk. 1030451 "what is the weather in allentown?" has its judged document at
# #0 and every one of its ten answers the query: three are Allentown's December
# climate, four are Allentown the city, three are another town's weather. The
# query has two facets and the ten split along them, so the two halves never
# refer to each other -- median mutual rank 788, 14 of the 45 pairs past L, with
# every document above similarity 0.55 to the query. Under SYN the same ten come
# to median 9 and 7 pairs past L: the loss moved these documents, it did not swap
# them for an easier set.
#
# 1099806 "honest kitchen foods" was the first choice and is instructive for the
# opposite reason. Its own SYN top-10 is far more coherent (1,464 -> 169), but the
# BASELINE's ten get worse under SYN (1,464 -> 2,239): there the gain came from a
# change of target set, which is not what this panel is for.
EASY, HARD = "1015641", "1030451"
# Two figures out of one measurement. The motivating figure contrasts an easy
# query with a fragmented one on the SAME embedding, which is a claim about the
# encoder; the results figure contrasts one query across two embeddings, which
# is a claim about the intervention. Drawing them from one script keeps the two
# panels of the hard query identical between them.
SETS = {
    "motivation": [(EASY, "vanilla", "coherent"),
                   (HARD, "vanilla", "fragmented")],
    "repair":     [(HARD, "vanilla", "fragmented · baseline embedding"),
                   (HARD, "syn", "the same query · SYN")],
    "all":        [(EASY, "vanilla", "easy · outside the hard decile"),
                   (HARD, "vanilla", "fragmented · baseline embedding"),
                   (HARD, "syn", "the same query · SYN")],
}
PANELS = SETS["all"]


def load_arm(corpus, arm):
    # the only figure that needs the inputs rather than the measurements: it
    # ranks each of the ten against the whole corpus, which no result file holds
    need = [EMB / f"{corpus}__{arm}_docs.npy", EMB / f"{corpus}__{arm}_queries.npy",
            RES / f"gt_{corpus}__{arm}_k10.npz"]
    miss = [q for q in need if not q.exists()]
    if miss:
        raise SystemExit(
            "fig_mutual_rank needs the document embeddings, which the artifact "
            "does not ship.\n  missing: " + "\n           ".join(str(q) for q in miss)
            + f"\n  rebuild with: encode_cache.py --corpus {corpus} --arm {arm}"
            "  and  gt.py --corpus " + corpus + f" --arm {arm} -k 10")
    X = np.load(EMB / f"{corpus}__{arm}_docs.npy", mmap_mode="r")
    Q = np.load(EMB / f"{corpus}__{arm}_queries.npy", mmap_mode="r")
    GT = np.load(RES / f"gt_{corpus}__{arm}_k10.npz")["idx"]
    D = torch.from_numpy(np.ascontiguousarray(X)).cuda().half()
    D /= D.norm(dim=1, keepdim=True).clamp(min=1e-6)
    return D, Q, GT


def ranks(T, D):
    """rank_i(j) for the ten, against the whole corpus."""
    V = D[T].float()
    rk = np.zeros((10, 10), dtype=np.int64)
    for i in range(10):
        o = np.argsort(-(D.float() @ V[i]).cpu().numpy())
        pos = {int(d): k for k, d in enumerate(o[:DEEP])}
        for j in range(10):
            rk[i, j] = pos.get(int(T[j]), DEEP)
    return rk


def matrix(row, GT, D, T=None):
    """11x11 of mutual ranks; row and column 0 are the query."""
    T = GT[row].astype(int) if T is None else np.asarray(T, int)
    rk = ranks(T, D)
    M = np.full((11, 11), np.nan)
    M[0, 1:] = np.arange(1, 11)
    M[1:, 0] = np.arange(1, 11)
    for i in range(10):
        for j in range(10):
            if i != j:
                M[i + 1, j + 1] = min(rk[i, j], rk[j, i])
    return M, T


def stats(M):
    off = M[1:, 1:][~np.isnan(M[1:, 1:])]
    return float(np.median(off)), int((off > L_PULL).sum() // 2)


def pick(qs, hard, want_hard, D, GT, exclude=()):
    """The stand-in when a printed query is not in this corpus subset.

    The rule the printed pair was chosen under: outside the decile the most
    coherent top-10, inside it the least coherent. Deterministic, no sampling,
    ties broken by query id through the sort.
    """
    scored = []
    for i in range(len(qs)):
        if (i in hard) != want_hard or qs[i]["qid"] in exclude:
            continue
        M, _ = matrix(i, GT, D)
        scored.append((stats(M)[0], qs[i]["qid"], i))
    scored.sort(reverse=want_hard)
    return scored[0][2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="msmarco-1m")
    ap.add_argument("--out", default=None)
    ap.add_argument("--panels", default="all", choices=tuple(SETS),
                    help="motivation: easy vs fragmented, one embedding. "
                         "repair: one query, both embeddings.")
    ap.add_argument("--name", default=None, help="output stem")
    a = ap.parse_args()
    global PANELS
    PANELS = SETS[a.panels]

    qf = LAB / "corpora" / f"{a.corpus}_queries.jsonl"
    if not qf.exists():
        raise SystemExit(
            f"fig_mutual_rank needs {qf}, which the artifact does not ship.\n"
            f"  rebuild with: python -m syn.corpus --corpora {a.corpus}")
    qs = [json.loads(l) for l in
          qf.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_qid = {q["qid"]: i for i, q in enumerate(qs)}
    arms = {}

    setup()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.6 * len(PANELS), 7.0))
    labels = ["q"] + [f"#{i}" for i in range(10)]
    chosen, used, im = {}, [], None
    # everything the panels are drawn from, so the figure can be redrawn without
    # the embeddings: the matrices themselves, the labels, and the scale
    dump = {"corpus": a.corpus, "k": 10, "L_pull": L_PULL, "deep_cap": DEEP,
            "quantity": "mutual rank: min(rank_i(j), rank_j(i)) over the whole "
                        "corpus; row and column 0 are the query, whose entries "
                        "are 1..10 by construction",
            "colour_scale": {"norm": "log", "vmin": 1, "vmax": 40000,
                             "cmap": "viridis_r",
                             "L_rule_at": L_PULL},
            "note": "nan on the diagonal is not missing data, it is a document "
                    "against itself; the figure draws those cells grey",
            "panels": []}
    for ax, (qid, arm, kind) in zip(axes, PANELS):
        if arm not in arms:
            arms[arm] = load_arm(a.corpus, arm)
        D, Q, GT = arms[arm]
        hard = set(np.load(RES / f"harddecile_{a.corpus}__{arm}.npz")["hard"].tolist())
        want_hard = not kind.startswith("easy")
        row = chosen.get(qid, by_qid.get(qid))
        if row is None:
            row = pick(qs, hard, want_hard, D, GT, exclude=set(used))
            print(f"  qid {qid} absent; using {qs[row]['qid']} by the rule",
                  flush=True)
        elif (row in hard) != want_hard:
            print(f"  note: qid {qid} is {'not ' if want_hard else ''}in "
                  f"{arm}'s hard decile but is labelled {kind!r}", flush=True)
        chosen[qid] = row
        used.append(qs[row]["qid"])

        rel = {str(x) for x in qs[row].get("rel", {})}
        M, T = matrix(row, GT, D)
        im = ax.imshow(M, cmap="viridis_r", norm=LogNorm(vmin=1, vmax=4e4))
        for i in range(11):
            for j in range(11):
                if np.isnan(M[i, j]):
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="0.9"))
                    continue
                v = int(M[i, j])
                ax.text(j, i, f"{v//1000}k" if v >= 10000 else f"{v:,}",
                        ha="center", va="center", fontsize=9.5,
                        color="white" if v > 300 else "black")
        ticks = list(labels)
        for i in range(10):
            if str(int(T[i])) in rel:
                ticks[i + 1] += "★"
        ax.set_xticks(range(11)); ax.set_xticklabels(ticks, fontsize=12)
        ax.set_yticks(range(11)); ax.set_yticklabels(ticks, fontsize=12)
        ax.axhline(0.5, color="black", lw=2.0)
        ax.axvline(0.5, color="black", lw=2.0)
        ax.set_title(f'"{qs[row]["text"]}"\n{kind}', pad=11, fontsize=15)
        med, beyond = stats(M)
        print(f"  {qs[row]['qid']} [{arm}] {qs[row]['text']!r}: "
              f"median {int(med):,} · {beyond} of 45 beyond {L_PULL}",
              flush=True)
        dump["panels"].append({
            "qid": qs[row]["qid"], "query": qs[row]["text"], "arm": arm,
            "kind": kind,
            "in_hard_decile": bool(row in hard),
            "tick_labels": ticks,
            "doc_ids": [int(x) for x in T],
            "judged_relevant": [bool(str(int(x)) in rel) for x in T],
            "matrix": [[None if np.isnan(v) else int(v) for v in r] for r in M],
            "median_offdiagonal": float(med),
            "pairs_beyond_L": int(beyond), "pairs_total": 45})

    # the baseline's ten, measured on the trained embedding: the repair without
    # a change of target set
    if "syn" in arms and HARD in chosen:
        Dv, _, GTv = arms["vanilla"]
        Ds, _, _ = arms["syn"]
        Tv = GTv[chosen[HARD]].astype(int)
        mv, bv = stats(matrix(chosen[HARD], GTv, Dv, T=Tv)[0])
        ms, bs = stats(matrix(chosen[HARD], GTv, Ds, T=Tv)[0])
        print(f"  the baseline's own ten, under SYN: median {int(mv):,} -> "
              f"{int(ms):,}, {bv} -> {bs} of 45 beyond {L_PULL}", flush=True)
        dump["baseline_ten_under_syn"] = {
            "what": "the BASELINE's exact top-10 for this query, scored on both "
                    "embeddings. It isolates the repair from a change of target "
                    "set: the same ten documents, moved.",
            "qid": HARD,
            "median_vanilla": float(mv), "median_syn": float(ms),
            "beyond_L_vanilla": int(bv), "beyond_L_syn": int(bs),
            "pairs_total": 45,
            "matrix_vanilla": [[None if np.isnan(v) else int(v) for v in r]
                               for r in matrix(chosen[HARD], GTv, Dv, T=Tv)[0]],
            "matrix_syn": [[None if np.isnan(v) else int(v) for v in r]
                           for r in matrix(chosen[HARD], GTv, Ds, T=Tv)[0]]}

    cb = fig.colorbar(im, ax=axes, fraction=0.019, pad=0.015)
    cb.set_label("mutual rank   min(rank$_i$(j), rank$_j$(i))", fontsize=13)
    cb.ax.axhline(L_PULL, color="crimson", lw=2.2)
    cb.ax.annotate("$L$", xy=(0.0, L_PULL), xytext=(-0.55, L_PULL),
                   textcoords=("axes fraction", "data"), color="crimson",
                   fontsize=15, fontweight="bold", va="center", ha="center")
    # every paper figure lands in figures under its script's own name, so
    # the collector can pick them up by one rule
    d = Path(a.out) if a.out else FIG / "final"
    d.mkdir(parents=True, exist_ok=True)
    stem = a.name or ("fig_mutual_rank" if a.panels == "all"
                      else f"fig_mutual_rank_{a.panels}")
    for e in ("pdf", "png"):
        fig.savefig(d / f"{stem}.{e}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {d}/{stem}.pdf / .png")
    j = RES / f"mutual_rank_panels_{a.panels}.json"
    j.write_text(json.dumps(dump, indent=2))
    print(f"wrote {j}  ({len(dump['panels'])} panels, redraws without the embeddings)")


if __name__ == "__main__":
    main()

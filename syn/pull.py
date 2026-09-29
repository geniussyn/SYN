"""Stage 1b -- active pairs, boundary bands, positive law.

Two definitions a reimplementation cannot guess, both read off the baseline
geometry and both frozen for the whole run:

  A_q = the pairs of S_q that cannot reach each other in one hop
        Level 0 of the index keeps at most maxM0 = 2M edges per node, so the
        most any builder can do is join each document to its true top-maxM0.
        Under that rule u and v are one hop apart exactly when
        min(rank_u(v), rank_v(u)) <= maxM0, so L = maxM0 makes the active set
        literally "the pairs phi_1 counts as broken". A pair already one hop
        apart needs nothing: pulling it would move documents other queries
        depend on for no gain.

        L is therefore not a tuned threshold but the index's own edge budget.
        It stays exact only at one hop: an h-hop ball is bounded by (2M)^h but
        is far smaller in practice -- the calibration puts the 3-hop horizon at
        rank 130 against a bound of 32,768 -- so a rank threshold stands in for
        multi-hop reachability only loosely, and h > 1 would have to compute
        reachability rather than threshold it.

  B(v) = N_L(v)[L_lo : L]
        a reference drawn from a band at v's own top-L boundary. Neighbour
        density varies across the corpus, so a fixed similarity threshold means
        a different rank in every region; a band does not. In expectation b_u is
        the rank-L neighbour; the band only reduces the variance of that
        estimate, and is not a second threshold.

L is a structural constant of the index, not a tuned one: `--reach-hops 1` with
the index's M gives L = maxM0 = 2M, the exact one-hop condition. Larger h only
bounds the ball, so those settings are approximations and are marked as such in
the manifest.

Ranks are exhaustive and accumulated in fp32 (see `syn.exact`).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from syn.exact import load_doc_matrix, neighbourhoods               # noqa: E402
from syn.env import ROOT, log, provenance, write_json               # noqa: E402


def active_pairs(S, NL, anchors):
    """Indices (q, i, j) of the pairs of S_q that lie outside each other's N_L."""
    row_of = {int(d): r for r, d in enumerate(anchors)}
    nq, K = S.shape
    iu, ju = np.triu_indices(K, 1)
    pq, pi, pj = [], [], []
    for q in range(nq):
        rows = NL[[row_of[int(d)] for d in S[q]]]              # (K, L)
        inN = (rows[:, None, :] == S[q][None, :, None]).any(-1)  # inN[i,j]: S[j] in N_L(S[i])
        act = ~inN[iu, ju] & ~inN[ju, iu]
        for i, j in zip(iu[act], ju[act]):
            pq.append(q)
            pi.append(i)
            pj.append(j)
    return (np.asarray(pq, dtype=np.int32), np.asarray(pi, dtype=np.int16),
            np.asarray(pj, dtype=np.int16))


def main():
    ap = argparse.ArgumentParser(description="Stage 1b: active pairs and bands")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--corpora-dir", default=str(ROOT / "corpora"))
    ap.add_argument("--mine-dir", required=True)
    ap.add_argument("--doc-emb", required=True)
    ap.add_argument("--pull-l", type=int, default=0,
                    help="L. 0 derives it from the index as (2M)^hops.")
    ap.add_argument("--graph-m", type=int, default=16,
                    help="the M the index is built with; maxM0 = 2M")
    ap.add_argument("--reach-hops", type=int, default=1,
                    help="h: the reachability the method targets. phi_1 -> 1.")
    ap.add_argument("--pull-low", type=int, default=0,
                    help="L_lo, the band floor. 0 uses 0.75*L.")
    ap.add_argument("--pos-tau", type=float, default=0.02)
    ap.add_argument("--qchunk", type=int, default=64)
    ap.add_argument("--name", default="pull",
                    help="output stem, so several L settings can coexist")
    a = ap.parse_args()

    L = a.pull_l or (2 * a.graph_m) ** a.reach_hops
    L_lo = a.pull_low or max(1, int(round(0.75 * L)))
    if not 0 < L_lo < L:
        raise SystemExit(f"require 0 < L_lo < L; got {L_lo} and {L}")
    if a.pull_l:
        log(f"L = {L} (given), band {L_lo}-{L}")
    else:
        exact = a.reach_hops == 1
        log(f"L = (2*{a.graph_m})^{a.reach_hops} = {L} "
            f"({'exact one-hop reachability' if exact else 'a loose bound on the '
              f'{a.reach_hops}-hop ball'}), band {L_lo}-{L}")
    src = Path(a.mine_dir)
    D = np.load(src / "mine.npz", allow_pickle=True)
    S, S_sim = D["S_train"], D["S_sim"]
    man = json.loads((Path(a.corpora_dir) / f"{a.corpus}_manifest.json").read_text())
    X = load_doc_matrix(a.doc_emb, man["n_docs_subset"])

    anchors = np.unique(S)
    log(f"{S.shape[0]:,} queries, K={S.shape[1]}, {anchors.size:,} distinct anchors")
    log(f"N_L: exhaustive top-{L} per anchor, fp32")
    NL = neighbourhoods(anchors, X, L, qchunk=a.qchunk)
    band = np.ascontiguousarray(NL[:, L_lo:L])

    pq, pi, pj = active_pairs(S, NL, anchors)
    nq, K = S.shape
    per_q = np.bincount(pq, minlength=nq)
    n_pairs = K * (K - 1) // 2
    log(f"active pairs: {pq.size:,} total, {per_q.mean():.2f} of {n_pairs} per query, "
        f"{(per_q == 0).mean():.1%} of queries have none")

    # Positive law, fixed: softmax over the baseline query-document scores, so a
    # document the baseline ranks higher is drawn as the positive more often
    # without excluding the rest of S_q.
    z = S_sim.astype(np.float64) / a.pos_tau
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    eff = 1.0 / (p ** 2).sum(1)
    log(f"positive law tau_p={a.pos_tau}: median p(top-1) {np.median(p[:, 0]):.3f}, "
        f"effective candidates {eff.mean():.2f} of {K}")

    np.savez_compressed(src / f"{a.name}.npz", anchors=anchors, band=band,
                        pair_q=pq, pair_i=pi, pair_j=pj,
                        pos_prob=p.astype(np.float32),
                        L=L, L_lo=L_lo, pos_tau=a.pos_tau)
    write_json(src / f"{a.name}_manifest.json", {
        "stage": "pull", "corpus": a.corpus, "doc_emb": str(a.doc_emb),
        "L": L, "L_lo": L_lo, "band_width": L - L_lo,
        "L_derived": not bool(a.pull_l), "graph_M": a.graph_m,
        "reach_hops": a.reach_hops,
        "active_set_is_exact_reachability": (not a.pull_l) and a.reach_hops == 1,
        "pos_tau": a.pos_tau, "n_queries": int(nq), "K": int(K),
        "n_anchors": int(anchors.size), "n_active_pairs": int(pq.size),
        "active_pairs_per_query_mean": float(per_q.mean()),
        "active_pairs_per_query_max": int(per_q.max()) if pq.size else 0,
        "share_queries_no_active_pair": float((per_q == 0).mean()),
        "pairs_per_query_possible": n_pairs,
        "pos_median_p_top1": float(np.median(p[:, 0])),
        "pos_effective_candidates": float(eff.mean()),
        "rank_precision": "fp32", **provenance()})
    log(f"wrote {src / (a.name + '.npz')}")


if __name__ == "__main__":
    main()

"""Data for HNSW-NGFix*, in the formats its binaries read.

NGFix repairs a graph using HISTORICAL queries, and SYN trains its adapter on
queries too, so the only comparison that means anything is one where both draw
from the same pool under the same filter. Two modes:

  pool   the collection's BEIR *train* split, leak-filtered by syn/mine.py --
         the very pool SYN's 500 queries were sampled from, 59,242 queries on
         MS MARCO. History and evaluation are disjoint by construction, so every
         arm is measured on the FULL evaluation set and the hard decile the rest
         of the paper uses. Numbers stay comparable with every other table.

  split  for collections whose BEIR repo has no training split (NQ, Quora, MS
         COCO): the evaluation queries are halved by a fixed seed, one half is
         the history, and every arm is measured on the other half and on the
         hard decile cut inside it. Those numbers are internally consistent and
         NOT comparable with the rest of the paper -- and they favour NGFix,
         which gets a per-collection query log where SYN gets nothing.

Formats: .fbin is uint32 n, uint32 d, then n*d float32; the ground truth is the
same header with int32 ids. Both are what test/tools/data_loader.h reads.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT                                          # noqa: E402

EMB, RES = ROOT / "emb", ROOT / "results"
OUT = Path("/dev/shm/ngfix")
SPLIT_SEED = 20260924


def write_bin(path, A, dtype):
    A = np.ascontiguousarray(A, dtype=dtype)
    with open(path, "wb") as f:
        np.array(A.shape, dtype=np.uint32).tofile(f)
        A.tofile(f)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--history", choices=("pool", "split"), default="split")
    ap.add_argument("--pool-corpus", default=None,
                    help="pool mode: the pseudo-collection holding the encoded "
                         "train-split pool, e.g. msmarco-1m-trainpool")
    ap.add_argument("--history-n", type=int, default=0,
                    help="pool mode: how many historical queries to give NGFix; "
                         "0 is all of them. The sample is seeded and nested, so "
                         "a budget sweep is one pool drawn once.")
    ap.add_argument("--frac", type=float, default=0.5,
                    help="split mode: share of the evaluation queries given to "
                         "NGFix as history")
    ap.add_argument("--decile", type=float, default=0.10)
    ap.add_argument("--no-base", action="store_true",
                    help="skip the 4 GB base vectors (already written)")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.corpus}__{a.arm}"
    Q = np.load(EMB / f"{a.corpus}__{a.arm}_queries.npy").astype(np.float32)
    gt = np.load(RES / f"gt_{a.corpus}__{a.arm}_k10.npz")["idx"].astype(np.int32)
    sel = np.load(RES / f"harddecile_{a.corpus}__{a.arm}.npz")["mean_recall"]

    H = None
    if a.history == "pool":
        if not a.pool_corpus:
            raise SystemExit("--history pool needs --pool-corpus")
        P = np.load(EMB / f"{a.pool_corpus}__{a.arm}_queries.npy").astype(np.float32)
        n = len(P) if a.history_n <= 0 else min(a.history_n, len(P))
        # nested by construction: one permutation, prefixes of it are the budgets
        pick = np.random.default_rng(SPLIT_SEED).permutation(len(P))[:n]
        H = P[np.sort(pick)]
        # history is a disjoint split, so nothing is held out of the evaluation
        hist, test = np.array([], np.int64), np.arange(len(Q))
        hard = np.load(RES / f"harddecile_{a.corpus}__{a.arm}.npz")["hard"]
    else:
        perm = np.random.default_rng(SPLIT_SEED).permutation(len(Q))
        nh = int(round(a.frac * len(Q)))
        hist, test = np.sort(perm[:nh]), np.sort(perm[nh:])
        # the same rule as the rest of the paper, applied inside the test half
        order = test[np.argsort(sel[test], kind="stable")]
        hard = np.sort(order[:max(1, int(round(a.decile * len(test))))])

    if not a.no_base:
        X = np.load(EMB / f"{a.corpus}__{a.arm}_docs.npy", mmap_mode="r")
        p = OUT / f"{tag}_base.fbin"
        with open(p, "wb") as f:
            np.array(X.shape, dtype=np.uint32).tofile(f)
            for s in range(0, X.shape[0], 200_000):
                np.ascontiguousarray(X[s:s + 200_000], dtype=np.float32).tofile(f)
        print(f"  base   {X.shape} -> {p}")

    write_bin(OUT / f"{tag}_hist.fbin",
              H if H is not None else Q[hist], np.float32)
    write_bin(OUT / f"{tag}_test.fbin", Q[test], np.float32)
    write_bin(OUT / f"{tag}_test_gt.ibin", gt[test], np.int32)
    write_bin(OUT / f"{tag}_hard.fbin", Q[hard], np.float32)
    write_bin(OUT / f"{tag}_hard_gt.ibin", gt[hard], np.int32)
    nh = len(H) if H is not None else len(hist)
    np.savez_compressed(RES / f"ngfix_split_{tag}.npz", hist=hist, test=test,
                        hard=hard, split_seed=SPLIT_SEED, frac=a.frac,
                        mode=a.history, n_history=nh,
                        pool_corpus=a.pool_corpus or "")
    print(f"  {tag}: history {nh:,} ({a.history}) · test {len(test):,} · "
          f"hard decile {len(hard):,}")


if __name__ == "__main__":
    main()

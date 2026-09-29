"""Gold recall at depth, per arm, on that arm's own hard decile.

recall@10 against an arm's own exact top-10 says what the index realises of its
encoder. It says nothing about whether the answer is in the list at all. This
measures qrels recall at k = 10, 50, 100, 1000 -- the ceiling no reranker
downstream can exceed -- twice: as the index returns it, and exhaustively. The
two questions (does the encoder rank it, does the index find it) stay separate.

Retrieval here is NOT a timed measurement: it takes neither the lock nor the
reserved core, and runs multi-threaded. Nothing downstream quotes its
milliseconds.

Differences from the withdrawn version: the stratum is the arm's own, and the
exhaustive reference is accumulated in fp32 (it was fp16, which disagrees with
the index on near-tied documents) and is not re-normalised -- the index searches
inner product over the stored vectors, so the reference has to do the same.
"""
import argparse, gc, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import EMB, ROOT as LAB                          # noqa: E402
RES = LAB / "results"
from index_hnsw import build, timed_grid                       # noqa: E402
from metrics import qrels_recall_at_k                          # noqa: E402
from gt import exact_topk                                      # noqa: E402
from run_cell import load_rels                                 # noqa: E402

GOLD_K = (10, 50, 100, 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--arms", default="vanilla,syn")
    ap.add_argument("-M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    a = ap.parse_args()

    rels = load_rels(a.corpus)
    out = {"corpus": a.corpus, "M": a.M, "efConstruction": a.efc,
           "note": "each arm on its own hard decile; retrieval untimed", "arms": {}}
    for arm in a.arms.split(","):
        hard = np.load(RES / f"harddecile_{a.corpus}__{arm}.npz")["hard"]
        rels_hard = [rels[i] for i in hard]
        Q = np.load(EMB / f"{a.corpus}__{arm}_queries.npy").astype(np.float32)
        ip = Path("/dev/shm/synidx") / f"goldk_{a.corpus}__{arm}__M{a.M}.bin"
        ip.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{a.corpus}/{arm}] building M={a.M}", flush=True)
        ix, bt, n, d = build(a.corpus, arm, a.M, a.efc, seed=100, index_path=str(ip))
        del ix; gc.collect()

        X = np.load(EMB / f"{a.corpus}__{arm}_docs.npy", mmap_mode="r")
        gold, n_hard = {}, int(len(hard))
        for k in GOLD_K:
            # ef has to clear k by a margin or the list is truncated by the
            # search rather than by the depth being asked about
            r = timed_grid(str(ip), Q, ef_grid=[max(2 * k, 512)], k=k, reps=1,
                           pin=False, lock=False)[0]
            ret = r["labels"]
            ex, _ = exact_topk(Q, X, k)
            gold[str(k)] = {
                "ann_hard": float(qrels_recall_at_k(ret[hard], rels_hard, k).mean()),
                "ann_all": float(qrels_recall_at_k(ret, rels, k).mean()),
                "exhaustive_hard": float(qrels_recall_at_k(
                    ex[hard].astype(np.int32), rels_hard, k).mean()),
                "exhaustive_all": float(qrels_recall_at_k(
                    ex.astype(np.int32), rels, k).mean())}
            g = gold[str(k)]
            print(f"  [{arm}] gold@{k:<5} ANN {g['ann_hard']:.4f} / exact "
                  f"{g['exhaustive_hard']:.4f}  (hard, n={n_hard})", flush=True)
        out["arms"][arm] = {"n_hard": n_hard, "build_time_s": bt, "gold": gold}
        ip.unlink(missing_ok=True)

    p = RES / f"goldk_{a.corpus}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p.name}", flush=True)


if __name__ == "__main__":
    main()

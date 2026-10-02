"""End-to-end: retrieve R candidates, rerank them, measure the finished list.

A retrieval gain only matters if it survives the stage that follows. This runs
the pipeline a deployment runs -- retrieve the top R with the ANN index, score
those R with a cross-encoder, keep the best 10 -- and reports nDCG@10, MRR@10 and
gold recall@10 on the RERANKED list against the sum of both latencies.

Why the depth sweep is the point. The reranker is the expensive stage and its
cost is linear in R, so an arm whose candidate list is better can stop at a
smaller R and pay less for the same final quality. That is the question the MRR
panels raise and cannot answer: SYN retrieves its embedding's ranking far faster
but that ranking scores lower against qrels, and whether the pipeline cares
depends on whether the right document is in the list at all for a reranker to
promote.

Both arms get the same reranker, the same batch size and the same max length, so
the model cannot favour either; only the candidate list differs. Each arm is
read on its OWN hard decile, as everywhere else.

Search latency is NOT re-timed here. It is taken from the measured cell at the
ef used, which was timed single-threaded on the reserved core under the lock;
re-timing it from a process that also drives a GPU would be a worse measurement,
not a better one. Retrieval in this script therefore runs untimed and unpinned.
Rerank latency IS measured here, on the GPU, and charged per query.
"""
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import EMB, ROOT                                          # noqa: E402
# LAB comes from paths.ROOT: SYN_ROOT if set, else this file's parent.
# It used to be a fixed absolute path, which resolved only on the machine
# this was written on, where that path was a symlink into the tree.
LAB = ROOT; RES = LAB / "results"
from index_hnsw import build, timed_grid                        # noqa: E402
from metrics import mrr_at_k, ndcg_at_k, qrels_recall_at_k      # noqa: E402
from paths import cell_or_strata                                # noqa: E402, ROOT

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEPTHS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150)
# the ef actually used for a depth: the cheapest measured grid point that can
# return R at all. ef < R truncates the list by the search rather than by R.
# must be ef values the timed cell actually measured, or the search term of the
# total latency has no measurement behind it
GRID = (64, 96, 128, 192, 256, 384, 512, 768, 1024)


def ef_for(R):
    return next(e for e in GRID if e >= max(R, 64))


def search_ms(cell, ef, stratum="hard"):
    xk = "ms_per_query_hard" if stratum == "hard" else "ms_per_query"
    for r in cell["grid"]:
        if r["ef"] == ef and xk in r:
            return r[xk]
    return None


def load_reranker(name):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        name, torch_dtype=torch.float16).cuda().eval()
    return tok, mdl


@torch.no_grad()
def rerank(tok, mdl, qtexts, dtexts, cand, bs=512, max_len=256):
    n, K = cand.shape
    pairs = [(qtexts[i], dtexts[int(d)]) for i in range(n) for d in cand[i]]
    torch.cuda.synchronize(); t0 = time.time()
    out = np.empty(len(pairs), dtype=np.float32)
    for s in range(0, len(pairs), bs):
        ch = pairs[s:s + bs]
        enc = tok([a for a, _ in ch], [b for _, b in ch], padding=True,
                  truncation=True, max_length=max_len, return_tensors="pt").to("cuda")
        out[s:s + len(ch)] = mdl(**enc).logits.squeeze(-1).float().cpu().numpy()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / n * 1000.0
    order = np.argsort(-out.reshape(n, K), axis=1)
    return np.take_along_axis(cand, order, axis=1), ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="msmarco-1m")
    ap.add_argument("--arms", default="vanilla,syn")
    ap.add_argument("-M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    a = ap.parse_args()

    qs = [json.loads(l) for l in
          (LAB / f"corpora/{a.corpus}_queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = (LAB / f"corpora/{a.corpus}_docs.txt").read_text(encoding="utf-8").split("\n")
    if docs and not docs[-1]:
        docs.pop()
    rels = [{int(k): int(v) for k, v in q.get("rel", {}).items()} for q in qs]
    tok, mdl = load_reranker(RERANKER)

    out = {"corpus": a.corpus, "reranker": RERANKER, "M": a.M,
           "efConstruction": a.efc, "depths": list(DEPTHS),
           "note": "all queries; per-query metrics saved so any difficulty "
                   "stratum is a slice. search ms from the timed cell, "
                   "rerank ms measured here",
           "arms": {}}
    for arm in a.arms.split(","):
        # ALL queries. The strata are slices of this, taken afterwards from the
        # per-query arrays, so one pass of the expensive stage serves every
        # stratum instead of one pass each.
        idx = np.arange(len(qs))
        qtext = [qs[i]["text"] for i in idx]
        rh = [rels[i] for i in idx]
        # An arm added for comparison may have no measured cell, only
        # strata_cell.py's timing of the same grid. cell_or_strata reads
        # whichever exists; see paths.py for why that is the same number.
        cell = cell_or_strata(a.corpus, arm, a.M, a.efc, results=RES)
        Q = np.load(EMB / f"{a.corpus}__{arm}_queries.npy").astype(np.float32)
        ip = Path("/dev/shm/synidx") / f"rr_{a.corpus}__{arm}__M{a.M}.bin"
        ip.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{a.corpus}/{arm}] building (untimed)", flush=True)
        ix, *_ = build(a.corpus, arm, a.M, a.efc, seed=100, index_path=str(ip))
        del ix; gc.collect()

        rows, pq = [], {}
        for R in DEPTHS:
            ef = ef_for(R)
            r = timed_grid(str(ip), Q, ef_grid=[ef], k=R, reps=1,
                           pin=False, lock=False)[0]
            cand = r["labels"]
            rr, ms_rr = rerank(tok, mdl, qtext, docs, cand)
            ms_s = search_ms(cell, ef, "all")
            row = {"R": R, "ef": ef, "ms_search": ms_s, "ms_rerank": ms_rr,
                   "ms_total": (ms_s + ms_rr) if ms_s else None}
            for tag, lst in (("pre", cand), ("post", rr)):
                nd = ndcg_at_k(lst, rh, 10); mr = mrr_at_k(lst, rh, 10)
                gd = qrels_recall_at_k(lst, rh, 10)
                row[f"ndcg10_{tag}"] = float(nd.mean())
                row[f"mrr10_{tag}"] = float(mr.mean())
                row[f"gold10_{tag}"] = float(gd.mean())
                pq[f"ndcg_{tag}_{R}"] = nd.astype(np.float32)
                pq[f"mrr_{tag}_{R}"] = mr.astype(np.float32)
                pq[f"gold_{tag}_{R}"] = gd.astype(np.float32)
            # The ceiling the reranker is handed: qrels recall of the WHOLE
            # depth-R list rather than of its first ten. Reranking permutes those
            # R documents, so it is the same before and after. Without it a low
            # gold10_post cannot be attributed -- gold@R minus gold10_post is what
            # the reranker failed to promote, 1 minus gold@R is what the search
            # never returned, and the two call for opposite fixes.
            gR = qrels_recall_at_k(cand, rh, R)
            row["goldR"] = float(gR.mean())
            pq[f"goldR_{R}"] = gR.astype(np.float32)
            rows.append(row)
            print(f"  [{arm}] R={R:<4} ef={ef:<4} search {ms_s:6.2f} + rerank "
                  f"{ms_rr:7.1f} ms · nDCG {row['ndcg10_pre']:.4f} -> "
                  f"{row['ndcg10_post']:.4f}", flush=True)
        out["arms"][arm] = {"n_queries": int(len(idx)), "rows": rows}
        # per-query, so any difficulty stratum is a slice rather than a re-run
        np.savez_compressed(RES / f"rerank_pq_{a.corpus}__{arm}.npz", **pq)
        ip.unlink(missing_ok=True)

    # MERGE rather than overwrite. The cross-encoder pass is per arm and the
    # arms do not fit on disk at once, so they are run one at a time; writing the
    # file fresh each run would leave only the last arm in it.
    p = RES / f"rerank_pipeline_all_{a.corpus}.json"
    if p.exists():
        prev = json.loads(p.read_text())
        merged = dict(prev.get("arms", {}))
        merged.update(out["arms"])
        out["arms"] = merged
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p.name}", flush=True)


if __name__ == "__main__":
    main()

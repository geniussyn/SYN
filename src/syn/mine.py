"""Stage 1a -- frozen baseline signals: target sets, regions, monitor assets.

Everything written here is computed once, on the baseline encoder, and never
refreshed during training. S_q is that encoder's exact top-K over the corpus,
not a relevance judgement: no qrels enter the objective at any point. Regions
are a k-means partition of the document matrix and supply the negative pool
C(v) of each anchor.

Training queries come from a split disjoint from the evaluation split, and are
filtered for leakage on three deterministic criteria before sampling.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from syn.corpus import ALIASES, CORPORA, QUERIES_REL, HF, _download, \
    _read_parquet_records, load_corpus, load_qrels, text_of         # noqa: E402
from syn.exact import load_doc_matrix, topk                         # noqa: E402
from syn.encoders import REGISTRY, encode, load_encoder, spec_record  # noqa: E402
from syn.env import ROOT, disk_guard, log, provenance, write_json   # noqa: E402

TRAIN_SEED = 20260809


def load_train_queries(corpus, split, work_dir):
    src = ALIASES.get(corpus, corpus)
    repo = CORPORA[src]["repo"]
    qrels = load_qrels(repo, split)
    work_dir.mkdir(parents=True, exist_ok=True)
    qpath = work_dir / f"{corpus}_queries_{split}.parquet"
    try:
        _download(f"{HF}/{repo}/resolve/main/{QUERIES_REL}", qpath)
        qtext = {r["_id"]: text_of(r) for r in _read_parquet_records(qpath)
                 if r["_id"] in qrels}
    finally:
        qpath.unlink(missing_ok=True)
    return qrels, qtext


def kmeans_assign(X, n_clusters, seed, niter=25, backend="faiss"):
    """Spherical k-means over the document matrix, in fp32.

    faiss runs on the CPU: its GPU kernels are not built for every compute
    capability, and a k-means that aborts mid-run is worse than a slower one.
    """
    Xf = np.ascontiguousarray(X, dtype=np.float32)
    if backend == "faiss":
        import faiss
        faiss.omp_set_num_threads(os.cpu_count() or 1)
        km = faiss.Kmeans(Xf.shape[1], n_clusters, niter=niter, seed=seed,
                          verbose=False)
        km.train(Xf)
        assign = km.index.search(Xf, 1)[1].ravel().astype(np.int32)
        log(f"  k-means: faiss cpu, {n_clusters} regions")
        return assign
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xg = torch.from_numpy(Xf).to(dev)
    C = Xg[torch.from_numpy(rng.choice(len(Xf), n_clusters, replace=False)).to(dev)].clone()
    C /= C.norm(dim=1, keepdim=True).clamp(min=1e-9)
    for _ in range(niter):
        a = torch.empty(len(Xg), dtype=torch.long, device=dev)
        for s in range(0, len(Xg), 100_000):
            a[s:s + 100_000] = (Xg[s:s + 100_000] @ C.T).argmax(1)
        Cn = torch.zeros_like(C).index_add_(0, a, Xg)
        cnt = torch.bincount(a, minlength=n_clusters).unsqueeze(1)
        live = cnt.squeeze(1) > 0
        C[live] = Cn[live] / cnt[live]
        C /= C.norm(dim=1, keepdim=True).clamp(min=1e-9)
    out = torch.empty(len(Xg), dtype=torch.int32, device=dev)
    for s in range(0, len(Xg), 100_000):
        out[s:s + 100_000] = (Xg[s:s + 100_000] @ C.T).argmax(1).int()
    log(f"  k-means: torch {dev}, {n_clusters} regions")
    return out.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="Stage 1a: frozen baseline signals")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--corpora-dir", default=str(ROOT / "corpora"))
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--train-queries-jsonl", default=None,
                    help="For collections whose BEIR repo has no training-split "
                         "qrels. Takes {qid,text[,rel]}; the leak filters still apply.")
    ap.add_argument("--encoder", default="bge-m3")
    ap.add_argument("--doc-emb", required=True,
                    help="Document matrix the index is built from. Mining must use "
                         "the same vectors, or the target set is not the set the "
                         "index holds.")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--k", type=int, default=10, help="anchors per query, |S_q|")
    ap.add_argument("--n-clusters", type=int, default=2048)
    ap.add_argument("--kmeans", choices=["faiss", "torch"], default="faiss")
    ap.add_argument("--n-monitor", type=int, default=64)
    ap.add_argument("--monitor-pool", type=int, default=200)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=TRAIN_SEED)
    ap.add_argument("--require-gold-in-subset", action="store_true",
                    help="Keep a training query only if at least one of its "
                         "positives is inside the subset.")
    ap.add_argument("--no-leak-filter", action="store_true")
    ap.add_argument("--dump-pool", default=None,
                    help="Write the leak-filtered eligible training queries and "
                         "stop. The pool a query-supervised baseline must share.")
    ap.add_argument("--allow-remote-code", action="store_true")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    cdir = Path(a.corpora_dir)
    docs, eval_qtext, eval_rel, eval_qids, man = load_corpus(a.corpus, cdir)
    if man["eval_split"] == a.train_split:
        raise SystemExit(f"train split equals eval split: {a.train_split}")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    disk_guard(0.5, path=out, label=f"{a.corpus} mine")

    if a.train_queries_jsonl:
        rows = [json.loads(l) for l in Path(a.train_queries_jsonl)
                .read_text(encoding="utf-8").split("\n") if l.strip()]
        tr_qtext = {str(r["qid"]): r["text"] for r in rows}
        tr_qrels = {str(r["qid"]): {str(k): int(v)
                                    for k, v in (r.get("rel") or {}).items()}
                    for r in rows}
    else:
        tr_qrels, tr_qtext = load_train_queries(a.corpus, a.train_split, ROOT / "work")
    n_before = len(tr_qtext)
    log(f"train split {a.train_split}: {len(tr_qrels):,} qrels, {n_before:,} texts")

    # ---- leakage: three deterministic criteria, taken as a union.
    # L1 a training query shares a judged document with an evaluation query
    # L2 the training query id is itself an evaluation gold document
    # L3 the training query text equals an evaluation gold document
    # L3 is the path that matters on Quora, where queries and documents are both
    # questions and dev/test can be near-duplicates even with disjoint ids.
    sub = np.load(cdir / f"subset_{a.corpus}.npz", allow_pickle=True)
    doc_ids = [str(x) for x in sub["doc_ids"]]
    subset_ids = set(doc_ids)
    eval_doc_local = {d for R in eval_rel for d in R}
    eval_doc_ids = {doc_ids[i] for i in eval_doc_local}

    def _norm(s):
        return " ".join(s.lower().split())

    gold_text = {_norm(docs[i]) for i in eval_doc_local}
    L1 = {q for q, v in tr_qrels.items() if q in tr_qtext and (set(v) & eval_doc_ids)}
    L2 = {q for q in tr_qtext if q in eval_doc_ids}
    L3 = {q for q, t in tr_qtext.items() if _norm(t) in gold_text}
    leaked = L1 | L2 | L3
    log(f"leak filter: L1 {len(L1):,}  L2 {len(L2):,}  L3 {len(L3):,}  "
        f"union {len(leaked):,}")

    elig = ({q for q, v in tr_qrels.items()
             if q in tr_qtext and any(s > 0 and d in subset_ids for d, s in v.items())}
            if any(tr_qrels.values()) else set(tr_qtext))
    pool = set(tr_qtext)
    if a.require_gold_in_subset:
        pool &= elig
    if not a.no_leak_filter:
        pool -= leaked
    keep = sorted(pool)
    log(f"eligible {len(elig):,}  after leak filter {len(elig - leaked):,}  "
        f"usable {len(keep):,}")
    if a.dump_pool:
        # The whole eligible pool, not the sample. A query-supervised baseline
        # compared against this one has to draw from the same pool, under the
        # same leak filter, or the two are not spending the same thing.
        Path(a.dump_pool).write_text("\n".join(
            json.dumps({"qid": q, "text": tr_qtext[q]}, ensure_ascii=False)
            for q in keep), encoding="utf-8")
        log(f"wrote the eligible pool ({len(keep):,} queries) to {a.dump_pool}")
        return
    if len(keep) < a.n_train:
        log(f"WARNING only {len(keep):,} usable training queries < {a.n_train:,}")

    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(keep), min(a.n_train, len(keep)), replace=False)
    train_q = [keep[i] for i in sorted(pick)]
    train_text = [tr_qtext[q] for q in train_q]
    rest = [q for q in keep if q not in set(train_q)]
    mpick = rng.choice(len(rest), min(a.n_monitor, len(rest)), replace=False)
    mon_q = [rest[i] for i in sorted(mpick)]
    mon_text = [tr_qtext[q] for q in mon_q]
    log(f"training queries {len(train_q):,}, monitor queries {len(mon_q):,} (disjoint)")

    X = load_doc_matrix(a.doc_emb, len(docs))
    mf = Path(str(a.doc_emb).replace("_docs.npy", "_manifest.json"))
    tr_d = json.loads(mf.read_text()).get("truncation_rate_docs") if mf.exists() else None
    log(f"document matrix {X.shape} {X.dtype} from {a.doc_emb}")

    max_len = a.max_len or (256 if a.corpus.startswith("msmarco") else None)
    tok, mdl, spec = load_encoder(a.encoder, allow_remote_code=a.allow_remote_code)
    Qt, tr_q_rate = encode(mdl, tok, spec, train_text, is_query=True, bs=a.bs,
                           max_len=max_len, log_every=0)
    Qm, _ = encode(mdl, tok, spec, mon_text, is_query=True, bs=a.bs,
                   max_len=max_len, log_every=0)
    del mdl
    torch.cuda.empty_cache()

    S_train, S_sim = topk(Qt, X, a.k)
    log(f"S_q: {S_train.shape}, {np.unique(S_train).size:,} distinct anchors")

    assign = kmeans_assign(X, a.n_clusters, a.seed, backend=a.kmeans)
    sizes = np.bincount(assign, minlength=a.n_clusters)
    log(f"  region sizes min {sizes.min():,} median {int(np.median(sizes)):,} "
        f"max {sizes.max():,}")
    anchor_region = sizes[assign[np.unique(S_train)]]
    if anchor_region.min() <= a.k:
        log(f"WARNING smallest anchor region holds {anchor_region.min()} documents")

    # Monitor frame: a fixed reference pool, so the in-loop check re-encodes a few
    # thousand documents rather than the corpus. Watched, not measured.
    S_mon_full = topk(Qm, X, a.monitor_pool, return_sim=False)
    mon_pool = np.unique(S_mon_full)
    remap = {int(g): i for i, g in enumerate(mon_pool)}
    S_mon = np.array([[remap[int(g)] for g in row[:a.k]] for row in S_mon_full],
                     dtype=np.int32)

    np.savez_compressed(out / "mine.npz", S_train=S_train, S_sim=S_sim,
                        assign=assign, n_clusters=a.n_clusters, k=a.k,
                        train_qids=np.array(train_q, dtype=object))
    np.savez_compressed(out / "monitor.npz", pool=mon_pool, S_mon=S_mon,
                        mon_qids=np.array(mon_q, dtype=object))
    for fn, qs, ts in (("queries_train.jsonl", train_q, train_text),
                       ("queries_monitor.jsonl", mon_q, mon_text)):
        (out / fn).write_text("\n".join(
            json.dumps({"qid": q, "text": t}, ensure_ascii=False)
            for q, t in zip(qs, ts)), encoding="utf-8")

    write_json(out / "manifest.json", {
        "stage": "mine", "corpus": a.corpus, "train_split": a.train_split,
        "eval_split": man["eval_split"], "encoder": a.encoder,
        "doc_emb": str(a.doc_emb), "n_docs": len(docs),
        "n_train_queries": len(train_q), "K": a.k, "seed": a.seed,
        "n_clusters": a.n_clusters,
        "region_size_min": int(sizes.min()), "region_size_max": int(sizes.max()),
        "region_size_median": float(np.median(sizes)),
        "leak_filter": not a.no_leak_filter, "n_train_before_filter": n_before,
        "leak_L1_shared_judgement": len(L1), "leak_L2_qid_is_gold_doc": len(L2),
        "leak_L3_qtext_is_gold_doc": len(L3), "n_leaked_excluded": len(leaked),
        "require_gold_in_subset": bool(a.require_gold_in_subset),
        "n_eligible": len(elig), "n_eligible_after_leak_filter": len(elig - leaked),
        "n_pool_after_all_filters": len(keep),
        "n_monitor_queries": len(mon_q), "monitor_pool_top": a.monitor_pool,
        "monitor_pool_size": int(mon_pool.size),
        "truncation_rate_docs": tr_d, "truncation_rate_train_queries": tr_q_rate,
        "topk_precision": "fp32", **provenance(**spec_record(spec, max_len))})
    log(f"wrote {out}")


if __name__ == "__main__":
    main()

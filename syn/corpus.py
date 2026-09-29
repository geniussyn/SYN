"""BEIR collection -> fixed-size nested subset.

The subset, not the full collection, is the corpus of this study: exact top-k is
defined inside it, so the accepted id set is the unit of reproduction and is
written to `subset_<name>.npz`.

Rule: every judged document is kept, the remainder is filled uniformly at random
to exactly `n_target`, under a fixed seed. BEIR ships parquet, which does not
stream by line, so the file is downloaded to the work directory and deleted in a
`finally` as soon as the subset is taken.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from syn.env import ROOT, disk_guard, log, provenance, write_json  # noqa: E402

SUBSET_SEED = 20260809
HF = "https://huggingface.co/datasets"
CORPUS_REL = "corpus/corpus-00000-of-00001.parquet"
QUERIES_REL = "queries/queries-00000-of-00001.parquet"
RAW_CACHE = Path(os.environ.get("SYN_RAW_CACHE", ROOT / "raw"))

CORPORA = {
    "quora":          {"repo": "BeIR/quora",          "split": "test", "n_docs": 522_931},
    "nq":             {"repo": "BeIR/nq",             "split": "test", "n_docs": 2_681_468},
    "fever":          {"repo": "BeIR/fever",          "split": "test", "n_docs": 5_416_568},
    "hotpotqa":       {"repo": "BeIR/hotpotqa",       "split": "test", "n_docs": 5_233_329},
    "dbpedia-entity": {"repo": "BeIR/dbpedia-entity", "split": "test", "n_docs": 4_635_922},
    "climate-fever":  {"repo": "BeIR/climate-fever",  "split": "test", "n_docs": 5_416_593},
    "msmarco":        {"repo": "BeIR/msmarco",        "split": "dev",  "n_docs": 8_841_823},
    # 1M subsets: the collections the paper measures.
    "msmarco-1m":     {"repo": "BeIR/msmarco",        "split": "dev",  "n_docs": 1_000_000},
    "fever-1m":       {"repo": "BeIR/fever",          "split": "test", "n_docs": 1_000_000},
    "nq-1m":          {"repo": "BeIR/nq",             "split": "test", "n_docs": 1_000_000},
}

# The eval artifact names the 1M collections `fever` and `nq`; keep both spellings.
ALIASES = {"fever": "fever-1m", "nq": "nq-1m"}


def load_corpus(name, corpora_dir=None):
    """Read `<name>_docs.txt`, `_queries.jsonl` and `_manifest.json`.

    Only "\\n" is a line break: `splitlines()` also splits on U+0085 and U+2028,
    which occur in MS MARCO bodies and shift every row index.
    """
    d = Path(corpora_dir) if corpora_dir else ROOT / "corpora"
    docs = (d / f"{name}_docs.txt").read_text(encoding="utf-8").split("\n")
    if docs and not docs[-1]:
        docs.pop()
    qs = [json.loads(l) for l in
          (d / f"{name}_queries.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]
    man = json.loads((d / f"{name}_manifest.json").read_text())
    assert len(docs) == man["n_docs_subset"], (len(docs), man["n_docs_subset"])
    return docs, [q["text"] for q in qs], \
        [{int(k): int(v) for k, v in q["rel"].items()} for q in qs], \
        [str(q["qid"]) for q in qs], man


def _cached(url):
    import re
    m = re.search(r"datasets/BeIR/([^/]+?)(-qrels)?/resolve/main/(.+)$", url)
    if not m:
        return None
    name, is_qrels, rel = m.group(1), bool(m.group(2)), m.group(3)
    d = RAW_CACHE / name
    if is_qrels:
        p = d / f"qrels_{Path(rel).stem}.tsv"
    elif rel.startswith("corpus/"):
        p = d / "corpus.parquet"
    elif rel.startswith("queries/"):
        p = d / "queries.parquet"
    else:
        return None
    return p if p.exists() and p.stat().st_size > 0 else None


def _download(url, dst: Path, keep_cache=True):
    cached = _cached(url)
    if cached is not None:
        import shutil
        shutil.copyfile(cached, dst)
        log(f"  cache hit {cached.name}")
        return dst
    req = urllib.request.Request(url, headers={"User-Agent": "syn/1.0"})
    t0 = time.time()
    with urllib.request.urlopen(req) as resp, dst.open("wb") as f:
        while chunk := resp.read(1 << 22):
            f.write(chunk)
    log(f"  fetched {dst.name} {dst.stat().st_size / 1e6:.0f} MB ({time.time() - t0:.0f}s)")
    if keep_cache:
        import re
        import shutil
        m = re.search(r"datasets/BeIR/([^/]+?)(-qrels)?/resolve/main/(.+)$", url)
        if m:
            d = RAW_CACHE / m.group(1)
            d.mkdir(parents=True, exist_ok=True)
            rel = m.group(3)
            name = (f"qrels_{Path(rel).stem}.tsv" if m.group(2)
                    else "corpus.parquet" if rel.startswith("corpus/")
                    else "queries.parquet" if rel.startswith("queries/") else None)
            if name:
                shutil.copyfile(dst, d / name)
    return dst


def _read_parquet_records(path: Path):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    cols = set(pf.schema_arrow.names)
    want = [c for c in ("_id", "title", "text") if c in cols]
    for batch in pf.iter_batches(batch_size=100_000, columns=want):
        d = batch.to_pydict()
        for i in range(len(d[want[0]])):
            yield {c: d[c][i] for c in want}


def load_qrels(repo, split):
    """`<repo>-qrels/<split>.tsv`: query-id, corpus-id, score."""
    url = f"{HF}/{repo}-qrels/resolve/main/{split}.tsv"
    req = urllib.request.Request(url, headers={"User-Agent": "syn/1.0"})
    qrels = {}
    with urllib.request.urlopen(req) as resp:
        for i, raw in enumerate(resp):
            line = raw.decode("utf-8").rstrip("\n")
            if i == 0 and line.lower().startswith("query-id"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qrels.setdefault(parts[0], {})[parts[1]] = int(float(parts[2]))
    return qrels


def text_of(rec):
    """BEIR record -> one encodable line; title and body joined, BEIR convention."""
    t, b = (rec.get("title") or "").strip(), (rec.get("text") or "").strip()
    s = f"{t} {b}".strip() if t else b
    return s.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def _head(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "syn/1.0"})
    try:
        with urllib.request.urlopen(req) as r:
            n = r.headers.get("Content-Length") or r.headers.get("x-linked-size")
            return int(n) if n else None
    except Exception:
        return None


def build(name, n_target, out_dir, work_dir, out_name=None):
    spec = CORPORA[name]
    out_name = out_name or name
    corpus_url = f"{HF}/{spec['repo']}/resolve/main/{CORPUS_REL}"
    queries_url = f"{HF}/{spec['repo']}/resolve/main/{QUERIES_REL}"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    log(f"===== {out_name} =====")

    qrels = load_qrels(spec["repo"], spec["split"])
    judged = {d for v in qrels.values() for d in v}
    log(f"  qrels: {len(qrels):,} queries, {len(judged):,} judged documents")

    sz = _head(corpus_url) or 1_200_000_000
    disk_guard(sz / 1e9 * 1.6, path=work_dir, label=f"{out_name} parquet")
    disk_guard(sz / 1e9 * 0.6, path=out_dir, label=f"{out_name} outputs")

    cpath = work_dir / f"{out_name}_corpus.parquet"
    qpath = work_dir / f"{out_name}_queries.parquet"
    ids, texts, judged_hit = [], [], set()
    try:
        _download(corpus_url, cpath)
        for i, rec in enumerate(_read_parquet_records(cpath)):
            ids.append(rec["_id"])
            texts.append(text_of(rec))
            if rec["_id"] in judged:
                judged_hit.add(rec["_id"])
            if (i + 1) % 1_000_000 == 0:
                log(f"    read {i + 1:,} rows")
        _download(queries_url, qpath)
        qtext = {r["_id"]: text_of(r) for r in _read_parquet_records(qpath)
                 if r["_id"] in qrels}
    finally:
        cpath.unlink(missing_ok=True)
        qpath.unlink(missing_ok=True)

    n_all = len(ids)
    missing = judged - judged_hit
    log(f"  corpus {n_all:,} documents, judged found {len(judged_hit):,}/{len(judged):,}")

    idx_of = {d: i for i, d in enumerate(ids)}
    core = np.array(sorted(idx_of[d] for d in judged_hit), dtype=np.int64)
    if n_target >= n_all:
        sel = np.arange(n_all, dtype=np.int64)
    elif core.size > n_target:
        raise SystemExit(f"judged {core.size:,} > target {n_target:,}")
    else:
        rng = np.random.default_rng(SUBSET_SEED)
        rest = np.setdiff1d(np.arange(n_all, dtype=np.int64), core)
        fill = rng.choice(rest, n_target - core.size, replace=False)
        sel = np.sort(np.concatenate([core, fill]))
    log(f"  subset {sel.size:,} = {core.size:,} judged + {sel.size - core.size:,} random "
        f"(seed {SUBSET_SEED})")

    need = sum(len(texts[i]) for i in sel[:10_000]) / 10_000 * sel.size * 1.1 / 1e9
    disk_guard(need, path=out_dir, label=f"{out_name} text")
    (out_dir / f"{out_name}_docs.txt").write_text(
        "\n".join(texts[i] for i in sel), encoding="utf-8")
    sub_ids = [ids[i] for i in sel]
    np.savez_compressed(out_dir / f"subset_{out_name}.npz", local_index=sel,
                        doc_ids=np.array(sub_ids, dtype=object))

    pos = {d: j for j, d in enumerate(sub_ids)}
    kept = {q: {pos[d]: s for d, s in v.items() if d in pos}
            for q, v in qrels.items() if q in qtext}
    kept = {q: v for q, v in kept.items() if v}
    (out_dir / f"{out_name}_queries.jsonl").write_text(
        "\n".join(json.dumps({"qid": q, "text": qtext[q],
                              "rel": {str(k): v for k, v in kept[q].items()}},
                             ensure_ascii=False) for q in sorted(kept)),
        encoding="utf-8")

    # Judged is not positive: dbpedia-entity carries score-0 judgements, and
    # counting those as positives makes every relevance metric wrong.
    njudged = [len(v) for v in kept.values()]
    npos = [sum(1 for s in v.values() if s > 0) for v in kept.values()]
    graded = sorted({s for v in kept.values() for s in v.values()})
    write_json(out_dir / f"{out_name}_manifest.json", {
        "corpus": out_name, "repo": spec["repo"], "eval_split": spec["split"],
        "relevance_levels": graded, "graded_relevance": graded != [1],
        "judged_per_query_mean": float(np.mean(njudged)) if njudged else 0.0,
        "positives_per_query_mean": float(np.mean(npos)) if npos else 0.0,
        "positives_per_query_median": float(np.median(npos)) if npos else 0.0,
        "n_docs_original": int(n_all), "n_docs_subset": int(sel.size),
        "n_judged_total": len(judged), "n_judged_in_corpus": len(judged_hit),
        "n_judged_missing": len(missing), "subset_seed": SUBSET_SEED,
        "n_core": int(core.size), "n_fill": int(sel.size - core.size),
        "n_eval_queries": len(kept), **provenance()})
    log(f"  done: {sel.size:,} documents, {len(kept):,} queries")


def main():
    ap = argparse.ArgumentParser(description="BEIR -> nested subset")
    ap.add_argument("--corpora", nargs="+", required=True)
    ap.add_argument("--n-target", type=int, default=1_000_000)
    ap.add_argument("--out", default=str(ROOT / "corpora"))
    ap.add_argument("--work", default=str(ROOT / "work"))
    a = ap.parse_args()
    for name in a.corpora:
        src = ALIASES.get(name, name) if name not in CORPORA else name
        if src not in CORPORA:
            raise SystemExit(f"unknown corpus: {name} (have {sorted(CORPORA)})")
        build(src, a.n_target, Path(a.out), Path(a.work), out_name=name)


if __name__ == "__main__":
    main()

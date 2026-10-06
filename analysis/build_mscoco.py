"""MS COCO captions as a retrieval collection, in this repository's format.

The other four collections come from BEIR. MS COCO does not, so the task is
constructed here, and it is constructed the way Quora already is: the corpus and
the queries are drawn from one pool of short texts, and a query's relevant
documents are the ones describing the same thing. For Quora that is a duplicate
question; here it is another caption of the same image. Five people wrote five
captions of one photograph independently, so the positives are paraphrases that
share no vocabulary by construction -- which is the property that makes this a
different distribution from the other four rather than a fifth sample of them.

    corpus   every caption except the 5,000 chosen as queries
    query    the first caption of each image in the Karpathy test split
    relevant the same image's remaining captions

Captions are deduplicated by exact string, keeping the first occurrence: two
identical strings are one document, and no ranking can separate them. A query
whose positives all vanish that way is dropped.

Source: yerevann/coco-karpathy (the Karpathy split of MS COCO 2014, captions
only -- no images are downloaded, and none are needed).
"""
import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CORPORA                                       # noqa: E402

URL = ("https://huggingface.co/datasets/yerevann/coco-karpathy/"
       "resolve/main/coco.parquet")
NAME = "mscoco"


def load(cache):
    if cache.exists():
        return pq.read_table(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}", flush=True)
    b = urllib.request.urlopen(URL).read()
    cache.write_bytes(b)
    return pq.read_table(io.BytesIO(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/dev/shm/coco.parquet")
    ap.add_argument("--query-split", default="test",
                    help="the Karpathy split whose images provide the queries")
    a = ap.parse_args()

    t = load(Path(a.cache))
    sents = t.column("sentences").to_pylist()
    split = t.column("split").to_pylist()
    cocoid = t.column("cocoid").to_pylist()

    # one query per image of the chosen split: its first caption
    qrows = [i for i, s in enumerate(split) if s == a.query_split]
    qtext = {i: sents[i][0].strip() for i in qrows}

    docs, row_of, src = [], {}, []
    for i, caps in enumerate(sents):
        for j, c in enumerate(caps):
            c = c.strip()
            if not c or (i in qtext and j == 0):
                continue                      # the query caption is not a document
            if c in row_of:
                continue                      # identical strings are one document
            row_of[c] = len(docs)
            docs.append(c)
            src.append(i)

    queries = []
    for i in qrows:
        rel = {}
        for c in sents[i][1:]:
            c = c.strip()
            r = row_of.get(c)
            if r is not None:
                rel[str(r)] = 1
        if not rel:
            continue
        queries.append({"qid": str(cocoid[i]), "text": qtext[i], "rel": rel})

    CORPORA.mkdir(parents=True, exist_ok=True)
    (CORPORA / f"{NAME}_docs.txt").write_text(
        "\n".join(d.replace("\n", " ") for d in docs) + "\n", encoding="utf-8")
    with open(CORPORA / f"{NAME}_queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    npos = np.array([len(q["rel"]) for q in queries])
    total = sum(len(s) for s in sents)
    man = {
        "corpus": NAME, "repo": "yerevann/coco-karpathy",
        "task": "caption-to-caption: a query's positives are the other captions "
                "of the same image",
        "eval_split": a.query_split, "relevance_levels": [1],
        "graded_relevance": False,
        "n_images": t.num_rows, "n_docs_original": int(total),
        "n_docs_subset": len(docs),
        "n_captions_dropped_as_duplicates": int(total - len(docs) - len(qtext)),
        "n_eval_queries": len(queries),
        "positives_per_query_mean": float(npos.mean()),
        "positives_per_query_median": float(np.median(npos)),
        "note": "no images are used; captions only",
    }
    (CORPORA / f"{NAME}_manifest.json").write_text(json.dumps(man, indent=2))
    np.savez_compressed(CORPORA / f"subset_{NAME}.npz",
                        doc_image=np.array(src, np.int32))
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()

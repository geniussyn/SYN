"""MS MARCO with the queries translated: one axis moved, everything else held.

The cross-lingual case is the cleanest out-of-distribution setting this encoder
can be put in. bge-m3 is multilingual, so a Hindi query and an English passage
live in one space by construction -- but the query side has moved off the
distribution the documents were written in, and the graph is built from
document-document proximity alone. That is exactly the condition the paper's
claim is about.

mMARCO is MS MARCO translated. Its dev.small split is the same 6,980 queries this
repository already evaluates, under the same ids, so:

    documents   unchanged -- the same 1M English passages, the same embeddings
    relevance   unchanged -- the same qrels, by query id
    queries     the only thing that differs

Each language becomes a collection `msmarco-<lang>` whose document files are
symlinks to the English ones. Nothing downstream needs to know.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CORPORA, ROOT                                 # noqa: E402

BASE = ("https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/"
        "data/google/queries/dev/")
SRC = "msmarco-1m"
from paths import EMB                                       # noqa: E402
# two neighbours of English, two distant scripts, two more families
LANGS = {"de": "german", "fr": "french", "ru": "russian", "zh": "chinese",
         "ja": "japanese", "ar": "arabic", "hi": "hindi", "id": "indonesian"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--cache", default="/dev/shm/mmarco")
    a = ap.parse_args()

    src = [json.loads(l) for l in
           (CORPORA / f"{SRC}_queries.jsonl").read_text(encoding="utf-8").splitlines()
           if l.strip()]
    rel = {q["qid"]: q["rel"] for q in src}
    order = [q["qid"] for q in src]
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)

    for code in a.langs.split(","):
        name = LANGS[code]
        f = cache / f"{name}.tsv"
        if not f.exists():
            f.write_bytes(urllib.request.urlopen(
                BASE + f"{name}_queries.dev.small.tsv").read())
        text = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            qid, t = line.split("\t", 1)
            text[qid] = t.strip()
        missing = [q for q in order if q not in text]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} queries have no translation")

        corpus = f"{SRC.split('-')[0]}-{code}"
        with open(CORPORA / f"{corpus}_queries.jsonl", "w", encoding="utf-8") as g:
            for qid in order:                      # same order, same ids, same rel
                g.write(json.dumps({"qid": qid, "text": text[qid],
                                    "rel": rel[qid]}, ensure_ascii=False) + "\n")
        for suffix in ("_docs.txt", "_manifest.json"):
            t = CORPORA / f"{corpus}{suffix}"
            if not t.exists():
                t.symlink_to(CORPORA / f"{SRC}{suffix}")
        for arm in ("vanilla", "syn"):
            for suffix in ("_docs.npy",):
                s_, t = EMB / f"{SRC}__{arm}{suffix}", EMB / f"{corpus}__{arm}{suffix}"
                if s_.exists() and not t.exists():
                    t.symlink_to(s_)
        print(f"  {corpus}: {len(order):,} {name} queries, documents shared "
              f"with {SRC}", flush=True)


if __name__ == "__main__":
    main()

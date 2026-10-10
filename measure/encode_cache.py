"""Encode a corpus (docs + eval queries) with an encoder, optionally LoRA-merged.

Writes emb/<corpus>__<arm>_docs.npy (fp16, unit-normalised) and _queries.npy.
Reuses syn.encoders verbatim so pooling/prefix match the shipped intervention.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

# The training package, which this imports the encoder registry from. It sits
# beside the repository as src/syn/ and in the training tree at its root; both
# are found without an absolute path, and SYN_TRAIN overrides for a third
# layout. It used to be a fixed path on the machine this was written on, which
# resolved nowhere else and named that machine's directory layout.
_here = Path(__file__).resolve().parent
for _c in (os.environ.get("SYN_TRAIN"), _here.parent / "src", _here.parent):
    if _c and (Path(_c) / "syn" / "__init__.py").exists():
        sys.path.insert(0, str(_c))
        break
sys.path.insert(0, str(Path(__file__).resolve().parent))
from syn.encoders import REGISTRY, load_encoder, spec_record, _pool, _tokenize  # noqa: E402
from paths import ROOT, RESULTS  # noqa: E402

LAB = ROOT

from paths import EMB                                       # noqa: E402
CORPORA = LAB / "corpora"

def load_docs(corpus):
    t = (CORPORA / f"{corpus}_docs.txt").read_text(encoding="utf-8").split("\n")
    if t and not t[-1]: t.pop()
    man = json.loads((CORPORA / f"{corpus}_manifest.json").read_text())
    assert len(t) == man["n_docs_subset"], (len(t), man["n_docs_subset"])
    return t, man

def load_queries(corpus):
    qs = [json.loads(l) for l in
          (CORPORA / f"{corpus}_queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [q["qid"] for q in qs], [q["text"] for q in qs], \
           [{int(k): int(v) for k, v in q["rel"].items()} for q in qs]

@torch.no_grad()
def encode_to(mdl, tok, spec, texts, is_query, out_path, bs, max_len, dim):
    n = len(texts)
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, dim))
    prefix = spec.query_prefix if is_query else spec.doc_prefix
    n_trunc = 0; t0 = time.time()
    for bi, s in enumerate(range(0, n, bs)):
        idx = order[s:min(s + bs, n)]
        chunk = [prefix + texts[i] for i in idx] if prefix else [texts[i] for i in idx]
        enc = _tokenize(tok, chunk, max_len, spec).to("cuda")
        n_trunc += int((enc["attention_mask"].sum(1) >= max_len).sum())
        import contextlib
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if getattr(spec, "autocast", True) else contextlib.nullcontext())
        with ctx:
            v = _pool(mdl(**enc), enc["attention_mask"], spec.pooling)
            if spec.mrl: v = v[:, :dim]
            v = torch.nn.functional.normalize(v.float(), dim=-1)
        out[idx] = v.half().cpu().numpy()
        if bi % 200 == 0:
            done = min(s + bs, n); el = time.time() - t0
            print(f"    {done:,}/{n:,}  {done/max(el,1e-9):.0f} doc/s  eta {(n-done)/max(done/max(el,1e-9),1e-9)/60:.0f}m", flush=True)
    out.flush()
    # A silently NaN matrix costs everything downstream -- index, GT, cells --
    # and nothing complains until a metric comes back exactly zero.
    for s0 in range(0, n, 200_000):
        blk = np.asarray(out[s0:s0 + 200_000])
        if not np.isfinite(blk).all():
            del out
            Path(out_path).unlink(missing_ok=True)
            raise SystemExit(f"non-finite values in rows {s0}..{s0+len(blk)}; "
                             f"refusing to write {out_path}")
    del out
    el = time.time() - t0
    print(f"    encoded {n:,} in {el:.0f}s ({n/max(el,1e-9):.0f}/s)", flush=True)
    return n_trunc / max(n, 1), el

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--encoder", default="bge-m3")
    ap.add_argument("--arm", required=True, help="tag for the output files, e.g. vanilla / syn")
    ap.add_argument("--adapter", default=None, help="PEFT LoRA dir to merge before encoding")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--docs-only", action="store_true")
    ap.add_argument("--queries-only", action="store_true")
    a = ap.parse_args()

    EMB.mkdir(parents=True, exist_ok=True)
    spec = REGISTRY[a.encoder]
    max_len = a.max_len or (256 if a.corpus.startswith("msmarco") else spec.max_len)
    tok, mdl, spec = load_encoder(a.encoder, dtype=torch.float16,
                                  allow_remote_code=spec.trust_remote_code)
    if a.adapter:
        from peft import PeftModel
        mdl = PeftModel.from_pretrained(mdl, a.adapter).merge_and_unload()
        print(f"  merged adapter {a.adapter}", flush=True)
    mdl.eval()

    base = EMB / f"{a.corpus}__{a.arm}"
    rec = {"corpus": a.corpus, "encoder": a.encoder, "arm": a.arm,
           "adapter": a.adapter, "max_len": max_len, **spec_record(spec, max_len)}

    if not a.queries_only:
        docs, man = load_docs(a.corpus)
        print(f"[{a.corpus}/{a.arm}] docs {len(docs):,} max_len={max_len}", flush=True)
        # Encoding the corpus is part of what a training-side remedy costs and an
        # index-side one does not. It is a one-off, but it is not free, and the
        # cost table has to carry it next to build time.
        rec["truncation_rate_docs"], rec["encode_docs_s"] = encode_to(
            mdl, tok, spec, docs, False, str(base) + "_docs.npy", a.bs, max_len, spec.dim)
        rec["n_docs"] = len(docs); del docs
    if not a.docs_only:
        qid, qtext, _ = load_queries(a.corpus)
        print(f"[{a.corpus}/{a.arm}] queries {len(qtext):,}", flush=True)
        rec["truncation_rate_queries"], rec["encode_queries_s"] = encode_to(
            mdl, tok, spec, qtext, True, str(base) + "_queries.npy", a.bs, max_len, spec.dim)
        rec["n_queries"] = len(qtext)
    old = json.loads((Path(str(base) + "_manifest.json")).read_text()) if Path(str(base) + "_manifest.json").exists() else {}
    rec["encode_total_s"] = (rec.get("encode_docs_s", 0.0)
                             + rec.get("encode_queries_s", 0.0))
    # Encoding is GPU-bound but tokenisation and the host copy are not, so an
    # encode that ran beside index builds is not a clean cost number. Record what
    # else the machine was doing rather than discovering later that it mattered.
    try:
        load1 = os.getloadavg()[0]
        busy = sorted({l.split()[-1] for l in subprocess.run(
            ["ps", "-eo", "comm"], capture_output=True, text=True).stdout.splitlines()[1:]}
            & {"hnsw_bench", "run_cell.py", "python3"})
        rec["encode_load1"] = round(load1, 1)
        rec["encode_exclusive"] = load1 < 8.0
    except Exception:
        pass
    old.update(rec)
    Path(str(base) + "_manifest.json").write_text(json.dumps(old, indent=2))
    print(f"  wrote {base}_*.npy", flush=True)

if __name__ == "__main__":
    main()

# The embeddings

Everything in this repository runs on four document matrices and their four
query matrices. They are too large for git, so they are hosted separately and
fetched into `emb/`.

| file | shape | dtype | size |
|---|---|---|---|
| `msmarco-1m__vanilla_docs.npy` | (1 000 000, 1024) | float16 | 1.91 GiB |
| `msmarco-1m__vanilla_queries.npy` | (6 980, 1024) | float16 | 13.6 MiB |
| `msmarco-1m__syn_docs.npy` | (1 000 000, 1024) | float16 | 1.91 GiB |
| `msmarco-1m__syn_queries.npy` | (6 980, 1024) | float16 | 13.6 MiB |
| `fever__vanilla_docs.npy` | (1 000 000, 1024) | float16 | 1.91 GiB |
| `fever__vanilla_queries.npy` | (6 666, 1024) | float16 | 13.0 MiB |
| `fever__syn_docs.npy` | (1 000 000, 1024) | float16 | 1.91 GiB |
| `fever__syn_queries.npy` | (6 666, 1024) | float16 | 13.0 MiB |

7.68 GiB in total, plus four small `_manifest.json` that record the encoder,
the pooling, the normalisation and the corpus snapshot each matrix was made
from. `vanilla` is the stock BGE-M3 encoder; `syn` is the same encoder with the
LoRA adapter in `adapters/` merged in. Row *i* of a `_docs.npy` is line *i* of
the matching `corpora/<corpus>_docs.txt`, and row *i* of a `_queries.npy` is
line *i* of `<corpus>_queries.jsonl`.

float16 is what was measured, not a convenience for the download. Every number
in the paper comes from these exact bytes; see the note at the bottom.

## Getting them

```sh
./scripts/fetch_embeddings.sh          # into emb/
./scripts/fetch_embeddings.sh /data/emb && export SYN_EMB_DIR=/data/emb
```

The script reads `scripts/embeddings.manifest`, fetches each file and checks it
against the SHA-256 recorded there. A file already present and already matching
is skipped, so an interrupted fetch resumes by re-running.

### If the fetch fails

Large files on this host are served through a confirmation page and are subject
to a per-file download quota. Two failures are expected and neither is a bug in
this repository:

* **`Download quota exceeded`** — the file has been fetched too many times
  recently. It clears on its own, usually within a day. There is no way to
  raise it from the client side.
* **a 3 KB HTML file where a 1.9 GiB matrix should be** — the confirmation page
  was saved instead of the payload. The script detects this and refuses rather
  than leaving a corrupt `.npy` in place.

In both cases open the folder in a browser, download by hand, and verify:

```sh
sha256sum -c scripts/embeddings.manifest
```

## Regenerating them instead

The embeddings are the one large input that is purely derived. Nothing is lost
by rebuilding them, and it is the better path if the download is blocked:

```sh
python3 -u analysis/encode_cache.py --corpus msmarco-1m --arm vanilla \
    --encoder bge-m3
python3 -u analysis/encode_cache.py --corpus msmarco-1m --arm syn \
    --encoder bge-m3 --adapter adapters/syn
```

`--adapter` loads the LoRA directory with PEFT and calls `merge_and_unload()`
before anything is encoded, so the encoder that runs has the adapter folded
into its weights and costs exactly what the stock one costs. Omit the flag for
the baseline. Substitute `fever` for the other collection, and `adapters/nopull`
or `adapters/syn-plain` for the ablation arms.

One GPU, roughly 7.4 minutes per collection per arm on an A100-80G at batch 256.
The adapter is 9.5 MB and is in this repository, so the `syn` arm needs no
download at all.

A rebuild is bitwise-identical only on the same GPU, driver and library stack.
Elsewhere expect the last bits of some float16 values to differ, which moves
mutual ranks among exactly-tied neighbours and can shift a reported recall in
the fourth decimal. `verify.sh` checks the shapes and the manifests, not the
bytes, for this reason. If you need the paper's numbers to the digit, fetch
rather than rebuild.

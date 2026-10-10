# SYN

Taming Embedding Models: The Geometry Behind Hard Queries in ANNS

This repository is what derives the paper's results: the training procedure, the
LoRA adapters it produced, the measurement code the numbers were read with, and
the lanes that drove it. It does not ship the measurements themselves, the
figures drawn from them, or the data those figures were drawn with. A figure is
a rendering of a result and a result is a rendering of a measurement; what is
here is the thing underneath both, so a reader reruns rather than redraws.

```sh
./verify.sh                            # ten structural checks
./experiments/fetch_embeddings.sh      # the eight matrices this runs on
```

| | |
|---|---|
| `src/syn/` | the training procedure — corpus, mining, the pull objective, training |
| `measure/` | what reads a number: the index harness, the exact top-k, the φ_h reader |
| `experiments/` | the lanes that drive `measure/`, one per experiment in the paper |
| `adapters/` | the seven LoRA adapters training produced; `syn` is the paper's |
| `corpora/` | each collection's manifest — the snapshot and the seed, not the text |
| `studies/` | the three side measurements whose evidence no code here can recompute |
| `native/` | hnswlib, vendored unmodified |
| `patches/` | diffs against other projects, applied to their own source |

Start with **[ARTIFACT.md](docs/ARTIFACT.md)**. The embeddings are 7.68 GiB and
are hosted separately — **[DATA.md](docs/DATA.md)** says where, how to check
them, and how to rebuild them instead from `corpora/` and `adapters/syn`.

Anonymized for double-blind review; see [ANONYMITY.md](docs/ANONYMITY.md).
Licensed under MIT, except `native/third_party/` which carries its own.

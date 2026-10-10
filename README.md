# SYN_Taming-Embedding-Models

Taming Embedding Models: The Geometry Behind Hard Queries in ANNS

This repository is the artifact for the submission: the code, the measurements,
the figures they draw and the lane scripts that produced them. It is assembled
from the lab tree by `collect.sh`, which ships only what the paper uses, and
checked by `verify.sh`.

Start with **[ARTIFACT.md](docs/ARTIFACT.md)** — what is here, how the three parts
(`src/syn/`, `analysis/`, `patches/`) relate, and how to regenerate a figure.

The measurements need the embeddings, which are hosted separately because they
are 7.68 GiB. See **[DATA.md](docs/DATA.md)**; they can also be regenerated from the
corpora and the LoRA adapter in `adapters/`, which are both in this repository.

```sh
./scripts/fetch_embeddings.sh     # the eight matrices, into emb/
./verify.sh                       # structural checks over the assembled artifact
```

Anonymized for double-blind review; see [ANONYMITY.md](docs/ANONYMITY.md).
Licensed under MIT, except `native/third_party/` which carries its own.

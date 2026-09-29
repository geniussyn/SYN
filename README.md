# SYN_Taming-Embedding-Models

Taming Embedding Models: The Geometry Behind Hard Queries in ANNS

This repository is the artifact for the submission: the code, the measurements,
the figures they draw and the lane scripts that produced them. It is assembled
from the lab tree by `collect.sh`, which ships only what the paper uses, and
checked by `verify.sh`.

Start with **[ARTIFACT.md](ARTIFACT.md)** — what is here, how the three parts
(`src/syn/`, `analysis/`, `patches/`) relate, and how to regenerate a figure.

```sh
./verify.sh      # eleven structural checks over the assembled artifact
```

Anonymized for double-blind review; see [ANONYMITY.md](ANONYMITY.md).
Licensed under MIT, except `native/third_party/` which carries its own.

# Table: fragmentation across data types and embedding models (§3.3)

`table_coverage.tex` — replaces the nine-row Table 2 with four rows.

## Which four, and why

The section's claim is that fragmentation is not specific to one data type or one
embedding model. These four span that claim with the fewest rows:

| row | what it covers |
|---|---|
| **SIFT** (L₂, 128, classical descriptor) | the only L₂ row; the **weakest** case (φ₃ 1.11, Pr 0.902) |
| **GloVe** (angular, 100, word embedding) | the **strongest** case (φ₃ 5.24, Pr 0.031); the corpus the rest of the work uses |
| **DEEP-image** (angular, 96, 9.99M) | the only point above 1M, and an image CNN |
| **OpenAI 1536** (angular, learned text) | the paper's own domain; matches the ada-002 corpus in Figure 3 |

SIFT and GloVe bracket the full observed range of both φ₃ (1.11–5.24) and
Pr[φ₃=1] on the hard decile (0.031–0.902). Metric, dimension (96–1536), data type
and scale (1M, 10M) are all covered.

**GIST is dropped, not SIFT.** SIFT is the more conservative classical row — keeping
only GIST would overstate the effect on classical descriptors.

**The `Pr[φ₃=1] all` column is dropped.** It mixes both deciles and only dilutes the
hard/easy contrast the table exists to show.

**Dropped entirely: GIST, word2vec, OpenAI-3072, Cohere, Wikipedia.** Not moved
to an appendix. The four kept rows already bracket the observed range, and a
second table of corpora the paper makes no claim about is surface for a
reviewer to work through, not evidence.

## Caption already carries

- every easy decile sits at φ₃ ≈ 1.00 — a three-hop detour rejoins an easy query's
  exact top-10 into one component, and does not rejoin a hard one;
- OpenAI-3072 is indistinguishable from OpenAI-1536 (φ₃ 1.64 vs 1.65). Since that
  row is dropped, the caption should either carry this number or drop the claim.

## Provenance

The numbers are transcribed from the existing nine-corpus campaign, not re-measured
here. `phih.cpp` / `dump_graph.py` in the code repo implement the same definition, but
the indexes and ground truth for GIST, word2vec, DEEP, Cohere and Wikipedia are not in
that repo — only GloVe, MSMARCO ada-002 and YouTube-1M are. Re-verify before
camera-ready if that matters.

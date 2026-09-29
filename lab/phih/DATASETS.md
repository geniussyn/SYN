# Where the corpora come from

The phi_h comparison and the coverage table use public ANN benchmark corpora
rather than the BEIR collections the rest of the paper measures. That is
deliberate: the question in §3.3 is whether fragmentation is a property of one
embedding family or of embeddings, so the corpora are chosen to differ from each
other and from §5 — classical descriptors, word vectors, an image CNN, and two
learned text embeddings from different providers.

None of these are the paper's MS MARCO 1M. In particular the row labelled
`MS MARCO / ada-002` is the ann-benchmarks MS MARCO v1 embedded with OpenAI
text-embedding-ada-002, not the BEIR subset under BGE-M3 used in §5.

| corpus | n | dim | metric | file | where it comes from |
|---|---|---|---|---|---|
| SIFT1M | 1,000,000 | 128 | L2 | `sift/sift_base.fvecs` | TEXMEX / ann-benchmarks; Jégou et al. |
| GIST1M | 1,000,000 | 960 | L2 | `gist/gist_base.fvecs` | TEXMEX / ann-benchmarks |
| GloVe | 1,183,514 | 100 | angular | `glove-100-angular.hdf5` | ann-benchmarks; Pennington et al. |
| MS MARCO ada-002 | 1,000,000 | 1536 | IP | `msmarco-v1-openai-ada2-1M-ip.hdf5` | ann-benchmarks; MS MARCO v1 passages under OpenAI text-embedding-ada-002 |
| YouTube-1M | 1,000,000 | 1024 | angular | `youtube1m_base.npy` | the first 1M rows of `youtube-15M-angular.hdf5`, the same file `tailcensor/` uses at full size |
| DEEP-image | 9,990,000 | 96 | angular | ann-benchmarks DEEP1B subset | Babenko and Lempitsky |
| OpenAI (dbpedia) | 990,000 | 1536 | angular | `dbpedia-entities-openai-1M` | Hugging Face, commit af9b8869cc2d8debbd254d77737865bb09a2067f |
| Landmark | — | 768 | angular | `landmark-nomic-768-angular.hdf5` | measured, not reported |

`PHIH_DATA` is the directory holding these; `scripts/common.py` names each file
and has no default, so a wrong path refuses rather than silently reading the
wrong corpus.

## YouTube-1M

`youtube1m_base.npy` and `youtube1m_query.npy` are written into `PHIH_WORK` by
the build step, not shipped: they are the first 1M base rows and the 1,000
queries of `youtube-15M-angular.hdf5`, unit-normalised. The 15M file itself is
what `tailcensor/` measures, so the two experiments are on the same embedding at
two scales. It is the one corpus here without a published citation of its own;
cite it as the ann-benchmarks YouTube embedding and say it was subsampled.

## What the paper should say

> The measurements in §3.3 and Table 2 use public ANN benchmark corpora rather
> than the collections of §5. This is intentional: the claim is that
> fragmentation is not specific to one embedding family, and these corpora span
> two metrics, 96 to 1536 dimensions, classical descriptors, word vectors, an
> image CNN and two learned text embeddings. The row labelled MS MARCO ada-002
> is the ann-benchmarks MS MARCO v1 under OpenAI text-embedding-ada-002 and is
> not the BEIR subset under BGE-M3 measured in §5.

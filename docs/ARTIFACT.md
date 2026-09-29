# SYN — artifact and reproduction

Everything the paper's numbers and figures come from, and what it takes to
regenerate them. The artifact is 52 MB. It holds the measurements, not the data
they were taken on.

Three parts, kept separate because they are built and run at different times:

| | what it is | what ships |
|---|---|---|
| `src/syn/`, `tests/` | training: mining, the boundary pull, the query-radial objective | source |
| `analysis/`, `results/`, `figures/` | measurement: index builds, timed search, every figure | source and results |
| `patches/` | the changes made to two baselines | diffs only |

## 1. What is here and what is not

The evidence ships:

```
results/*.json             every measurement: the parameter that was executed,
                           the recall it returned, the milliseconds it cost
figures/                   the paper's figures, as PDF and PNG
analysis/, src/syn/        the scripts that produced both
scripts/                   the lanes that ran them, in the order they ran
native/                    the timing harnesses, with hnswlib vendored beside
corpora/*_manifest.json    the seed and BEIR snapshot each collection was cut
                           from
studies/                   three side measurements with their own data:
                           tail-censoring, the phi_h hardness comparison, and
                           the annotation behind the 85% claim
```

The inputs do not, because they are large and follow from the above:

```
corpora/      1.4 GB    rebuilt by `python -m syn.corpus`
emb/           25 GB    rebuilt by `analysis/encode_cache.py`
graph_*.npz   1.2 GB    dumped level-0 adjacency. Its only product is phi,
                        which is in the cell JSON already, and it cannot be
                        reproduced bit for bit anyway
gt_*.npz                exact top-10, rebuilt by `analysis/gt.py`
```

Rebuilding the corpora is exact. Each is a seeded sample (`subset_seed`
20260809) of a pinned BEIR snapshot and the manifest carries both. Rebuilding
the embeddings is not exact across GPUs and driver versions. Everything
downstream of the embeddings is deterministic given them.

### What the paper does not use is not here

We measured more than we report. A fifth collection (MS COCO, the 617k caption
corpus) and six more mMARCO languages were measured and left out, and 63 scripts
from earlier campaigns no longer have data to run on. None of that ships. A
reviewer should not have to work out which files are evidence and which are
residue, so `collect.sh` copies only what the nine figures and three tables
actually read, and `verify.sh` fails if anything else gets in.

The two lists `collect.sh` works from were computed, not remembered. The module
list is the transitive import closure of the entry points. The result-file list
is what a Python audit hook recorded the figures and tables opening when they
ran.

## 2. What is not reproducible, and why

Stated here and again at each place it matters.

**Graph construction depends on the thread count.** hnswlib inserts in parallel,
so the level-0 graph is a function of how many threads built it.
`SYN_BUILD_THREADS` is fixed at 100 for every arm so the arms are comparable,
but a machine with a different count builds a different graph. Per-query recall
moves by up to 0.124 between two builds of the same index; the hard-decile
population statistic moves by sd 0.0108. Only population statistics are quoted
anywhere in the paper.

**Milliseconds are a property of this machine.** Search is timed
single-threaded on one reserved core, under a machine-wide lock, through the
native harness in `native/`. Absolute times do not transfer. Ratios do. Every
timed result carries `timing_valid` in its JSON, and 88 of the 121 shipped JSON
files carry one.

**Seven measurements were withdrawn and re-taken.** Until 2026-09-22 the lock
covered searches but not builds, so a 100-thread build in one lane could run
through a pinned timed search in another. The signature is a larger search
budget that came back faster, and `analysis/audit_timing.py` finds it. The seven
files it flagged are in `results/superseded/` with the inversion that
condemned each and a note explaining the cause. They were re-measured, not
repaired. The audit now reports zero over every shipped file at a 3% tolerance
as well as the default 10%.

## 3. Determinism

Every source of randomness is seeded. There is no unseeded `default_rng()` in
`analysis/` or `syn/`, and `verify.sh` checks that.

| what | seed | where |
|---|---|---|
| corpus subset | 20260809 | `syn/corpus.py`, recorded in each manifest |
| training queries, regions, positives | 20260808 / 20260809 | `syn/mine.py`, `syn/train.py` |
| the selector index behind every stratum | 777 | `run_cell.hard_decile`, never reused by a measured cell |
| every measured index | 100 | `index_hnsw.build` |
| the NGFix and DARTH query split | 20260924 | `ngfix_prep.py` |

Figures name their examples by query id, not by matching text. If a corpus
subset does not contain one, `fig_mutual_rank.py` falls back on the rule the
example was chosen under and says so. No figure samples.

## 4. Running it

```bash
pip install -r requirements.txt

# hnswlib is vendored (header-only, MIT). faiss is not: the pip wheel bundles
# the library but ships no headers, so point at a source tree of the same version
git clone --depth 1 --branch v1.14.0 https://github.com/facebookresearch/faiss /tmp/faiss
make -C native FAISS_INC=/tmp/faiss

export SYN_ROOT=$PWD SYN_EXCLUSIVE=1 SYN_PIN_CORE=127
```

`native/Makefile` finds the wheel's bundled library itself and links it with
`--disable-new-dtags`. That flag is not optional: the default RUNPATH is not
searched for a dependency's own dependencies, and libfaiss needs the BLAS
bundled beside it.

`SYN_EXCLUSIVE=1` is required for any reported millisecond. It pins the search
to one core, takes the machine-wide lock, and runs one thread.

**Run one stage at a time.** The lock serialises timed regions, but a second
lane's untimed arithmetic still competes for memory bandwidth. That is what cost
the seven withdrawn files.

### 4.1 Corpora and the baseline arm

```bash
python -m syn.corpus --corpora msmarco-1m fever nq quora --n-target 1000000
python analysis/build_mmarco.py --langs fr ar      # same documents, translated queries
python analysis/encode_cache.py --corpus C --arm vanilla --encoder bge-m3
python analysis/gt.py --corpus C --arm vanilla -k 10
```

About 40 minutes per collection on one RTX PRO 6000.

### 4.2 The adapter, trained once

```bash
python -m syn.mine  --corpus msmarco-1m --train-split train \
    --require-gold-in-subset --n-train 500 --k 10 --n-clusters 2048
python -m syn.pull  --corpus msmarco-1m --pull-l 1100 --pull-low 1000 --pos-tau 0.02
python -m syn.train --corpus msmarco-1m --tag syn \
    --steps 500 --lambda-pull 1.0 --pos-select softmax
```

About 50 minutes, of which k-means over 1M x 1024 into 2048 regions is 18.
**Leave `--kmeans` at its default.** The GPU path produces different regions,
and with them different negatives and a different adapter.

The adapter is trained once, on 500 English MS MARCO queries, and applied
unchanged everywhere else, including to the French and Arabic query sets:

```bash
python analysis/encode_cache.py --corpus C --arm syn --adapter <adapter_syn>
python analysis/gt.py --corpus C --arm syn -k 10
python analysis/own_decile.py --corpus C --arm ARM
```

### 4.3 Measurements

| stage | script | wall clock |
|---|---|---|
| headline cells | `run_cell.py` | ~1.5 h per cell |
| index families (HNSW, NSG, IVF, Vamana) | `run_family.py` | ~45 min per run |
| difficulty strata (all, worst 20/10/5/1 %) | `strata_cell.py` | 1.5–8 h per arm |
| IVF cell spread | `ivf_spread.py` | ~20 min, no search |
| out-of-distribution drift | `oodness.py` | ~5 min, no search |
| gold recall at depth | `goldk_own.py` | ~20 min |
| end to end | `rerank_pipeline.py`, `rerank_strata.py` | ~30 min per collection |
| NGFix and RFix | `ngfix_prep.py`, `run_ngfix.py` | ~25 min per arm |
| NGFix budget sweep | `ngfix_pool.sh`, `ngfix_pool_table.py` | ~4 h |
| DARTH and ada-ef | `darth_prep.py`, `run_darth.py` | ~3 h per arm |

`scripts/*.sh` runs these in order, one lane at a time. Every stage guards on
its own output, so re-running one that already finished is a no-op and an
interrupted stage resumes by being run again.

Two things to know before starting a long run. DARTH's per-arm base vector file
is 4 GiB and the lanes delete it between collections, so re-running a search
stage on a finished collection needs `darth_prep.py --only-base` first. And the
lanes write their scratch to `/dev/shm`; DARTH's feature dumps are 58 GB and
should be removed once its predictors are fitted.

### 4.4 Figures

Every figure regenerates from `results/` alone. No index, no GPU, seconds
each. The exception is `fig_mutual_rank.py`, which ranks against the whole
corpus and needs the document embeddings.

```bash
for f in analysis/fig*.py; do python "$f"; done
python analysis/audit_timing.py        # must report zero flagged files
```

| paper figure | script |
|---|---|
| index families, per collection and population | `fig_families.py` |
| mutual rank, three panels | `fig_mutual_rank.py` |
| fragmentation and recall distributions | `fig_dist.py` |
| difficulty strata, including cross-lingual | `fig_strata.py` |
| cross-lingual, both populations | `fig_crosslingual.py` |
| IVF cell spread and nprobe* | `fig_ivf_spread.py` |
| degree and construction budget | `fig_sweep.py` |
| composition with NGFix and DARTH | `fig_ngfix.py` |
| end to end, three metrics by stratum | `fig_rerank_strata.py` |

Tables come from `measured_speedup.py`, `darth_gain.py` and
`ngfix_pool_table.py`.

## 5. Baselines

Their code is not redistributed. Clone the commits the paper measured and apply
the diffs:

```
NGFix     9acd987   + patches/ngfix-stage-flag.patch
DARTH     0d9bafc   + patches/darth-datasets.patch
                    + patches/darth-feature-order.patch
DiskANN   tag 0.7.0, built with -DCMAKE_CXX_FLAGS="-include cstdint"
```

`ngfix-stage-flag.patch` adds one flag to the authors' own build driver so the
two stages of HNSW-NGFix\* can be measured apart; they ship only the combined
version. `darth-datasets.patch` registers this paper's collections in their
dataset table and adds a second entry per collection whose test queries are the
hard decile, because their loader knows three query types and a stratum is not
one of them. Neither changes an algorithm.

### The DARTH feature-order patch

`darth-feature-order.patch` does change what DARTH computes, so it needs its own
paragraph.

`notebooks_scripts/predictor_training.py` fits the LightGBM booster on the
features in this order:

```
step, dists, inserts, first_nn_dist, nn_dist, furthest_dist,
avg_dist, variance, percentile_25, percentile_50, percentile_75
```

The two C++ call sites that ask the booster for a prediction
(`faiss/impl/DeclarativeRecall.cpp:911` for HNSW, `:1298` for IVF) fill their
eleven doubles in a different order:

```
... nn_dist, avg_dist, furthest_dist, variance, median, percentile_25, percentile_75
```

Two transpositions, so four of the eleven columns arrive as a different feature.
On the same booster and the same rows the effect is measurable: prediction MAE
0.0152 becomes 0.0323, and correlation with true recall 0.956 becomes 0.917.

We reported this to the authors, who confirmed it on 2026-09-26: a minor
feature-ordering bug in the released code, and the trained model's order is the
one to use, in both places. The patch does that.

Both sets of numbers are in the artifact. `darth_<collection>__<arm>.json` is
the released code as published; `darth_<collection>__<arm>__fixfeat.json` is the
same index, the same logged training search and the same fitted booster, with
the corrected order reaching it. The stopping parameters are retuned in the
second set, because the first set tuned them against a scrambled predictor. The
paper reports the corrected numbers. `fig_ngfix.py --darth-tag ""` draws the
released ones.

### Query supervision

Both baselines are query-supervised, so both draw their historical queries from
the same leak-filtered training pool SYN's 500 were sampled from
(`syn/mine.py --dump-pool`), and the budget is swept over nested prefixes of it:
500, 4,000, 16,000, and the full 59,242. At 500 the baseline is holding exactly
the queries SYN was given. Where BEIR has no training split (NQ, Quora) the
evaluation queries are halved by a fixed seed instead, which favours the
baseline: it gets a per-collection query log and SYN gets none.

## 6. Verifying

```bash
bash verify.sh
```

It reads and never measures, and it runs against the artifact by default rather
than against the working tree, because the artifact is what a reviewer has.
Eight checks:

1. the numerical core tests pass
2. no unseeded randomness in `analysis/` or `syn/`
3. every result file parses and carries its protocol fields
4. the timing audit is clean at a 3% tolerance
5. the withdrawn measurements are present with the note explaining them
6. every shipped figure has the script that drew it, and there are no strays
7. nothing from a dropped collection leaked into results, corpora or figures
8. every corpus manifest pins its seed and its snapshot

## 7. Layout

```
ARTIFACT.md          this file
collect.sh           assembles the artifact from the working trees
verify.sh            the eight checks above
run.sh               the training entry point
requirements.txt
patches/             three diffs against two baselines
syn/  tests/         training source and its unit tests
analysis/            37 modules: the import closure of what the paper runs
native/             timing harnesses, Makefile, vendored hnswlib
scripts/            the lane scripts that produced the shipped results
results/         155 files, including superseded/
figures/            nine figures, PDF and PNG
corpora/         seven manifests
```

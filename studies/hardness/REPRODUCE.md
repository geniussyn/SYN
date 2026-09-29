# Reproducing the φ₁ experiments

Per-query graph fragmentation (φ_h) as a hardness measure for graph-based ANN,
compared against classical distance measures and against Steiner-hardness (VLDB'25).

Everything runs on one machine, **10 threads** (shared lab server cap).

---

## 0. What φ_h is

For a query `q`, let `GT_k(q)` be its `k` exact nearest neighbours (ties broken by
ascending document id). Let `G` be the **level-0 adjacency of the stored HNSW index** —
the graph the search actually traverses. `G` is directed and is *not* symmetric, so two
GT documents are put in the same component only if **each reaches the other within `h`
hops**. `φ_h(q)` is the number of components of `GT_k(q)` under that relation.
`R_h` is an exact `h`-step BFS and `h` counts edges, not candidates.
`φ_h ≤ k`, so values at different `k` are never placed side by side.

Everything in the paper uses `h = 1` (`φ₁`).

## 1. Layout

| file | role |
|---|---|
| `common.py` | corpus registry (paths, metric, dim) and the 10-thread cap |
| `build.py` / `build100.py` | HNSW build (hnswlib), exact GT, per-query recall |
| `dump_graph.py` | parse the **stored** index → level-0 CSR in label space |
| `phih.cpp` | φ_h for `h = 1..4`, `k ≤ 128` (OpenMP) |
| `effort.cpp` | per-query NDC and recall over an ef ladder |
| `hardness.py` | LID (MLE), RC@k, `d_2k/d_k`, `d_k` from an exact top-200 pass |
| `crosscheck.py` | independent φ_h via plain forward BFS (scipy) — validation |
| `validate.py` | independent ef-search over the extracted CSR — validation |
| `prep_steiner.py` | write our corpora + our graph in Steiner-hardness's layout |
| `mrng_rev.py` | MRNG → reverse graph in their `_reversed_std` format |
| `fig_*.py`, `plot_*.py` | figures |

Artefacts live under `$PHIH_WORK`, the corpora under `$PHIH_DATA`, and the
Steiner-hardness checkout under `$PHIH_STEINER`. None of the three has a default:
these experiments ran on a different machine from the rest of the paper, on public
ANN benchmark corpora rather than on the BEIR collections, so a guessed path would
silently read the wrong corpus.

```sh
export PHIH_WORK=/scratch/phih          # needs room for indexes and dumps
export PHIH_DATA=/scratch/ann-datasets  # glove-100-angular.hdf5, etc.
export PHIH_STEINER=/opt/steiner        # the VLDB'25 authors' checkout
```

## 2. Pipeline

```sh
# per corpus: HNSW M=16 efC=200, exact GT (ties by id), recall at a ladder of ef
python3 build.py     <corpus>          # k=10
python3 build100.py  <corpus>          # k=100
python3 run_k50.py   <corpus>          # k=50  (Steiner-hardness's own setting)

# the graph the search traverses, read out of the saved index
python3 dump_graph.py <corpus>

# phi_h  (h = 1..4 in one sweep)
g++ -O3 -march=native -fopenmp -o phih phih.cpp
./phih $WORK/<corpus> $WORK/<corpus>_gt100.i32 100 $WORK/<corpus>_phi100.i32 10

# per-query NDC over an ef ladder  -> "least NDC to reach a target recall"
g++ -O3 -march=native -fopenmp -o effort effort.cpp
python3 run_effort.py <corpus>

# distance-based measures
python3 hardness.py <corpus>
```

### Validation (run these — they are cheap and they caught two real bugs)

```sh
python3 validate.py   <corpus>   # our ef-search on the extracted CSR vs hnswlib
python3 crosscheck.py <corpus> 20 100   # phi_h vs plain forward BFS
```

`validate.py` reproduces hnswlib's recall to ≤0.004 (sift 0.9890→0.9895,
glove 0.8265→0.8230), which is what licenses calling `G` "the graph the search
traverses". `crosscheck.py` recomputes φ_h with no bidirectional decomposition and no
bitmasks and must match exactly (20/20 at k=10, 8/8 at k=100, both corpora, half the
queries drawn from the hard decile).

## 3. Steiner-hardness (VLDB'25) on the same graph

Upstream: <https://github.com/DSM-fudan/Steiner-hardness>.
`get_delta0_point` has a `-m 1` (HNSW) path, so their minimum-effort algorithm runs on
**our** level-0 graph — same graph, same queries, same `k`/`Acc`/`prob` as φ₁.

Dependencies without root:

```sh
# Eigen 3.4 (header-only) -> Steiner-hardness/src/Eigen
curl -sSL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | tar xz

# COIN-OR Cbc/Clp/Osi/Cgl/CoinUtils + OpenBLAS, extracted from .deb
apt-get download coinor-libcbc-dev coinor-libcbc3.1 coinor-libclp-dev coinor-libclp1 \
                 coinor-libosi-dev coinor-libosi1v5 coinor-libcgl-dev coinor-libcgl1 \
                 coinor-libcoinutils-dev coinor-libcoinutils3v5 libopenblas-dev
for d in *.deb; do dpkg -x "$d" root/; done

g++ -o src/get_delta0_point src/get_delta0_point.cpp -I src/ -I root/usr/include \
    -O3 -mavx2 -fopenmp -L root/usr/lib/x86_64-linux-gnu \
    -Wl,-rpath,root/usr/lib/x86_64-linux-gnu \
    -lCbcSolver -lCbc -lOsiClp -lOsi -lCoinUtils -lClp
```

Then:

```sh
python3 prep_steiner.py <corpus>    # base.fvecs, deep GT, our graph as ibin + reversed
# add the corpus to stringToMetricsMap in get_delta0_point.cpp (M and ef name the files)
./get_delta0_point -d <corpus> -b $STEINER_BASE -r 0.98 -p 0.98 -k 100 -o 0 -m 1  # δ₀
./get_delta0_point -d <corpus> -b $STEINER_BASE -r 0.98 -p 0.98 -k 100 -o 1 -m 1  # ME
```

### Three traps in their pipeline

1. **GT depth is chosen by dataset *name*.** `glove*` needs a **100,000**-deep GT for
   `purpose < 10`; unregistered names get no suffix (`{data}_groundtruth.ivecs`). This
   is not cosmetic: our measured δ₀ on GloVe reaches **52,401**, so a 50,000-deep list
   would silently truncate. On YouTube-1M/MSMARCO at depth 50,000, 14 and 3 queries
   still saturate — regenerate at 100,000 if you need those.
2. **The tool always uses `L2Space`.** Feed **normalised** vectors for cosine corpora;
   L2 order then equals cosine order.
3. **File formats.** `ibin_plain` = `int32 n, int32 maxdeg`, then `n × maxdeg` int32
   neighbour ids padded with `-1`. `_reversed_std` = per node, uint32 ids terminated by
   `0xFFFFFFFF`. `.mrng` = per node `uint32 deg` then `deg` uint32 ids.

### The MRNG route does not finish

The published measure is defined on MRNG, not HNSW. `efanna` (K=2048, L=2400, 10 iters)
took 16 min and `index_mrng` 45 s on GloVe, but the **δ₀ step on MRNG did not finish in
19.5 h** (12 h at 20 threads + 7.5 h at 10), against **12 min** for the identical step on
HNSW. The likely cause is the MRNG's hub structure: max in-degree 3,006 vs 340 for our
HNSW level-0 graph, and 100,925 nodes (8.5%) with in-degree 0 vs 0 for HNSW. All
Steiner numbers reported here are therefore **ME on our HNSW level-0 graph**, which is
in any case the correct control for a head-to-head against φ₁.

## 4. Figures

```sh
python3 plot_glove_k100.py   # fig 1 — GloVe φ₁ vs recall@100 scatter
python3 fig_corr.py          # fig 2, 3 — |ρ| with recall@k and with least NDC
python3 fig_cost.py          # fig 4 — wall-clock cost of each measure
```

`sigmod_style.py` sets the ACM `acmart` body face (Libertinus Serif 7.040, installed
into `~/.local/share/fonts`) and `pdf.fonttype = 42` so no Type-3 fonts are embedded.

## 5. Known limitations

- GIST and YouTube-1M ship only **1,000 queries**; GloVe and MSMARCO have 10,000 and
  6,980. Correlations on the former are correspondingly noisier.
- YouTube-1M is an evenly-spaced **1,000,000-point subsample** of YouTube-15M (the full
  corpus's hnswlib index is ~66 GB and the shared mount had ~70 GB free). GT is
  recomputed on the subsample.
- Five YouTube queries have ≥100 exact duplicates in the base (`d_k = 0`), which makes
  LID and RC diverge. They are excluded everywhere and the count is reported.
- The recall axis saturates: at ef=128 mean recall@50 is already 0.95 (MSMARCO) and
  0.97 (YouTube), so correlations there are compressed. This is why the ef ladder runs
  from 16, not from the deployment operating point.

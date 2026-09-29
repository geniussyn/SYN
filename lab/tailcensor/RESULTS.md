# Tail censoring: what "average recall" hides on SpaceV100M and YouTube15M

Question asked: draw the recall-QPS curve with the hardest 1% of queries removed,
with the hardest 10% removed, and with all queries, using **mean recall@10** and
**single-thread QPS**.

## Setup

| | SpaceV100M | YouTube15M (seahorse) |
|---|---|---|
| n | 100,000,000 | 15,686,309 |
| dim / metric | 100 (int8 source) / L2 | 1024 / cosine (unit-normalised, IP space) |
| queries | 10,000 of the 29,316 shipped (evenly spaced) | 1,000 |
| index | hnswlib M=16, efC=200, seed 100 | same |
| build | 54.0 min, 24 threads (independent index: 58.1 min) | 15.6 min, 24 threads |
| search | **1 thread**, pinned to core 47, ef ∈ {16 … 2048} | same |
| timing | 9 repetitions per (ef, subset), round-robin over subsets, min taken | same |

**Data note.** The `train` dataset inside `msspacev-100M-i8-euclidean.hdf5` is an
external link into another user's home directory and is unreadable; the base is
read from `spacev100m_raw/spacev100m_base.i8bin` (10 GB int8) instead. Queries and
the k=100 ground truth come from the hdf5 itself. The int8 base is held once and
converted to float32 chunk-by-chunk (a materialised float32 copy would be 40 GB on
top of hnswlib's own).

**Ground truth is audited, not assumed.**

* SpaceV — the shipped GT was re-derived from scratch for 200 sampled queries by
  brute force over all 100M points (faiss, 49 s): mean |mine ∩ shipped|@10 =
  **9.985/10**, max relative gap on d₁₀ = **0.000e+00**, and **0/200** queries where
  the shipped 10th neighbour is farther than the true 10th. So it is exact, it is
  L2, and it belongs to this base; the 0.015 is distance ties.
* YouTube — GT recomputed here in full (blocked exact scan) and checked against the
  `distances` dataset in the hdf5: max |Δd₁₀| = **6.0e-7** in the 1−cos convention.

Recall is tie-safe (an answer counts if its distance ≤ d₁₀·(1+1e-5)); plain
id-intersection recall is recorded alongside and never differs by more than 0.003.

### How "hard" is defined (and why it is not circular)

A query's hardness is its **mean recall@10 over the whole ef ladder on a second
index built with a different seed** (2026), ties broken by recall at ef=2048 then
by LID. The censored curves are then measured on the *other* index (seed 100), so
the removed queries are not selected on the noise of the curve being drawn — the
hard set is fixed before the curve is drawn and never re-picked per ef.

## Result 1 — the curves (`tail_censor.pdf`)

mean recall@10 / single-thread QPS:

| | SpaceV all | SpaceV −1% | SpaceV −10% | YT all | YT −1% | YT −10% |
|---|---|---|---|---|---|---|
| ef=16 | 0.7225 / 9572 | 0.7283 / 9568 | 0.7740 / 9731 | 0.8606 / 5225 | 0.8654 / 5152 | 0.9040 / 5290 |
| ef=64 | 0.8731 / 3752 | 0.8796 / 3827 | 0.9235 / 3936 | 0.9480 / 2235 | 0.9534 / 2239 | 0.9838 / 2289 |
| ef=128 | 0.9171 / 2109 | 0.9237 / 1996 | 0.9569 / 2085 | 0.9641 / 1350 | 0.9695 / 1347 | 0.9914 / 1381 |
| ef=512 | 0.9661 / 634 | 0.9719 / 588 | 0.9854 / 604 | 0.9791 / 413 | 0.9844 / 413 | 0.9972 / 423 |
| ef=2048 | 0.9831 / 167 | 0.9865 / 155 | 0.9924 / 156 | 0.9855 / 101 | 0.9909 / 118 | 0.9983 / 122 |

**QPS is not what moves.** At a fixed ef the three subsets are the same system on
almost the same queries; their measured QPS differs by a few percent in either
direction (cross-subset spread at a fixed ef: median 5.6% SpaceV / 2.5% YouTube),
which is the noise floor on this shared box — max/min over the 9 reps of one batch
is already median 9.2% / 12.0%. Censoring the tail moves the curve **horizontally**.
The all-query QPS is monotone in ef on both corpora.

## Result 2 — the price of the censoring (`tail_censor_gain.pdf`)

What the full query set must pay to report the recall that the censored set reports
at the same ef:

| | SpaceV −10% | YouTube −10% |
|---|---|---|
| ef=16 | ef 1.5× → 1.29× apparent speedup | ef 1.7× → 1.35× |
| ef=32 | 1.7× → 1.48× | 2.9× → 1.98× |
| ef=64 | 2.3× → 2.02× | 16.7× → **12.4×** |
| ef=128 | 2.8× → 2.54× | unreachable at any ef ≤ 2048 |
| ef=384 | 4.7× → **4.33×** | unreachable |
| ef≥512 | unreachable at any ef ≤ 2048 | unreachable |

Removing only 1% is worth 1.03–1.22× on SpaceV up to ef=256 (1.45× at ef=512), and
1.03–1.58× on YouTube up to ef=256, rising to 2.5× at ef=384.

## Result 3 — why both curves end in a wall (`cohorts.txt`)

Mean recall@10 of the fixed cohorts (ranked on the independent index):

| ef | SpaceV hardest 1% | SpaceV −10% cohort | SpaceV other 90% | YT hardest 1% | YT other 90% |
|---|---|---|---|---|---|
| 64 | 0.2350 | 0.4197 | 0.9235 | 0.4100 | 0.9838 |
| 128 | 0.2630 | 0.5588 | 0.9569 | 0.4300 | 0.9914 |
| 512 | 0.3870 | 0.7922 | 0.9854 | 0.4500 | 0.9972 |
| 2048 | 0.6440 | 0.8992 | 0.9924 | **0.4500** | 0.9983 |

The two corpora fail differently:

* **YouTube** — the hardest 1% is **flat in ef from 48 to 2048** (0.41 → 0.45) while
  the rest saturates at 0.9983, and 6 queries still return recall 0 at ef=2048
  (11 at ef=128). Budget buys that cohort nothing at all.
* **SpaceV** — the hardest 1% does climb (0.15 → 0.64) but only by spending
  the whole ladder: 128× more ef for +0.49, while the other 90% moved +0.02.
  Queries with recall 0 fall 1300 → 4 across the ladder, so at 100M the tail is
  mostly *expensive* rather than *unreachable*.

Either way, mean recall at a deployable ef is ~90% a statement about queries that
were already finished at ef=64.

## Caveats

1. SpaceV's 10,000 queries make the 1% cohort 100 queries and the 10% cohort 1,000
   — solid. YouTube ships only 1,000 queries, so its 1% cohort is 10 queries; quote
   its 10% row, not its 1% row.
2. **Build-to-build variance is large on YouTube.** Three builds of the identical
   config (hnswlib's parallel insertion is not deterministic even at a fixed seed)
   plateau at recall 0.9632 / 0.9754 / 0.9855. The censoring effect is stable across
   them (−10% gives +0.025…+0.028 at ef=192), but the absolute height of the wall is
   not a two-decimal quantity. Numbers here are from one index (seed 100, run 2);
   run 1 is kept in `out_run1/`. SpaceV was built once per seed.
3. **Shared machine.** Other users were at load 4–6 during the runs; timing used
   min-of-9 with core pinning, but one YouTube point (ef=2048, all-query subset) is
   still visibly contaminated (21.3% cross-subset spread vs 2.5% median).
4. The QPS axis is single-thread hnswlib through the Python binding, one batch call
   per measurement (no per-query Python overhead in the timed path).

## Files

```
run_corpus.py    GT + 2 builds + recall sweeps + timed sweep  (REUSE_B=1, PIN_CORE=n)
cfg.py           corpora, params, 24-thread cap, int8 float32-view
audit_spacev.py  SpaceV GT audit vs a 100M brute-force recomputation
check_gt.py      GT audit for the recomputed corpora
summarize.py     tables incl. the iso-reported-recall price
cohort.py        per-cohort recall table
plot_fig.py      figures: main | gain
out/             *_curves.json, *_perq.npz, summary.txt, cohorts.txt,
                 tail_censor{,_gain}.{pdf,png}
out_gist/        the GIST1M pass this replaced
out_run1/        YouTube first pass (unpinned timing, 5 reps) — superseded
```

Figure style as requested: greyscale only, 0.9 pt lines, markers ●/■/△ with fills
black / 0.42 grey / white and line greys 0.00 / 0.42 / 0.62.

# What goes into the paper (§3.3 + Appendix C)

φ_h is the paper's *instrument*, not its contribution, so this material is sized for
one figure slot in §3.3 and one table in the appendix.

| file | goes where | what it says |
|---|---|---|
| `fig3_hardness_measures.pdf` | §3.3, replaces/extends **Figure 3** | φ₁ vs. Steiner ME, LID, d₂ₖ/dₖ, RC@k on two axes + partials |
| `table_cost.tex` | **Appendix C**, next to Table 13/14 | what each measure costs to compute |
| `VALUES.txt` | — | every number in the figure, for the text |

Drawn at acmart `\textwidth` (7.0 in) in Libertinus with `pdf.fonttype=42`, so
**include it at natural size** — do not rescale in LaTeX.

---

## Figure 3 — caption draft

> Per-query hardness measures against two outcomes (k=100, HNSW M=16, ef_C=200,
> all queries). **(a)** Spearman |ρ| with recall at a fixed budget. **(b)** |ρ| with
> the least NDC that reaches recall 0.98. **(c)** rank-partial |ρ| with fixed-budget
> recall, each measure controlling for the other. φ₁ leads at a fixed budget; Steiner
> ME leads on the effort axis it is defined for. The two agree closely
> (Spearman 0.91–0.97), and LID carries almost nothing that φ₁ does not already carry.

Only k=100 is shown. φ_h ≤ k, so at k=10 the index has ten levels and the comparison
is resolution-limited — the paper already makes this point in §3.3 (16% of GloVe
queries sit at φ₁ = k).

## Sentences for §3.3

> φ₁ and Steiner-hardness [49] measure the same structure: their per-query rank
> correlation is 0.965 (GloVe), 0.931 (MSMARCO ada-002) and 0.914 (YouTube-1M).
> They differ in where each is sharper. At a fixed search budget φ₁ tracks recall
> better (0.984 / 0.950 / 0.935 against 0.970 / 0.931 / 0.904); on the least-NDC axis
> Steiner-hardness is defined for, it leads (0.987 / 0.980 / 0.973 against
> 0.937 / 0.909 / 0.885). Controlling for φ₁ leaves LID with |ρ| ≤ 0.10, while
> controlling for LID leaves φ₁ at 0.90–0.93.

This is what backs the claim already made in §6 ("φ_h is the coarser quantity beneath
it"), which is currently unsupported.

## Sentence for Appendix C (cost)

> Computing Steiner-hardness costs 4–59× more than φ₁ (Table N), 53–89% of it the δ₀
> calibration φ₁ has no counterpart for. φ₁ is read from the stored graph and the
> top-k ground truth alone, which is what makes the sweeps in Table 2, Table 8 and
> Figure 10 — hundreds of separate computations, each at a different index and target
> — feasible, and what the sampling in Appendix B is trading against.

**Frame the cost this way, not as a standalone claim.** φ_h runs once per corpus, so
"6,434 s vs 110 s" invites "you only run it once." What the cost actually buys is the
paper's own experimental design: δ₀ must be recalibrated per (arm, index, target)
whereas φ_h is a single value that transfers across operating points.

---

## Values

### Context
| corpus | queries used | mean recall@100 @ef=128 | median least NDC @0.98 |
|---|---|---|---|
| GloVe | 10,000 | 0.7208 | 25,476 |
| MSMARCO ada-002 | 6,980 | 0.9222 | 3,222 |
| YouTube-1M | 995 | 0.9466 | 2,212 |

### (a) |ρ| with recall@100 at ef=128
| measure | GloVe | MSMARCO | YouTube |
|---|---|---|---|
| **φ₁** | **0.984** | **0.950** | **0.935** |
| Steiner ME | 0.970 | 0.931 | 0.904 |
| d₂ₖ/dₖ | 0.939 | 0.835 | 0.833 |
| LID | 0.878 | 0.651 | 0.604 |
| RC@k | 0.870 | 0.829 | 0.665 |

### (b) |ρ| with least NDC to recall 0.98
| measure | GloVe | MSMARCO | YouTube |
|---|---|---|---|
| **Steiner ME** | **0.987** | **0.980** | **0.973** |
| φ₁ | 0.937 | 0.909 | 0.885 |
| RC@k | 0.911 | 0.865 | 0.646 |
| d₂ₖ/dₖ | 0.902 | 0.814 | 0.809 |
| LID | 0.813 | 0.602 | 0.582 |

### (c) rank-partial |ρ| with fixed-budget recall
| | GloVe | MSMARCO | YouTube |
|---|---|---|---|
| φ₁ \| LID | −0.931 | −0.911 | −0.896 |
| LID \| φ₁ | **+0.104** | −0.041 | −0.007 |
| φ₁ \| ME | −0.759 | −0.621 | −0.626 |
| ME \| φ₁ | −0.448 | −0.410 | −0.347 |

### φ₁ ↔ Steiner ME agreement
GloVe +0.965 · MSMARCO +0.931 · YouTube-1M +0.914

### Cost (single thread, k=100; full table in `table_cost.tex`)
| corpus | φ₁ | LID/RC/dₖ | Steiner ME | ×φ₁ |
|---|---|---|---|---|
| GloVe (10,000 q) | 109.5 s | 85.5 s | 6,434 s | 58.8 |
| MSMARCO (6,980 q) | 223.6 s | 203.1 s | 901.3 s | 4.0 |
| YouTube-1M (1,000 q) | 42.5 s | 47.2 s | 1,046.5 s | 24.6 |

Ranges over corpora, so quote **4–59×**, not 58.8×. The spread is δ₀: 68 ms/query on
MSMARCO, 521 on GloVe, 931 on YouTube. On high-dimensional corpora the exact ground
truth both measures need dominates, which narrows the gap.

---

## Three things to settle before submitting

**1. The corpora do not line up with the rest of the paper -- and that is the
point, by the authors' decision (2026-09-28).** They are kept as measured and
their provenance is recorded in DATASETS.md, so the section shows fragmentation
across embedding families rather than repeating §5's collections. The rest of
this note is the reasoning that was weighed before that call. Figure 3 currently uses
SIFT / GloVe / GIST / Landmark; this material is on GloVe / MSMARCO ada-002 /
YouTube-1M; the paper's collections are MS MARCO / FEVER / NQ / Quora under BGE-M3.
Only GloVe is common to any two. In particular **"MSMARCO ada-002" here is not the
paper's MS MARCO 1M** — different encoder, different corpus — and must not be
labelled as it. The clean fix is to rerun §3.3 on the paper's own four collections;
the pipeline is built and it is about half a day per collection.

**2. The k=10 panel.** The paper's current Figure 3 shows φ₁ losing to LID on all four
corpora at k=10 (0.29/0.33, 0.75/0.81, 0.59/0.61, 0.17/0.18). Adding Steiner, which
also leads on the effort axis, compounds that. Dropping the k=10 panel and stating the
resolution limit is both honest and safer; §3.3 already argues the limit.

**3. Steiner ME is computed on the HNSW level-0 graph**, via the `-m 1` path of the
authors' code — not on the MRNG the published measure is defined over. For §3 this is
the right control (the whole section is about the served graph), but the definitional
difference belongs in a footnote. The MRNG run itself is not reported.

---

## Regenerating

```sh
python3 scripts/paper_fig3.py > VALUES.txt    # figure + every number
python3 timing_all.py <corpus>        # single-threaded cost, feeds table_cost.tex
```

Full pipeline, validation and the Steiner build (Eigen + COIN-OR without root):
see `REPRODUCE.md`.

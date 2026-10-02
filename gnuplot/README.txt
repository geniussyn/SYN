Figures 1-3: data and gnuplot
=============================

The TSV is the deliverable. It holds the numbers each panel is drawn from, in
the units its axis uses, so a different plotting stack does not have to
re-derive anything. The .gp files are a reference rendering, not the paper's --
the paper's panels come from matplotlib and differ in typography.

The .gp scripts were run on gnuplot 6.0 and produce the three PDFs in this
directory. They need gnuplot >= 5.2 for `array`, and pdfcairo for the terminal.
Everything is set `noenhanced`, because the measure names carry underscores that
would otherwise be read as subscripts.


fig1 -- tail censoring
----------------------
  fig1_spacev.tsv       SpaceV100M,  n=100,000,000, 100d, L2,     nq=10,000
  fig1_youtube.tsv      YouTube15M,  1024d, cosine
  fig1_*_gain.tsv       the derived panel (see below)

  columns: subset, ef, recall, qps
  three index blocks per file, in this order:
      0  all queries
      1  hardest 1% removed
      2  hardest 10% removed

  Plot recall (col 3) against QPS (col 4), y on a log scale. The claim is the
  VERTICAL GAP at a fixed recall: the three curves come from one index and one
  set of searches, and differ only in which queries are averaged. A linear y
  axis flattens the gap and loses the point.

  The _gain files answer the same thing as a ratio: for each censored curve,
  the QPS the all-query set actually reaches over the QPS it would need to
  report that curve's mean recall. 1.0 is the no-censoring line.


fig2 -- mutual rank
-------------------
  fig2_panel1.tsv  vanilla  "what type of dog mix is a morkie"   (easy)
  fig2_panel2.tsv  vanilla  "what is the weather in allentown?"  (fragmented)
  fig2_panel3.tsv  syn      "what is the weather in allentown?"  (repaired)
  fig2_panels.txt  the same map, machine-readable

  Each file is an 11x11 matrix, read directly by `matrix with image`.
  Cell (i,j) is min(rank_i(j), rank_j(i)) over the whole corpus: how far down
  each document sits in the other's neighbour list, taking the nearer view.
  Low means the pair is mutually reachable.

  Row 0 and column 0 are the QUERY. Their entries are 1..10 by construction --
  that is the definition of the top-10, not a pattern in the data.

  THE DIAGONAL IS nan, not 0. It is a document against itself. Written as 0 it
  would plot as the strongest colour and read as the tightest pair in the
  matrix, which is the opposite of true. Draw those cells grey.

  Colour scale: log, 1 to 40,000, reversed viridis. The rule at 1,100 is L, the
  pull threshold -- pairs beyond it are what training is asked to close.


fig3 -- hardness measures
-------------------------
  fig3_hardness_measures.tsv

  columns: block, measure, GloVe, MS-MARCO, YouTube1M

  Measure names are shortened for plotting: phi1 is phi_1, Steiner is Steiner
  minimum effort, d2k/dk is the distance ratio, RC@k is relative contrast.
  three index blocks:
      0  (a) |rho| with recall@100 at ef=128
      1  (b) |rho| with least NDC to 0.98
      2  (c) partial |rho| at a fixed budget

  Absolute Spearman correlation of each per-query hardness measure against the
  quantity the block names. phi_1 is this paper's measure; the others are the
  classical ones it is being compared against. Values are already absolute, so
  bars run 0 to 1 and higher is a better predictor of search difficulty.

  Block (c) is partial correlation and its sign carries meaning in the source,
  but the figure plots magnitude, as (a) and (b) do.

  These corpora are public ANN benchmarks, not the BEIR collections of section
  5. That is deliberate: the question is whether fragmentation is a property of
  one embedding family or of embeddings.

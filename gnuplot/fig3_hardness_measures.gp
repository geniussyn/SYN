# Figure 3 -- hardness measures. gnuplot >= 5.2
# |Spearman rho| of each per-query hardness measure against the quantity the
# block names, on three corpora. phi_1 is the paper's measure.
# Columns: 1 block, 2 measure, 3-5 glove / msmarco / youtube1m.
# noenhanced throughout: the measure names carry underscores that would
# otherwise be read as subscripts.
set terminal pdfcairo size 7.2in,2.7in font "Linux Libertine O,10" noenhanced
set output "fig3_hardness_measures.pdf"

# --- house style: seaborn darkgrid, as the matplotlib figures use -----------
set style line 101 lc rgb "#FFFFFF" lw 1.1 dt 1
set grid back ls 101
set border 0
set tics nomirror scale 0 textcolor rgb "#262626"
set xlabel textcolor rgb "#262626"
set ylabel textcolor rgb "#262626"
set title textcolor rgb "#262626"
set key textcolor rgb "#262626"
set object 1 rectangle from graph 0,0 to graph 1,1 behind \
    fillcolor rgb "#EAEAF2" fillstyle solid 1.0 noborder
array CORP[3] = ["GloVe", "MS MARCO", "YouTube1M"]
array TITLE[3] = ["(a) vs recall@100", "(b) vs least NDC", "(c) partial"]
array FACE[3] = ["#4878a8", "#c04a4a", "#5aa05a"]
set style data histograms
set style histogram clustered gap 1
set style fill solid 0.85 border lc rgb "#ffffff"
set boxwidth 0.9
set yrange [0:1.05]
set ytics 0,0.5,1
set key at screen 0.5,0.07 center horizontal maxrows 1 samplen 1.2 noenhanced
set xtics font "Linux Libertine O,8" noenhanced
set ylabel "|rho|" offset 1.5,0
set multiplot layout 1,3 margins 0.07,0.98,0.26,0.90 spacing 0.07
do for [b=1:3] {
  set title TITLE[b] font "Linux Libertine O,10"
  plot for [col=3:5] "fig3_hardness_measures.tsv" index b-1 \
       using col:xtic(2) title CORP[col-2] lc rgb FACE[col-2]
}
unset multiplot

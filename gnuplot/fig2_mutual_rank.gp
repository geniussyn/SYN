# Figure 2 -- mutual rank. gnuplot >= 5.2
# Each panel is an 11x11 matrix: the query and its exact top-10, coloured by
# min(rank_i(j), rank_j(i)) over the whole corpus. Low is reachable.
# The diagonal is NaN -- a document against itself -- and is left blank.
set terminal pdfcairo size 7.6in,2.7in font "Helvetica,10" noenhanced
set output "fig2_mutual_rank.pdf"

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
array TITLE[3] = ["Vanilla, easy query", "Vanilla, fragmented", "SYN, the same query"]
set logscale cb
set cbrange [1:40000]
# viridis, reversed: dark is a distant pair
set palette defined (0 "#fde725", 0.5 "#21918c", 1 "#440154")
set cbtics 1,10,10000
set size ratio 1
unset key
# row and column 0 are the query; the ten documents follow
set xrange [-0.5:10.5]
set yrange [-0.5:10.5]
set tics scale 0.4 out
set xtics 0,2,10
set ytics 0,2,10
set multiplot layout 1,3 margins 0.06,0.90,0.12,0.88 spacing 0.07
do for [i=1:3] {
  set title TITLE[i] font "Helvetica,10"
  # one colour bar is enough; naming it on every panel runs the label into
  # the next panel's tics
  if (i == 3) { set cblabel "mutual rank" offset 1.2,0 } else { unset cblabel }
  plot sprintf("fig2_panel%d.tsv", i) matrix with image
}
unset multiplot

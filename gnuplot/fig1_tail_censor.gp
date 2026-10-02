# Figure 1 -- tail censoring. gnuplot >= 5.2
# Removing the hardest queries from the evaluation set moves the reported
# curve up without changing the index. The gap at fixed recall is the claim.
# Each file holds three index blocks, in the order of SUB below.
set terminal pdfcairo size 7.0in,2.8in font "Linux Libertine O,10" noenhanced
set output "fig1_tail_censor.pdf"

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
array SUB[3] = ["all queries", "hardest 1% removed", "hardest 10% removed"]
array CORP[2] = ["spacev", "youtube"]
array NAME[2] = ["SpaceV100M  100d, L2", "YouTube15M  1024d, cosine"]
array PT[3] = [7, 5, 9]
array LC[3] = ["#000000", "#6b6b6b", "#9e9e9e"]
set multiplot layout 1,2 margins 0.09,0.98,0.17,0.90 spacing 0.09
set logscale y
set key left bottom spacing 1.15 samplen 2 width -2
set xlabel "mean recall@10" offset 0,0.4
set ylabel "queries per second" offset 1.5,0
do for [c=1:2] {
  set title NAME[c] font "Linux Libertine O,10"
  plot for [i=1:3] sprintf("fig1_%s.tsv", CORP[c]) index i-1 using 3:4 \
       with linespoints lw 1.4 pt PT[i] ps 0.5 lc rgb LC[i] title SUB[i]
}
unset multiplot

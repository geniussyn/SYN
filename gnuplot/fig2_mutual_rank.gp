# Figures 2 and 6 -- mutual rank. gnuplot >= 5.2
# Each panel is an 10+1 square: the query and its exact top-10, coloured by
# min(rank_i(j), rank_j(i)) over the whole corpus. Low is reachable.
# The diagonal is NaN -- a document against itself -- and is left blank. Written
# as 0 it would plot as the tightest pair in the matrix, the opposite of true.
#
# The numbers in the cells are the figure's content, not decoration: a reader
# compares 788 against 4, not two shades of green. They come from a second file
# per panel, already formatted, and are drawn white on the dark cells and black
# on the light ones so they read either way.
#
# Row and column 0 are the query, and their entries are 1..10 by construction --
# that is the definition of the top-10, not a pattern. A star on a tick marks a
# document the qrels judge relevant.
set terminal pdfcairo size 7.6in,2.9in font "Helvetica,8" noenhanced
set output "fig2_mutual_rank.pdf"
set logscale cb
set cbrange [1:40000]
# viridis, reversed: dark is a distant pair
set palette defined (0 "#fde725", 0.5 "#21918c", 1 "#440154")
set cbtics 1,10,10000
# the label colours the `tc variable` column selects: 0 black, 1 white
set linetype 1 lc rgb "#000000"
set linetype 2 lc rgb "#FFFFFF"
set size ratio 1
unset key
set xrange [-0.5:10.5]
set yrange [-0.5:10.5]
set tics scale 0.3 out textcolor rgb "#262626" font ",6.5"
set multiplot layout 1,3 margins 0.05,0.90,0.13,0.83 spacing 0.07

set title "\"what type of dog mix is a morkie\"\neasy · outside the hard decile" font ",7.5" offset 0,-0.5
set xtics ("q" 0, "#0★" 1, "#1" 2, "#2" 3, "#3" 4, "#4" 5, "#5" 6, "#6" 7, "#7" 8, "#8" 9, "#9" 10)
set ytics ("q" 0, "#0★" 1, "#1" 2, "#2" 3, "#3" 4, "#4" 5, "#5" 6, "#6" 7, "#7" 8, "#8" 9, "#9" 10)
unset colorbox
plot "fig2_panel1.tsv" matrix with image notitle, \
     "fig2_panel1_labels.tsv" using 1:2:3:4 with labels font ",5.2" tc variable notitle

set title "\"what is the weather in allentown?\"\nfragmented · baseline embedding" font ",7.5" offset 0,-0.5
set xtics ("q" 0, "#0★" 1, "#1" 2, "#2" 3, "#3" 4, "#4" 5, "#5" 6, "#6" 7, "#7" 8, "#8" 9, "#9" 10)
set ytics ("q" 0, "#0★" 1, "#1" 2, "#2" 3, "#3" 4, "#4" 5, "#5" 6, "#6" 7, "#7" 8, "#8" 9, "#9" 10)
unset colorbox
plot "fig2_panel2.tsv" matrix with image notitle, \
     "fig2_panel2_labels.tsv" using 1:2:3:4 with labels font ",5.2" tc variable notitle

set title "\"what is the weather in allentown?\"\nthe same query · SYN" font ",7.5" offset 0,-0.5
set xtics ("q" 0, "#0★" 1, "#1" 2, "#2" 3, "#3" 4, "#4" 5, "#5" 6, "#6" 7, "#7" 8, "#8" 9, "#9" 10)
set ytics ("q" 0, "#0★" 1, "#1" 2, "#2" 3, "#3" 4, "#4" 5, "#5" 6, "#6" 7, "#7" 8, "#8" 9, "#9" 10)
set colorbox
set cblabel "mutual rank" offset -1,0
plot "fig2_panel3.tsv" matrix with image notitle, \
     "fig2_panel3_labels.tsv" using 1:2:3:4 with labels font ",5.2" tc variable notitle
unset multiplot

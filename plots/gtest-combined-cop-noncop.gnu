
set encoding utf8

set title figure_title
# set key right box height 1

set grid ytics
set xtics nomirror
set ytics nomirror
# unset ytics
# unset xtics

set terminal postscript eps size 16cm,12cm enhanced color font 'Helvetica,18' linewidth 2.5
set output output_file.".ps"

set xlabel 'Malicious Workers Percentage'
# set xrange [0.5:5.5]
set yrange [50:100]
set ylabel 'Accuracy' offset 1,0

set style histogram cluster
set style data histogram
# set style fill solid 0.3 border -1
# set xtic 1 offset character 0,0.3
# plot file1 using 2:xtic(1) title column(3) fs pattern 4 lc rgb '#29001e', \
# '' using 4 title column(5) lc rgb '#bf008c' fs pattern 7

# '' using 4 title "label 3", '' using 5 title "label 4"
plot file1 using 2:xtic(1) title column(3) fs pattern 1 lc rgb '#29001e', \
'' using 4 title column(5) fs pattern 2 lc rgb '#bf008c', \
'' using 6 title column(7) fs pattern 5 lc rgb '#29001e', \
'' using 8 title column(9) fs pattern 4 lc rgb '#bf008c'

# plot newhistogram "Set A", 'data1' using 2:xtic(1) title "tt1", '' using 3 title "label 2", '' using 4 title "label 3", '' using 5 title "label 4"
# plot newhistogram "Set A", 'data2' using 2 title "tt1", '' using 3 title "label 2", '' using 4 title "label 3", '' using 5 title "label 4"
# # replot

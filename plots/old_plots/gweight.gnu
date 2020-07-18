set encoding utf8

set title figure_title
set key right 
set key spacing 1.5

set grid x2tics
set xlabel 'Worker'
set xrange [0:11]
set yrange [-0.1:1.1]
set ylabel 'Weight' offset 1, 0
set xtics 1,1,10 nomirror
set ytics nomirror 

set terminal postscript eps size 16cm,12cm enhanced color font 'Helvetica,18' linewidth 2.5
set output output_file.".ps"

plot filename using 1:2 with points pt 5 ps 3.1 lc 'green' title 'Epoc1', \
'' using 1:3 with points pt 4 ps 3.1 lc 'red' title 'Epoc2', \
'' using 1:4 with points pt 3 ps 3.1 lc '#bf8c00' title 'Epoc3', \
'' using 1:5 with points pt 6 ps 3.1 lc '#0060dd' title 'Epoc4', \
'' using 1:6 with points pt 7 ps 3.1 lc 'gray' title 'Epoc5'

# replot

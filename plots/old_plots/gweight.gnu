set terminal postscript eps size 16cm,12cm enhanced color font 'Helvetica,24' linewidth 3
set output output_file.".ps"

set encoding utf8

# set title figure_title
set key left 
set key spacing 1.5

set grid x2tics
set xlabel 'Users'
set xrange [0:11]
set yrange [-0.1:1.1]
set ylabel "{/Symbol r}" offset 1, 0
set xtics 1,1,10 nomirror
set ytics nomirror 



plot filename using 1:2 with points pt 5 ps 4.5 lc 'green' title 'Epoc1', \
'' using 1:3 with points pt 4 ps 4.5 lc 'red' title 'Epoc2', \
'' using 1:4 with points pt 3 ps 4.5 lc '#bf8c00' title 'Epoc3', \
'' using 1:5 with points pt 6 ps 4.5 lc '#0060dd' title 'Epoc4', \
'' using 1:6 with points pt 7 ps 4.5 lc 'gray' title 'Epoc5'

# replot

set encoding utf8

set title  "Optimized Weight" font ",15"
set key right box height 1
set key spacing 2.10

set grid x2tics
set xlabel 'Workers'
set xrange [0:11]
set yrange [-0.1:1.1]
set ylabel 'Weight Value' 
# set x2label 'V' offset 0,2
set xtics nomirror offset 0,-0.5
set ytics nomirror 

set terminal postscript eps size 16cm,12cm enhanced color font 'Helvetica,13' lw 2.1 ps 1.8
set output filename.".ps"

# set x2tics rotate by 55 right offset 2,2.5
plot filename using 1:2 with points pt 5 ps 2.15 lc 'green' title 'Epoc1', \
'' using 1:3 with points pt 4 ps 2.10 lc 'red' title 'Epoc2', \
'' using 1:4 with points pt 3 ps 2.05 lc '#0060ad' title 'Epoc3', \
'' using 1:5 with points pt 6 ps 2.03 lc '#0060dd' title 'Epoc4', \
'' using 1:6 with points pt 7 ps 2 lc 'gray' title 'Epoc5'

# set terminal postscript eps size 16cm,12cm enhanced color font 'Helvetica,13' lw 2.1 ps 1.8
# set output filename.".ps"
# replot

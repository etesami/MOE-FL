
set encoding utf8

set style line 1 \
    linecolor rgb '#0060ad' \
    lt 1 linewidth 2 \
    pointtype 7 pointsize 1.5
set style line 2 \
    linecolor rgb '#D32636' \
    dt 2 linewidth 2 \
    pointtype 7 pointsize 1.5
set style line 3 \
    linecolor rgb '#048226' \
    dt 3 linewidth 2 \
    pointtype 7 pointsize 1.5
set style line 4 \
    linecolor rgb '#29001e' \
    dt 4 linewidth 2 \
    pointtype 7 pointsize 1.5
set style line 5 \
    linecolor rgb '#bf008c' \
    dt 7 linewidth 2 \
    pointtype 7 pointsize 1.5

set title figure_title
# set key bottom box height 1

set grid ytics
set xlabel 'Malicious Workers Percentage'
set xrange [0:100]
set yrange [50:100]
set ylabel 'Accuracy' offset 1,0
set xtics nomirror
set ytics nomirror

set terminal postscript eps size 16cm,12cm enhanced color font 'Helvetica,18' linewidth 2.5
set output output_file.".ps"

title1="20% Data Alteration"
title2="40% Data Alteration"
title3="50% Data Alteration"
title4="60% Data Alteration"
title5="80% Data Alteration"
# title1="Manupulated Data in Server"
# title2="Pure Data in Server"
plot file1 using 1:3:xtic(2) with linespoints ls 1 title title1, \
'' using 1:4 with linespoints ls 2 title title2, \
'' using 1:5 with linespoints ls 4 title title4, \
'' using 1:6 with linespoints ls 5 title title5,

# file3 using 1:3 with linespoints ls 3 title title3, \
# replot


set encoding utf8

set style line 1 \
    linecolor rgb '#0060ad' \
    linetype 1 linewidth 2 \
    pointtype 7 pointsize 1.5
set style line 2 \
    linecolor rgb '#E32636' \
    linetype 1 linewidth 2 \
    pointtype 7 pointsize 1.5

set title figure_title
set key right box height 1

set grid x2tics
set xlabel 'Epoch'
set xrange [0.5:5.5]
set yrange [0:2.5]
set ylabel 'Loss' offset 1.1, 0
set xtics nomirror
set ytics nomirror

set terminal postscript eps size 16cm,12cm enhanced color font 'Helvetica,18' linewidth 2.5
set output output_file.".ps"

title1="Average"
title2="Proposed Approach"
# title1="Manupulated Data in Server"
# title2="Pure Data in Server"
plot filename1 using 1:2 with linespoints ls 1 title title1, '' with labels point pt 7 offset char 2,1 textcolor ls 1 notitle, \
filename2 using 1:2 with linespoints ls 2 title title2, '' with labels point pt 7 offset char -2.5,-1.5 textcolor ls 2 notitle

# replot

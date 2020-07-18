
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

set title figure_title font ",16"
set key right box height 1 width 2

set grid x2tics
set xlabel 'Epoch'
set xrange [0.5:5.5]
set yrange [0:2.5]
set ylabel 'Loss'
set xtics nomirror
set ytics nomirror

set terminal postscript eps size 16cm,12cm enhanced color font 'Helvetica,18' linewidth 2.5
set output output_file.".ps"

title1="20%"
title2="40%"
title3="50%"
title4="60%"
title5="80%"
# title1="Manupulated Data in Server"
# title2="Pure Data in Server"
plot file1 using 1:2 with linespoints ls 1 title title1, '' with labels point pt 7 offset char -2.9,-1.4 textcolor ls 1 notitle, \
file2 using 1:2 with linespoints ls 2 title title2, '' with labels point pt 7 offset char -2.9,-1.1 textcolor ls 2 notitle, \
file3 using 1:2 with linespoints ls 3 title title3, '' with labels point pt 7 offset char -2.9,-1.1 textcolor ls 3 notitle, \
file4 using 1:2 with linespoints ls 4 title title4, '' with labels point pt 7 offset char -2.9,-1.1 textcolor ls 4 notitle, \
file5 using 1:2 with linespoints ls 5 title title5, '' with labels point pt 7 offset char -2.9,-1.1 textcolor ls 5 notitle

# replot

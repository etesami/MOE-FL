
set terminal postscript eps size 16cm,12cm enhanced color font 'Helvetica,24' linewidth 2
set output output_file.".ps"

set encoding utf8

set style line 1 \
    linecolor rgb '#0060ad' \
    lt 1 linewidth 6 \
    pointtype 7 pointsize 3
set style line 2 \
    linecolor rgb '#D32636' \
    dt 2 linewidth 6 \
    pointtype 7 pointsize 3
set style line 3 \
    linecolor rgb '#048226' \
    dt 3 linewidth 6 \
    pointtype 7 pointsize 3
set style line 4 \
    linecolor rgb '#29001e' \
    dt 4 linewidth 6 \
    pointtype 7 pointsize 3
set style line 5 \
    linecolor rgb '#bf008c' \
    dt 7 linewidth 6 \
    pointtype 7 pointsize 3

# set title figure_title
set key right

set grid ytics
set xlabel 'Epoch'
set xrange [0.5:5.5]
set yrange [0:2.5]
set ylabel 'Loss' offset 1,0
set xtics nomirror
set ytics nomirror


title1="20%"
title2="40%"
title3="50%"
title4="60%"
title5="80%"
# title1="Manupulated Data in Server"
# title2="Pure Data in Server"
plot file1 using 1:2 with linespoints ls 1 title title1, \
file2 using 1:2 with linespoints ls 2 title title2, \
file4 using 1:2 with linespoints ls 4 title title4, \
file5 using 1:2 with linespoints ls 5 title title5,

# file3 using 1:2 with linespoints ls 3 title title3, \
# replot


set encoding utf8

set style line 1 \
    linecolor rgb '#0060ad' \
    linetype 1 linewidth 2 \
    pointtype 7 pointsize 1.5

set title  "Test Loss/Accuracy" font ",15"
set key right box height 1

set grid x2tics
set xlabel 'Epoch'
set xrange [0.5:5.5]
set yrange [0:2.0]
set ylabel 'Loss'
# set x2label 'V' offset 0,2
set xtics nomirror
set ytics nomirror

set terminal postscript eps size 16cm,12cm enhanced color font 'Helvetica,13' linewidth 2.5
set output filename.".ps"

# set x2tics rotate by 55 right offset 2,2.5
plot filename using 1:2 with linespoints ls 1 title "Loss", '' with labels point pt 7 offset char 3,2 notitle

# set terminal postscript eps size 16cm,12cm enhanced color font 'Helvetica,13' linewidth 2.5
# set output filename.".ps"
# replot

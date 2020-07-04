
set encoding utf8

set style line 1 \
    linecolor rgb '#0060ad' \
    linetype 1 linewidth 2 \
    pointtype 7 pointsize 1.5

set title  "Train Loss" offset 0,3 font ",15"
set key right box height 1

set grid x2tics
set xlabel 'Epoch'
# set xrange [-5:]
# set ylabel 'T'
# set x2label 'V' offset 0,2
set xtics nomirror
set ytics nomirror

set terminal postscript eps size 18cm,12cm enhanced color font 'Helvetica,13' linewidth 3
set output filename.".ps"

set x2tics rotate by 65 right offset 1,3
plot filename using 1:5:xtic(int($1)%step_num == 0 ? strcol(2) : 1/0):x2tic(int($1)%10 == 0 ? strcol(4) : 1/0) with lines title "Loss" ls 1

# set terminal postscript eps size 18cm,12cm enhanced color font 'Helvetica,13' linewidth 3
# set output filename.".ps"
# replot
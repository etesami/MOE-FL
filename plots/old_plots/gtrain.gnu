set terminal postscript eps size 18cm,12cm enhanced color font 'Helvetica,20' linewidth 3
set output output_file.".ps"

set encoding utf8

set style line 1 \
    linecolor rgb '#0060ad' \
    linetype 1 linewidth 2 \
    pointtype 7 pointsize 1.5

# set title  "Train Loss" offset 0,3 font ",15"
set key right box height 1

set grid xtics
set grid ytics
set xlabel 'Epoch'
# set xrange [-5:]
set yrange [0:8]
set xtics nomirror
set ytics nomirror



# set x2tics rotate by 65 right offset 1,3
plot filename using 1:5:xtic(int($1)%step_num == 0 ? strcol(2) : 1/0) with lines title "Loss" ls 1
plot filename using 1:5:xtic(int($1)%step_num == 0 ? strcol(2) : 1/0):x2tic(int($1)%10 == 0 ? strcol(4) : 1/0) with lines title "Loss" ls 1

# replot
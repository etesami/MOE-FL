set terminal postscript eps size 16cm,12cm enhanced color font 'Helvetica,32'
set output output_file.".ps"

# set border 2
# set xrange [0:100]
# set yrange [0:]
# set xtics 0.1
# set ytics 0.1
# set xtics nomirror scale 0
# set ytics nomirror rangelimited
# unset xtics
unset key

# set xtics rotate by 65 right #offset 1,3

set encoding utf8

set grid ytics
# set xlabel xtitle
# set ylabel '{/Symbol rho}' offset -0.5,0

# set jitter spread 0.001
# set title font ",15"
# set title "swarm jitter with a large number of points\n approximates a violin plot"
# set style data points

# set style line 1 \
#     linecolor rgb '#0060ad' \
#     lt 1 linewidth 3 \
#     pointtype 7 pointsize 4
# set style line 2 \
#     linecolor rgb '#D32636' \
#     dt 3 linewidth 3 \
#     pointtype 7 pointsize 4

set linetype 1 lc "red" ps 10 pt 9
set linetype 2 lc "blue" ps 10 pt 9
set xtics 0.1
set ytics 0.05
# set style line 1 lc "red" ps 4 pt 5
# set style line 2 lc "blue" ps 4 pt 3

# set style increment user
# plot $viol1 lt 9, $viol2 lt 10
plot file1 using 1:2:3:3 lc var,\
# plot "data/output-a-moe_atk1_50_acc.txt" using 1:3 lt 9,
# "data/output-a-moe_atk1_50_acc.txt" using 1:3 lt 9
# "data/output-n-moe_atk1_50_acc.txt" using 1:3 lt 10

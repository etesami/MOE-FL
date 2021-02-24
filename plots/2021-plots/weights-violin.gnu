set terminal postscript eps size 16cm,12cm enhanced color font 'Helvetica,56'
set output output_file.".ps"

# set border 2
set xrange [0:100]
set yrange [0:]
# set xtics ("A" 0, "B" 1)
# set xtics nomirror scale 0
# set ytics nomirror rangelimited
unset xtics
unset key

# set xtics rotate by 65 right #offset 1,3

set encoding utf8

set grid ytics
set xlabel xtitle
set ylabel '{/Symbol rho}' offset -0.5,0

set jitter spread 0.001
set title font ",15"
# set title "swarm jitter with a large number of points\n approximates a violin plot"
set style data points

set linetype  1 lc "#d60000" ps 2.5 pt 5
set linetype 2 lc "#0300d6" ps 2.5 pt 5

# plot $viol1 lt 9, $viol2 lt 10
plot file1 using 1:3 lt lt_style,\
# plot "data/output-a-moe_atk1_50_acc.txt" using 1:3 lt 9,
# "data/output-a-moe_atk1_50_acc.txt" using 1:3 lt 9
# "data/output-n-moe_atk1_50_acc.txt" using 1:3 lt 10

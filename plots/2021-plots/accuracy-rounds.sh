#!/bin/bash

FILE1="moe_atk1_25.txt"
FILE2="avg_atk1_25.txt"
OUTPUT=$FILE1"_"$FILE2
gnuplot -persist -e "file1='$FILE1'" \
        -e "file2='$FILE2'" \
        -e "output_file='$OUTPUT'" plot-accuracy.gnu 

ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
rm $OUTPUT".ps"

###########

FILE1="moe_atk1_50.txt"
FILE2="avg_atk1_50.txt"
OUTPUT=$FILE1"_"$FILE2
gnuplot -persist -e "file1='$FILE1'" \
        -e "file2='$FILE2'" \
        -e "output_file='$OUTPUT'" plot-accuracy.gnu 

ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
rm $OUTPUT".ps"


###########


FILE1="moe_atk1_75.txt"
FILE2="avg_atk1_75.txt"
OUTPUT=$FILE1"_"$FILE2
gnuplot -persist -e "file1='$FILE1'" \
        -e "file2='$FILE2'" \
        -e "output_file='$OUTPUT'" plot-accuracy.gnu 

ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
rm $OUTPUT".ps"

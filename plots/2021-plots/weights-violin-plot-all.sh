#!/usr/bin/env bash

help="Usage: 
    ./plot-all-violin.sh --input INPUT"
eval "$(~/.docopts -A args -h "$help" : "$@")"

FILE_NAME=${args[INPUT]}
FILE_NAME=`echo $FILE_NAME | cut -d'.' -f 1`

FILE_A=$FILE_NAME"_a.txt"
FILE_N=$FILE_NAME"_n.txt"

PREFIX="/Users/ehsan/data/prepared_weights/"

FILE=$FILE_A
gnuplot -persist -e "file1='$PREFIX$FILE'" \
        -e "lt_style='1'" \
        -e "xtitle='Attackers'" \
        -e "output_file='$FILE'" weights-violin.gnu 
ps2pdf -dAutoRotatePages=/None -dEPSCrop $FILE".ps"
rm $FILE".ps"

FILE=$FILE_N
gnuplot -persist -e "file1='$PREFIX$FILE'" \
        -e "lt_style='2'" \
        -e "xtitle='Normal Users'" \
        -e "output_file='$FILE'" weights-violin.gnu 
ps2pdf -dAutoRotatePages=/None -dEPSCrop $FILE".ps"
rm $FILE".ps"
###########

#!/usr/bin/env bash
help="Usage: 
    ./plot-pca.sh DIR-NAME PREFIX"
eval "$(~/.docopts -A args -h "$help" : "$@")"

PARENT="/Users/ehsan/data/data_pca/"
PREFIX=${args[PREFIX]}
DIR=${args[DIR-NAME]}"/"
R=`ls -1 $PARENT$DIR | wc -l | awk '{print $1}'`
R=$(($R-1))
for ii in $(seq 0 $R); do
    FILE=$PREFIX"_R"$ii".txt"
    echo $FILE
    gnuplot -persist -e "file1='$PARENT$DIR$FILE'" \
            -e "output_file='$FILE'" pca.gnu 

    ps2pdf -dAutoRotatePages=/None -dEPSCrop $FILE".ps"
    rm $FILE".ps"
done


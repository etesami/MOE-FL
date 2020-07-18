#!/usr/local/bin/bash

DIR="../data_tmp/"
FILE1=$DIR"02_attk1_avg20_test"
FILE2=$DIR"02_attk1_avg40_test"
FILE3=$DIR"02_attk1_avg50_test"
FILE4=$DIR"02_attk1_avg60_test"
FILE5=$DIR"02_attk1_avg80_test"

TITLE="Non Cooperative Attack"
OUTPUT="att1-avg-all-test"
echo $OUTPUT
./paper-plot-1.sh test $FILE1 $FILE2 $FILE3 $FILE4 $FILE5 "$TITLE" $OUTPUT


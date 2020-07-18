#!/usr/local/bin/bash

# DIR="data_tmp/"
# FILE1=$DIR"02_attk1_avg20_test"
# FILE2=$DIR"02_attk1_avg40_test"
# FILE3=$DIR"02_attk1_avg50_test"
# FILE4=$DIR"02_attk1_avg60_test"
# FILE5=$DIR"02_attk1_avg80_test"

# TITLE="Non Cooperative Attack (Average)"
# OUTPUT="att1-avg-test-loss"
# echo $OUTPUT
# ./plot-main.sh test loss $FILE1 $FILE2 $FILE3 $FILE4 $FILE5 "$TITLE" $OUTPUT

# TITLE="Non Cooperative Attack (Average)"
# OUTPUT="att1-avg-test-acc"
# echo $OUTPUT
# ./plot-main.sh test acc $FILE1 $FILE2 $FILE3 $FILE4 $FILE5 "$TITLE" $OUTPUT

# FILE1=$DIR"03_attk1_opt20_test"
# FILE2=$DIR"03_attk1_opt40_test"
# FILE3=$DIR"03_attk1_opt50_test"
# FILE4=$DIR"03_attk1_opt60_test"
# FILE5=$DIR"03_attk1_opt80_test"

# TITLE="Non Cooperative Attack (Weighted Average)"
# OUTPUT="att1-opt-test-loss"
# echo $OUTPUT
# ./plot-main.sh test loss $FILE1 $FILE2 $FILE3 $FILE4 $FILE5 "$TITLE" $OUTPUT

# TITLE="Non Cooperative Attack (Weighted Average)"
# OUTPUT="att1-opt-test-acc"
# echo $OUTPUT
# ./plot-main.sh test acc $FILE1 $FILE2 $FILE3 $FILE4 $FILE5 "$TITLE" $OUTPUT


# DIR="data_tmp/"
# FILE1=$DIR"09-non-cooperative-test.txt"
# TITLE="Non-Cooperative Attack (Average vs. Weighted Average)"
# OUTPUT="att1-avg-opt-test"
# echo $OUTPUT
# ./plot-main.sh nc-combined $FILE1 "$TITLE" $OUTPUT


DIR="data_tmp/"
FILE1=$DIR"09-cooperative-non-cop-test.txt"
TITLE="Cooperative and Non-Cooperative Attack (Average vs. Weighted Average)"
OUTPUT="att2-avg-opt-test"
echo $OUTPUT
./plot-main.sh "cop-nc-combined" $FILE1 "$TITLE" $OUTPUT
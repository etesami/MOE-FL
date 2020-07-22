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

# TITLE="Non Cooperative Attack (Proposed Approach)"
# OUTPUT="att1-opt-test-loss"
# echo $OUTPUT
# ./plot-main.sh test loss $FILE1 $FILE2 $FILE3 $FILE4 $FILE5 "$TITLE" $OUTPUT

# TITLE="Non Cooperative Attack (Proposed Approach)"
# OUTPUT="att1-opt-test-acc"
# echo $OUTPUT
# ./plot-main.sh test acc $FILE1 $FILE2 $FILE3 $FILE4 $FILE5 "$TITLE" $OUTPUT



# ###################

# DIR="data_tmp/"
# FILE1=$DIR"04_attk2_avg20_100_test"
# FILE2=$DIR"04_attk2_avg40_100_test"
# FILE3=$DIR"04_attk2_avg50_100_test"
# FILE4=$DIR"04_attk2_avg60_100_test"
# FILE5=$DIR"04_attk2_avg80_100_test"

# TITLE="Cooperative Attack (Average)"
# OUTPUT="att2-avg-test-loss"
# echo $OUTPUT
# ./plot-main.sh test-cop loss $FILE1 $FILE2 $FILE3 $FILE4 $FILE5 "$TITLE" $OUTPUT

# TITLE="Cooperative Attack (Average)"
# OUTPUT="att2-avg-test-acc"
# echo $OUTPUT
# ./plot-main.sh test-cop acc $FILE1 $FILE2 $FILE3 $FILE4 $FILE5 "$TITLE" $OUTPUT



# FILE1=$DIR"05_attk2_opt20_100_test"
# FILE2=$DIR"05_attk2_opt40_100_test"
# FILE3=$DIR"05_attk2_opt50_100_test"
# FILE4=$DIR"05_attk2_opt60_100_test"
# FILE5=$DIR"05_attk2_opt80_100_test"

# TITLE="Cooperative Attack (Proposed Approach)"
# OUTPUT="att2-opt-test-loss"
# echo $OUTPUT
# ./plot-main.sh test-cop loss $FILE1 $FILE2 $FILE3 $FILE4 $FILE5 "$TITLE" $OUTPUT

# TITLE="Cooperative Attack (Proposed Approach)"
# OUTPUT="att2-opt-test-acc"
# echo $OUTPUT
# ./plot-main.sh test-cop acc $FILE1 $FILE2 $FILE3 $FILE4 $FILE5 "$TITLE" $OUTPUT

# # ## #############################


# DIR="data_tmp/"
# python prepare-non-cooperative-data.py
# FILE1=$DIR"09-non-cooperative-test.txt"
# TITLE="Non-Cooperative Attack (Average vs. Proposed Approach)"
# OUTPUT="att1-avg-opt-test"
# echo $OUTPUT
# ./plot-main.sh nc-combined $FILE1 "$TITLE" $OUTPUT


DIR="data_tmp/"
python prepare-cop-data.py
FILE1=$DIR"09-cooperative-test.txt"
TITLE="Cooperative and Non-Cooperative Attack (Average vs. Proposed Approach)"
OUTPUT="att2-avg-opt-test"
echo $OUTPUT
./plot-main.sh "cop-combined" $FILE1 "$TITLE" $OUTPUT






# python prepare-workers-vs-data-data.py

# DIR="data_tmp/"
# FILE1=$DIR"09-workers-data-avg.txt"
# TITLE="Cooperative Attack (AVG)"
# OUTPUT="att2-avg-workers-data"
# echo $OUTPUT
# ./plot-main.sh "workers-data" $FILE1 "$TITLE" $OUTPUT

# DIR="data_tmp/"
# FILE1=$DIR"09-workers-data-opt.txt"
# TITLE="Cooperative Attack (Proposed Approach)"
# OUTPUT="att2-opt-workers-data"
# echo $OUTPUT
# ./plot-main.sh "workers-data" $FILE1 "$TITLE" $OUTPUT
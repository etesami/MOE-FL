#!/bin/bash

DIR="../data_tmp/"
FILE1=$DIR"02_attk1_avg20_test"
FILE2=$DIR"02_attk1_avg40_test"
FILE3=$DIR"02_attk1_avg50_test"
FILE4=$DIR"02_attk1_avg60_test"
FILE5=$DIR"02_attk1_avg80_test"

TITLE="No\_Maliciou\_Worker"
OUTPUT="att1-avg-all-test"
echo $OUTPUT
echo $FILE1" "$FILE2" "$FILE3" "$FILE4" "$FILE5 && ./paper-plot-1.sh test $FILE1 $FILE2 $FILE3 $FILE4 $FILE5 $TITLE $OUTPUT

# =====================
# PER="20"
# PER_DATA="20"
# FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
# FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
# TITLE="$PER\_Percentage\_Maliciou\_Workers\_($PER_DATA\_Percentage\_of\_Data)"
# echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 $TITLE

# PER_DATA="40"
# FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
# FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
# TITLE="$PER\_Percentage\_Maliciou\_Workers\_($PER_DATA\_Percentage\_of\_Data)"
# echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 $TITLE

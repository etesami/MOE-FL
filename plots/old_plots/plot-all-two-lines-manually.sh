#!/bin/bash

DIR="../data_tmp/"
PRFX1="01_avg_"
PRFX2="01_opt_"
# PRFX3="02_att1_avg_"
# PRFX4="04_att1_opt_"
PRFX5="02_attk1_avg"
PRFX6="03_attk1_opt"
PRFX7="04_attk2_avg"
PRFX8="05_attk2_opt"


FILE1=$DIR$PRFX1"test"
FILE2=$DIR$PRFX2"test"
TITLE="No Maliciou Worker"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER="20"
FILE1=$DIR$PRFX5$PER"_test"
FILE2=$DIR$PRFX6$PER"_test"
TITLE="20 Percentage Maliciou Workers"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER="40"
FILE1=$DIR$PRFX5$PER"_test"
FILE2=$DIR$PRFX6$PER"_test"
TITLE="40 Percentage Maliciou Workers"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER="60"
FILE1=$DIR$PRFX5$PER"_test"
FILE2=$DIR$PRFX6$PER"_test"
TITLE="60 Percentage Maliciou Workers"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER="80"
FILE1=$DIR$PRFX5$PER"_test"
FILE2=$DIR$PRFX6$PER"_test"
TITLE="80 Percentage Maliciou Workers"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1


# =====================
PER="20"
PER_DATA="20"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER_DATA="40"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER_DATA="60"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER_DATA="80"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

# =======================
PER="40"
PER_DATA="20"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER_DATA="40"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER_DATA="60"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER_DATA="80"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

# # =======================
PER="50"
PER_DATA="20"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER_DATA="40"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER_DATA="60"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER_DATA="80"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1


# # =======================
PER="60"
PER_DATA="20"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER_DATA="40"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER_DATA="60"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER_DATA="80"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

# ========================
PER="80"
PER_DATA="20"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER_DATA="40"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER_DATA="60"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1

PER_DATA="80"
FILE1=$DIR$PRFX7$PER"_"$PER_DATA"_test"
FILE2=$DIR$PRFX8$PER"_"$PER_DATA"_test"
TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
echo $FILE1" "$FILE2 && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $FILE1


# ==============================
#
# ################################################
# Caution: Need to change labels in the gnu file #
# ################################################
#
# PER="80"
# PER_DATA="80"
# PRFX1="07_att2_opt_"
# PRFX2="08_att3_opt_"
# FILE1=$DIR$PRFX1$PER"_"$PER_DATA"_test"
# FILE2=$DIR$PRFX2$PER"_"$PER_DATA"_test"
# TITLE="$PER Percentage Maliciou Workers ($PER_DATA Percentage of Data)"
# OUTPUT_NAME="10_"$PRFX1$PER"_"$PER_DATA"_"$PRFX2$PER"_"$PER_DATA"_Pure_Data_Server"
# echo $FILE1" "$FILE2 "->" $OUTPUT_NAME && ./plot-two-lines.sh test $FILE1 $FILE2 "$TITLE" $OUTPUT_NAME


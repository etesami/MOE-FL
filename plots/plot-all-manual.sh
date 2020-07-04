#!/bin/bash

DIR="data_tmp/"
PRFX1="01_avg_"
PRFX2="02_opt_"
PRFX3="03_att1_avg_"
PRFX4="04_att1_opt_"

FILE=$DIR$PRFX1"train"
echo $FILE && ./plot.sh train $FILE
FILE=$DIR$PRFX1"train_server"
echo $FILE && ./plot.sh trainserv $FILE
FILE=$DIR$PRFX1"test"
echo $FILE && ./plot.sh test $FILE
echo 

FILE=$DIR$PRFX2"train"
echo $FILE && ./plot.sh train $FILE
FILE=$DIR$PRFX2"train_server"
echo $FILE && ./plot.sh trainserv $FILE
FILE=$DIR$PRFX2"test"
echo $FILE && ./plot.sh test $FILE
FILE=$DIR$PRFX2"weights"
echo $FILE && ./plot-weight.sh $FILE
echo 

FILE=$DIR$PRFX3"train"
echo $FILE && ./plot.sh train $FILE
FILE=$DIR$PRFX3"train_server"
echo $FILE && ./plot.sh trainserv $FILE
FILE=$DIR$PRFX3"test"
echo $FILE && ./plot.sh test $FILE
echo

FILE=$DIR$PRFX4"train"
echo $FILE && ./plot.sh train $FILE
FILE=$DIR$PRFX4"train_server"
echo $FILE && ./plot.sh trainserv $FILE
FILE=$DIR$PRFX4"test"
echo $FILE && ./plot.sh test $FILE
FILE=$DIR$PRFX4"weights"
echo $FILE && ./plot-weight.sh $FILE

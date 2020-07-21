#!/bin/bash



SCREENNAME="study1"
screen -dmS $SCREENNAME

# # # ################################  NO ATTACK

MODE="avg"
OUTPUT="01_"$MODE
TIME="1"
echo $OUTPUT
rm ~/FederatedLearning/data_tmp/$OUTPUT*
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=NO --output-file=$OUTPUT; date; }^M"

MODE="opt"
OUTPUT="01_"$MODE
TIME="1"
echo $OUTPUT
rm ~/FederatedLearning/data_tmp/$OUTPUT*
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT  -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=NO --output-file=$OUTPUT; date; }^M"


# # ################################  ATTACK 1

# SCREENNAME="study2"
# screen -dmS $SCREENNAME

## 02_attk1_avg20_
ATTK="1"; MODE="avg"
USRS="20"
TIME="400"
OUTPUT="02_attk"$ATTK"_"$MODE$USRS
echo $OUTPUT
rm ~/FederatedLearning/data_tmp/$OUTPUT*
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

### 02_attk1_avg40_
ATTK="1"; MODE="avg"
USRS="400"
OUTPUT="02_attk"$ATTK"_"$MODE$USRS
echo $OUTPUT
rm ~/FederatedLearning/data_tmp/$OUTPUT*
TIME="10"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

## 02_attk1_avg50_
ATTK="1"; MODE="avg"
USRS="50"
TIME="800"
OUTPUT="02_attk"$ATTK"_"$MODE$USRS
echo $OUTPUT
rm ~/FederatedLearning/data_tmp/$OUTPUT*
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

ATTK="1"; MODE="avg"
USRS="60"
TIME="800"
OUTPUT="02_attk"$ATTK"_"$MODE$USRS
echo $OUTPUT
rm ~/FederatedLearning/data_tmp/$OUTPUT*
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

ATTK="1"; MODE="avg"
USRS="80"
TIME="1200"
OUTPUT="02_attk"$ATTK"_"$MODE$USRS
echo $OUTPUT
rm ~/FederatedLearning/data_tmp/$OUTPUT*
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

ATTK="1"; MODE="avg"
USRS="100"
TIME="1200"
OUTPUT="02_attk"$ATTK"_"$MODE$USRS
echo $OUTPUT
rm ~/FederatedLearning/data_tmp/$OUTPUT*
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# # ############## OPTIMIZATION

ATTK="1"; MODE="opt"
USRS="20"
TIME="1600"
OUTPUT="03_attk"$ATTK"_"$MODE$USRS
echo $OUTPUT
rm ~/FederatedLearning/data_tmp/$OUTPUT*
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

ATTK="1"; MODE="opt"
USRS="40"
TIME="1600"
OUTPUT="03_attk"$ATTK"_"$MODE$USRS
echo $OUTPUT
rm ~/FederatedLearning/data_tmp/$OUTPUT*
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

ATTK="1"; MODE="opt"
USRS="50"
TIME="2000"
OUTPUT="03_attk"$ATTK"_"$MODE$USRS
echo $OUTPUT
rm ~/FederatedLearning/data_tmp/$OUTPUT*
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

ATTK="1"; MODE="opt"
USRS="60"
TIME="2000"
OUTPUT="03_attk"$ATTK"_"$MODE$USRS
echo $OUTPUT
rm ~/FederatedLearning/data_tmp/$OUTPUT*
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

ATTK="1"; MODE="opt"
USRS="80"
TIME="2400"
OUTPUT="03_attk"$ATTK"_"$MODE$USRS
echo $OUTPUT
rm ~/FederatedLearning/data_tmp/$OUTPUT*
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

ATTK="1"; MODE="opt"
USRS="100"
TIME="2400"
OUTPUT="03_attk"$ATTK"_"$MODE$USRS
echo $OUTPUT
rm ~/FederatedLearning/data_tmp/$OUTPUT*
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"






################################  ATTACK 2

# SCREENNAME="study3"
# screen -dmS $SCREENNAME

# ATTK="2"; MODE="avg"
# USRS="20"
# DATA="20"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="1800"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="40"
# DATA="20"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="1"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="50"
# DATA="20"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="1700"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="60"
# DATA="20"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="1"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="80"
# DATA="20"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="2150"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ########## 

# ATTK="2"; MODE="avg"
# USRS="20"
# DATA="20"
# TIME="2300"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="40"
# DATA="40"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="400"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="50"
# DATA="40"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="2400"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="60"
# DATA="40"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="450"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="80"
# DATA="40"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="2600"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ########## 

# ATTK="2"; MODE="avg"
# USRS="20"
# DATA="60"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="3000"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="40"
# DATA="60"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="800"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="50"
# DATA="60"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="3100"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="60"
# DATA="60"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="820"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="80"
# DATA="60"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="3200"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ########## 

# ATTK="2"; MODE="avg"
# USRS="20"
# DATA="80"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="3500"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="40"
# DATA="80"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="1100"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="50"
# DATA="80"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="3600"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="60"
# DATA="80"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="1150"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="80"
# DATA="80"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="4000"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ########## 

# ATTK="2"; MODE="avg"
# USRS="20"
# DATA="100"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="4200"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="50"
# DATA="100"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="4500"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="80"
# DATA="100"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="4750"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"






# ######################

# ATTK="2"; MODE="opt"
# USRS="20"
# DATA="20"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# TIME="1"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="40"
# DATA="20"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="1"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="50"
# DATA="20"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="300"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="60"
# DATA="20"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="300"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="80"
# DATA="20"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="600"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ########## 

# ATTK="2"; MODE="opt"
# USRS="20"
# DATA="40"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="600"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="40"
# DATA="40"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="900"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="50"
# DATA="40"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="900"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="60"
# DATA="40"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="1200"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="80"
# DATA="40"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="1250"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ########## 

# ATTK="2"; MODE="opt"
# USRS="20"
# DATA="60"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="1500"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="40"
# DATA="60"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="1500"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

#  USRS="50"
# DATA="60"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="1800"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="60"
# DATA="60"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="1800"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="80"
# DATA="60"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# TIME="2100"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ########## 

# ATTK="2"; MODE="opt"
# USRS="20"
# DATA="80"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="2100"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="40"
# DATA="80"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="2500"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="50"
# DATA="80"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="2600"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="60"
# DATA="80"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="2700"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="80"
# DATA="80"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="3000"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

########## 

# ATTK="2"; MODE="opt"
# USRS="20"
# DATA="100"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="3300"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="40"
# DATA="100"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# TIME="1"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="50"
# DATA="100"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# TIME="1"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="60"
# DATA="100"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# TIME="1"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="80"
# DATA="100"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# rm ~/FederatedLearning/data_tmp/$OUTPUT*
# TIME="4000"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"
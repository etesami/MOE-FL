#!/bin/bash



SCREENNAME="study1"
screen -dmS $SCREENNAME

# # # ################################  NO ATTACK

# MODE="avg"
# OUTPUT="01_"$MODE
# TIME="1"
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=NO --output-file=$OUTPUT; date; }^M"

# MODE="opt"
# OUTPUT="01_"$MODE
# TIME="1"
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT  -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=NO --output-file=$OUTPUT; date; }^M"


# # # ################################  ATTACK 1

# # SCREENNAME="study2"
# # screen -dmS $SCREENNAME

## 02_attk1_avg20_
ATTK="1"; MODE="avg"
USRS="20"
TIME="5"
OUTPUT="02_attk"$ATTK"_"$MODE$USRS
echo $OUTPUT

screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# ATTK="1"; MODE="avg"
# USRS="40"
# TIME="400"
# OUTPUT="02_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# ## 02_attk1_avg50_
# ATTK="1"; MODE="avg"
# USRS="50"
# TIME="800"
# OUTPUT="02_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# ATTK="1"; MODE="avg"
# USRS="60"
# TIME="800"
# OUTPUT="02_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# ATTK="1"; MODE="avg"
# USRS="80"
# TIME="1200"
# OUTPUT="02_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# ATTK="1"; MODE="avg"
# USRS="100"
# TIME="1200"
# OUTPUT="02_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# # # ############## OPTIMIZATION

# ATTK="1"; MODE="opt"
# USRS="20"
# TIME="1600"
# OUTPUT="03_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# ATTK="1"; MODE="opt"
# USRS="40"
# TIME="1600"
# OUTPUT="03_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# ATTK="1"; MODE="opt"
# USRS="50"
# TIME="2000"
# OUTPUT="03_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# ATTK="1"; MODE="opt"
# USRS="60"
# TIME="2000"
# OUTPUT="03_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# ATTK="1"; MODE="opt"
# USRS="80"
# TIME="2400"
# OUTPUT="03_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# ATTK="1"; MODE="opt"
# USRS="100"
# TIME="2400"
# OUTPUT="03_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"






################################  ATTACK 2

# ATTK="2"; MODE="avg"
# USRS="20"
# DATA="20"
# TIME="1"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="40"
# DATA="20"
# TIME="300"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="50"
# DATA="20"
# TIME="600"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="60"
# DATA="20"
# TIME="900"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="80"
# DATA="20"
# TIME="1200"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ########## 

# ATTK="2"; MODE="avg"
# USRS="20"
# DATA="40"
# TIME="1"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="40"
# DATA="40"
# TIME="1800"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="50"
# DATA="40"
# TIME="2100"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="60"
# DATA="40"
# TIME="2400"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="80"
# DATA="40"
# TIME="2700"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ########## 

# ATTK="2"; MODE="avg"
# USRS="20"
# DATA="60"
# TIME="3000"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="40"
# DATA="60"
# TIME="3300"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="50"
# DATA="60"
# TIME="3600"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="60"
# DATA="60"
# TIME="3900"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="80"
# DATA="60"
# TIME="4200"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ########## 

# ATTK="2"; MODE="avg"
# USRS="20"
# DATA="80"
# TIME="4500"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="40"
# DATA="80"
# TIME="4800"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="50"
# DATA="80"
# TIME="5100"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="60"
# DATA="80"
# TIME="5100"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="80"
# DATA="80"
# TIME="5400"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

########## 

# ATTK="2"; MODE="avg"
# USRS="20"
# DATA="100"
# TIME="5400"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="40"
# DATA="100"
# TIME="280"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="50"
# DATA="100"
# TIME="5400"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="60"
# DATA="100"
# TIME="1"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="avg"
# USRS="80"
# DATA="100"
# TIME="5700"
# OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"





# ######################

# ATTK="2"; MODE="opt"
# USRS="20"
# DATA="20"
# TIME="6000"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="40"
# DATA="20"
# TIME="6300"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="50"
# DATA="20"
# TIME="6600"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="60"
# DATA="20"
# TIME="6900"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="80"
# DATA="20"
# TIME="7200"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ########## 

# ATTK="2"; MODE="opt"
# USRS="20"
# DATA="40"
# TIME="7500"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="40"
# DATA="40"
# TIME="7800"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="50"
# DATA="40"
# TIME="8100"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="60"
# DATA="40"
# TIME="8400"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="80"
# DATA="40"
# TIME="8700"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ########## 

# ATTK="2"; MODE="opt"
# USRS="20"
# DATA="60"
# TIME="9000"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="40"
# DATA="60"
# TIME="9300"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# USRS="50"
# DATA="60"
# TIME="9600"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="60"
# DATA="60"
# TIME="9900"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="80"
# DATA="60"
# TIME="1000"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

########## 





# ATTK="2"; MODE="opt"
# USRS="20"
# DATA="80"
# TIME="1"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="40"
# DATA="80"
# TIME="1"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="50"
# DATA="80"
# TIME="300"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="60"
# DATA="80"
# TIME="400"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="80"
# DATA="80"
# TIME="600"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ######### 

# ATTK="2"; MODE="opt"
# USRS="20"
# DATA="100"
# TIME="600"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT

# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="40"
# DATA="100"
# TIME="900"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="50"
# DATA="100"
# TIME="900"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="60"
# DATA="100"
# TIME="1200"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

# ATTK="2"; MODE="opt"
# USRS="80"
# DATA="100"
# TIME="1200"
# OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
# echo $OUTPUT
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; rm -f ~/FederatedLearning/data_tmp/$OUTPUT*; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"
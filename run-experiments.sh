#!/bin/bash

# SCREENNAME="study1"
# screen -dmS $SCREENNAME

# # ################################  NO ATTACK

# ## 01_avg_
# MODE="avg"
# OUTPUT="01_"$MODE
# echo $OUTPUT
# TIME="1"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=NO --output-file=$OUTPUT; date; }^M"

# ## 01_opt_
# MODE="opt"
# OUTPUT="01_"$MODE
# echo $OUTPUT
# TIME="1"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT  -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=NO --output-file=$OUTPUT; date; }^M"


# # ################################  ATTACK 1

# SCREENNAME="study2"
# screen -dmS $SCREENNAME

# ATTK="1"
# MODE="avg"

# ## 02_attk1_avg20_
# USRS="20"
# OUTPUT="02_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT
# TIME="50"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# ### 02_attk1_avg40_
# USRS="40"
# OUTPUT="02_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT
# TIME="50"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# ## 02_attk1_avg50_
# USRS="50"
# OUTPUT="02_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT
# TIME="300"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# USRS="60"
# OUTPUT="02_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT
# TIME="350"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# USRS="80"
# OUTPUT="02_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT
# TIME="750"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# USRS="100"
# OUTPUT="02_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT
# TIME="800"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# ##############
# MODE="opt"

# USRS="20"
# OUTPUT="03_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT
# TIME="1150"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# USRS="40"
# OUTPUT="03_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT
# TIME="1150"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# USRS="50"
# OUTPUT="03_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT
# TIME="1500"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# USRS="60"
# OUTPUT="03_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT
# TIME="1500"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# USRS="80"
# OUTPUT="03_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT
# TIME="1850"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"

# USRS="100"
# OUTPUT="03_attk"$ATTK"_"$MODE$USRS
# echo $OUTPUT
# TIME="1850"
# screen -S $SCREENNAME -X screen -t $OUTPUT
# screen -S $SCREENNAME -p $OUTPUT -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --output-file=$OUTPUT; date; }^M"









################################  ATTACK 2

SCREENNAME="study3"
screen -dmS $SCREENNAME
ATTK="2"

MODE="avg"

USRS="20"
DATA="20"
OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="1800"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="50"
DATA="20"
OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="1700"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="80"
DATA="20"
OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="2150"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

########## 

USRS="20"
DATA="40"
OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="2300"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="50"
DATA="40"
OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="2400"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="80"
DATA="40"
OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="2600"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

########## 

USRS="20"
DATA="60"
OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="3000"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="50"
DATA="60"
OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="3100"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="80"
DATA="60"
OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="3200"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

########## 

USRS="20"
DATA="80"
OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="3500"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="50"
DATA="80"
OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="3600"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="80"
DATA="80"
OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="4000"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

########## 

USRS="20"
DATA="100"
OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="4200"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="50"
DATA="100"
OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="4500"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="80"
DATA="100"
OUTPUT="04_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="4750"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

######################

MODE="opt"
ATTK="2"

USRS="20"
DATA="20"
OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="5000"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="50"
DATA="20"
OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="5100"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="80"
DATA="20"
OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="5300"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

########## 

USRS="20"
DATA="40"
OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="5700"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="50"
DATA="40"
OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="6000"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="80"
DATA="40"
OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="6100"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

########## 

USRS="20"
DATA="60"
OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="6550"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="50"
DATA="60"
OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="6700"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="80"
DATA="60"
OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="7000"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

########## 

USRS="20"
DATA="80"
OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="7200"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="50"
DATA="80"
OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="7520"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="80"
DATA="80"
OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="7900"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

########## 

USRS="20"
DATA="100"
OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="8120"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="50"
DATA="100"
OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="8420"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"

USRS="80"
DATA="100"
OUTPUT="05_attk"$ATTK"_"$MODE$USRS"_"$DATA
echo $OUTPUT
TIME="8500"
screen -S $SCREENNAME -X screen -t $OUTPUT
screen -S $SCREENNAME -p $OUTPUT -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=$ATTK --workers-percentage=$USRS --data-percentage=$DATA --output-file=$OUTPUT; date; }^M"
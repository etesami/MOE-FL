#!/bin/bash


################################  NO ATTACK
# screen -dmS no-attack
# MODE="avg"
# OUTPUT="01_"$MODE
# TIME="1"
# screen -S no-attack -X screen -t $MODE
# screen -S no-attack -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=NO --output-file=$OUTPUT; date; }^M"

# MODE="opt"
# OUTPUT="01_"$MODE
# TIME="300"
# screen -S no-attack -X screen -t $MODE
# screen -S no-attack -X stuff \
# "{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=NO --output-file=$OUTPUT; date; }^M"


################################  ATTACK 1

MODE="opt"
OUTPUT="01_"$MODE
TIME="300"
screen -S no-attack -X screen -t $MODE
screen -S no-attack -X stuff \
"{ sleep $TIME; start="'`date`'"; python run-study.py --$MODE --attack=NO --output-file=$OUTPUT; date; }^M"

#!/usr/bin/env bash

help="Usage: 
    ./transfer-accuracy.sh AGENT DIR OUTPUT"
eval "$(~/ehsan/.docopts -A args -h "$help" : "$@")"

AGENT=${args[AGENT]}
DIR_NAME=${args[DIR]}
OUTPUT=${args[OUTPUT]}

scp $AGENT:/home/savi/ehsan/FederatedLearning/data_output/$DIR_NAME/accuracy ~/ehsan/data/$OUTPUT




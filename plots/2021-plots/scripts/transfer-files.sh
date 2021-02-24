#!/usr/bin/env bash

help="Usage: 
    ./transfer-accuracy.sh TYPE AGENT DIR OUTPUT"
eval "$(~/ehsan/.docopts -A args -h "$help" : "$@")"

TYPE=${args[TYPE]}
AGENT=${args[AGENT]}
DIR_NAME=${args[DIR]}
OUTPUT=${args[OUTPUT]}

if [[ $TYPE == "accuracy" || $TYPE == "train_loss" || $TYPE == "opt_weights"  || $TYPE == "attackers" ]]; then
    scp $AGENT:/home/savi/ehsan/FederatedLearning/data_output/$DIR_NAME/$TYPE ~/ehsan/data/$TYPE/$OUTPUT
else
    echo "ERROR: Not expected input for TYPE!"
    exit 1
fi




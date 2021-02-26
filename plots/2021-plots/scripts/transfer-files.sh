#!/usr/bin/env bash

help="Usage: 
    ./transfer-files.sh TYPE AGENT DIR OUTPUT
    ./transfer-files.sh --rsync AGENT DIR OUTPUT"
eval "$(~/ehsan/.docopts -A args -h "$help" : "$@")"

TYPE=${args[TYPE]}
AGENT=${args[AGENT]}
DIR_NAME=${args[DIR]}
OUTPUT=${args[OUTPUT]}
RSYNC=${args[--rsync]}

if $RSYNC; then
    rsync -arz $AGENT:/home/savi/ehsan/FederatedLearning/data_output/$DIR_NAME $OUTPUT
elif [[ $TYPE == "accuracy" || $TYPE == "train_loss" || $TYPE == "opt_weights"  || $TYPE == "attackers" ]]; then
    scp $AGENT:/home/savi/ehsan/FederatedLearning/data_output/$DIR_NAME/$TYPE ~/ehsan/data/$TYPE/$OUTPUT
else
    echo "ERROR: Not expected input for TYPE!"
    exit 1
fi
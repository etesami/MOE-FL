#!/usr/bin/env bash

help="Usage: 
    ./remote-pca.sh TYPE DIR OUTPUT ROUND"
eval "$(~/ehsan/.docopts -A args -h "$help" : "$@")"

TYPE=${args[TYPE]}
AGENT=${args[AGENT]}
DIR_NAME=${args[DIR]}
OUTPUT=${args[OUTPUT]}
ROUND=${args[ROUND]}

PREFIX="/home/savi/ehsan/FederatedLearning/"
source ~/ehsan/venv/bin/activate
python $PREFIX"/plots/2021-plots/scripts/prepare-pca.py" --type $TYPE --output-dir $OUTPUT --dir-name $DIR_NAME --round $ROUND




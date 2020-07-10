#!/bin/bash

unset ERR
if [[ -n "$1" ]]; then
   FILENAME=$1
else
   ERR=1
fi

if [[ -z "$ERR" ]]; then
    nn=`cat $FILENAME | wc -l | awk {'print $1'}`;
    cp $FILENAME $FILENAME".tmp"
    for ii in $(seq 1 $nn); do
      sed -i '' ""$ii"s/.*/$ii &/" $FILENAME".tmp";
    done
else
    echo "Input format: <fine-name>"
fi


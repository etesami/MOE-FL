#!/bin/bash

unset ERR
if [[ -n "$1" ]]; then
   FILENAME=$1
else
   ERR=1
fi

if [[ -z "$ERR" ]]; then
    nn=`cat $FILENAME | wc -l | awk {'print $1'}`;
    for ii in $(seq 1 $nn); do
      sed -i .bak ""$ii"s/.*/$ii &/" $FILENAME;
    done
else
    echo "Input format: <fine-name>"
fi


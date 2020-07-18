#!/bin/bash

unset ERR
if [[ -n "$1" ]]; then
   FILENAME=$1
else
   ERR=1
fi

if [[ -z "$ERR" ]]; then
	  if [ ! -e $FILENAME ]; then
	  	echo "File deos nto exist!"
	  	exit 1
	  fi
        source ~/venv/bin/activate
        python transform_weight.py $FILENAME
        NEW_FILE=$FILENAME"_tmp"
        gnuplot -persist -e "filename='$NEW_FILE'" gweight.gnu
        ps2pdf -dAutoRotatePages=/None -dEPSCrop $NEW_FILE".ps"
        rm $NEW_FILE".ps"
else
    echo "Input format: <fileName>"
fi


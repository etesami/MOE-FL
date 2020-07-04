#!/bin/bash

unset ERR
if [[ -n "$1" ]]; then
   MODE=$1
else
   ERR=1
fi
if [[ -n "$2" ]]; then
   FILENAME=$2
else
   ERR=1
fi


if [[ -z "$ERR" ]]; then
	  if [ ! -e $FILENAME ]; then
	  	echo "File does not exist!"
	  	exit 1
	  fi
		if [ "$1" == "train" ]; then
			./add-lines.sh $FILENAME
			gnuplot -persist -e "filename='$FILENAME'" -e "step_num=94" gtrain.gnu 
		elif [ "$1" == "trainserv" ]; then
			./add-lines.sh $FILENAME
			gnuplot -persist -e "filename='$FILENAME'" -e "step_num=19" gtrain.gnu 
		else
			gnuplot -persist -e "filename='$FILENAME'" -e "step_num=19" gtest.gnu 
		fi
		ps2pdf -dAutoRotatePages=/None -dEPSCrop $FILENAME".ps"
		rm $FILENAME".ps"
else
    echo "Input format: <test/train trainserv> <fileName>"
fi


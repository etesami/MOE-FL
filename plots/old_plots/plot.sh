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
			FILENAME_TMP=$FILENAME".tmp"
			if [ ! -e $FILENAME_TMP ]; then
				echo "Seems modified file was not created!"
				exit 1
			fi
			gnuplot -persist -e "filename='$FILENAME_TMP'" -e "step_num=94" gtrain.gnu 
			rm $FILENAME_TMP
		elif [ "$1" == "trainserv" ]; then
			./add-lines.sh $FILENAME
			FILENAME_TMP=$FILENAME".tmp"
			if [ ! -e $FILENAME_TMP ]; then
				echo "Seems modified file was not created!"
				exit 1
			fi
			gnuplot -persist -e "filename='$FILENAME_TMP'" -e "step_num=19" gtrain.gnu 
			rm $FILENAME_TMP
		else
			gnuplot -persist -e "filename='$FILENAME_TMP'" -e "step_num=19" gtest.gnu 
		fi
		ps2pdf -dAutoRotatePages=/None -dEPSCrop $FILENAME_TMP".ps"
		rm $FILENAME_TMP".ps"
else
    echo "Input format: <test/train trainserv> <fileName>"
fi


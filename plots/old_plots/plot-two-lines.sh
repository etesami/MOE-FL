#!/bin/bash

unset ERR
if [[ -n "$1" ]]; then
   MODE=$1
else
   ERR=1
fi
if [[ -n "$2" ]]; then
   FILENAME1=$2
else
   ERR=1
fi
if [[ -n "$3" ]]; then
   FILENAME2=$3
else
   ERR=1
fi
if [[ -n "$4" ]]; then
   TITLE="$4"
else
   ERR=1
fi
if [[ -n "$5" ]]; then
   OUTPUT_FILE="$5"
else
   ERR=1
fi

if [[ -z "$ERR" ]]; then
	  if [[ ! -e $FILENAME1 || ! -e $FILENAME2 ]]; then
	  	echo "Files does not exist!"
	  	exit 1
	  fi
		if [ "$1" == "test" ]; then
         # ./add-lines.sh $FILENAME1
			# FILENAME1=$FILENAME1".tmp"
         # ./add-lines.sh $FILENAME2
			# FILENAME2=$FILENAME2".tmp"
			gnuplot -persist -e "filename1='$FILENAME1'" -e "filename2='$FILENAME2'" -e "figure_title='$TITLE'" -e "output_file='$OUTPUT_FILE'" -e "step_num=19" gtest-two-lines.gnu 

         # rm $FILENAME1 $FILENAME2

         ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT_FILE".ps"
         rm $OUTPUT_FILE".ps"
         # NEW_FILE_NAME=`ls $OUTPUT_FILE".pdf" | tr -d '\'`
         # mv $OUTPUT_FILE".pdf" $NEW_FILE_NAME
         echo $NEW_FILE_NAME
        fi
else
    echo "Input format: <test> <fileName1> <fileName2> <title> <output-file-name>"
fi
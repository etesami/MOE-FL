#!/usr/local/bin/bash
#
# Usage: plot.sh MODE FILENAME OUTPUT
#		 plot.sh MODE FILENAME
#
source docopts.sh --auto "$@"

# for a in ${!ARGS[@]} ; do
#     echo "$a = ${ARGS[$a]}"
# done

MODEL=${ARGS[MODE]}
FILENAME=${ARGS[FILENAME]}
OUTPUT=${ARGS[OUTPUT]}


if [ ! -e $FILENAME ]; then
echo "File does not exist!"
exit 1
fi
if [ "$1" == "train" ]; then
	./../add-lines.sh $FILENAME
	FILENAME_TMP=$FILENAME".tmp"
	if [ ! -e $FILENAME_TMP ]; then
		echo "Seems modified file was not created!"
		exit 1
	fi
	gnuplot -persist -e "filename='$FILENAME_TMP'" -e "output_file=$OUTPUT" -e "step_num=94" gtrain.gnu 
	rm $FILENAME_TMP
elif [ "$1" == "trainserv" ]; then
	./../add-lines.sh $FILENAME
	FILENAME_TMP=$FILENAME".tmp"
	if [ ! -e $FILENAME_TMP ]; then
		echo "Seems modified file was not created!"
		exit 1
	fi
	gnuplot -persist -e "filename='$FILENAME_TMP'" -e "output_file='$OUTPUT'" -e "step_num=19" gtrain.gnu 
	rm $FILENAME_TMP
else
	gnuplot -persist -e "filename='$FILENAME'" -e "output_file='$OUTPUT'" -e "step_num=19" gtest.gnu 
fi
ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
rm $OUTPUT".ps"


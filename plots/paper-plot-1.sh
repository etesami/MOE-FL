#!/bin/bash
#
# Usage: plot.sh MODE FILE1 FILE2 FILE3 FILE4 FILE5 TITLE OUTPUT
#
#
source docopts.sh --auto "$@"

# for a in ${!ARGS[@]} ; do
#     echo "$a = ${ARGS[$a]}"
# done

MODE=${ARGS[MODE]}
FILE1=${ARGS[FILE1]}
FILE2=${ARGS[FILE2]}
FILE3=${ARGS[FILE3]}
FILE4=${ARGS[FILE4]}
FILE5=${ARGS[FILE5]}
TITLE=${ARGS[TITLE]}
OUTPUT=${ARGS[OUTPUT]}


if [[ ! -e $FILE1 || ! -e $FILE2 || ! -e $FILE3 || ! -e $FILE4 || ! -e $FILE5 ]]; then
echo "Files does not exist!"
exit 1
fi
if [ "$MODE" == "test" ]; then
    ./add-lines.sh $FILE1
    ./add-lines.sh $FILE2
    ./add-lines.sh $FILE3
    ./add-lines.sh $FILE4
    ./add-lines.sh $FILE5
    FILE1_TMP=$FILE1".tmp"
    FILE2_TMP=$FILE2".tmp"
    FILE3_TMP=$FILE3".tmp"
    FILE4_TMP=$FILE4".tmp"
    FILE5_TMP=$FILE5".tmp"

    gnuplot -persist -e "file1='$FILE1_TMP'" -e "file2='$FILE2_TMP'" \
    -e "file3='$FILE3_TMP'" -e "file4='$FILE4_TMP'" -e "file5='$FILE5_TMP'" 
    -e "figure_title='$TITLE'" -e "output_file='$OUTPUT'" -e "step_num=19" gtest-five-plots.gnu 

    rm $FILE1_TMP $FILE2_TMP $FILE3_TMP $FILE4_TMP $FILE5_TMP

    ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
    rm $OUTPUT".ps"
    NEW_FILE_NAME=`ls $OUTPUT".pdf" | tr -d '\'`
    mv $OUTPUT".pdf" $NEW_FILE_NAME
    echo $NEW_FILE_NAME
fi

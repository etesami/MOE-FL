#!/usr/local/bin/bash
#
# Usage: plot-main.sh MODE TYPE FILE1 FILE2 FILE3 FILE4 FILE5 TITLE OUTPUT
#        plot-main.sh MODE FILE1 TITLE OUTPUT
#
#
source docopts.sh --auto "$@"

# for a in ${!ARGS[@]} ; do
#     echo "$a = ${ARGS[$a]}"
# done

MODE=${ARGS[MODE]}
TYPE=${ARGS[TYPE]}
FILE1=${ARGS[FILE1]}
FILE2=${ARGS[FILE2]}
FILE3=${ARGS[FILE3]}
FILE4=${ARGS[FILE4]}
FILE5=${ARGS[FILE5]}
TITLE=${ARGS[TITLE]}
OUTPUT=${ARGS[OUTPUT]}

if [ "$MODE" == "test" ]; then
    if [[ ! -e $FILE1 || ! -e $FILE2 || ! -e $FILE3 || ! -e $FILE4 || ! -e $FILE5 ]]; then
    echo "Files does not exist!"
    exit 1
    fi
    if [[ "$TYPE" == "loss" ]]; then
        gnuplot -persist -e "file1='$FILE1'" -e "file2='$FILE2'" \
        -e "file3='$FILE3'" -e "file4='$FILE4'" -e "file5='$FILE5'" \
        -e "figure_title='$TITLE'" -e "output_file='$OUTPUT'" -e "step_num=19" gtest-loss.gnu 
    elif [[ "$TYPE" == "acc" ]]; then
        gnuplot -persist -e "file1='$FILE1'" -e "file2='$FILE2'" \
        -e "file3='$FILE3'" -e "file4='$FILE4'" -e "file5='$FILE5'" \
        -e "figure_title='$TITLE'" -e "output_file='$OUTPUT'" -e "step_num=19" gtest-acc.gnu 
    fi

    ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
    rm $OUTPUT".ps"
    echo $NEW_FILE_NAME

elif [ "$MODE" == "nc-combined" ]; then
    if [[ ! -e $FILE1 ]]; then
    echo "File(s) does not exist!"
    exit 1
    fi
    gnuplot -persist -e "file1='$FILE1'" \
        -e "figure_title='$TITLE'" -e "output_file='$OUTPUT'" gtest-combined.gnu 
    ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
    rm $OUTPUT".ps"
    echo $NEW_FILE_NAME
fi

#!/usr/local/bin/bash
#
# Usage: plot-weight.sh FILE1 TITLE OUTPUT
#
source docopts.sh --auto "$@"

# for a in ${!ARGS[@]} ; do
#     echo "$a = ${ARGS[$a]}"
# done

FILENAME=${ARGS[FILE1]}
TITLE=${ARGS[TITLE]}
OUTPUT=${ARGS[OUTPUT]}

if [ ! -e $FILENAME ]; then
echo "File deos nto exist!"
exit 1
fi

source ~/venv/bin/activate
python transform_weight.py $FILENAME
NEW_FILE=$FILENAME"_tmp"
gnuplot -persist -e "filename='$NEW_FILE'" -e "figure_title=\"$TITLE\"" -e "output_file='$OUTPUT'" gweight.gnu
ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
rm $OUTPUT".ps" $NEW_FILE


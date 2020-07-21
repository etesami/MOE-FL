#!/usr/local/bin/bash
#
# Usage: add-lines.sh FILENAME 
#
source docopts.sh --auto "$@"

# for a in ${!ARGS[@]} ; do
#     echo "$a = ${ARGS[$a]}"
# done

FILENAME=${ARGS[FILENAME]}

if [ ! -e $FILENAME ]; then
echo "File does not exist!"
exit 1
fi

nn=`cat $FILENAME | wc -l | awk {'print $1'}`;
cp $FILENAME $FILENAME".tmp"
for ii in $(seq 1 $nn); do
  sed -i '' ""$ii"s/.*/$ii &/" $FILENAME".tmp";
done


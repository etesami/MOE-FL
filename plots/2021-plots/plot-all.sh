#!/bin/bash
PREFIX="~/data/"

# FILE1="moe_niid_atk1_san_331.txt"
# FILE2="avg_niid_atk1_san_332.txt"

# OUTPUT=$FILE1"_"$FILE2
# gnuplot -persist -e "file1='$PREFIX$FILE1'" \
#         -e "file2='$PREFIX$FILE2'" \
#         -e "output_file='$OUTPUT'" accuracy-rounds-two.gnu 

# ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
# rm $OUTPUT".ps"

# ###########

# FILE1="moe_niid_atk1_5perc_san_331.txt"
# FILE2="moe_niid_atk1_15perc_san_335.txt"

# OUTPUT=$FILE1"_"$FILE2
# gnuplot -persist -e "file1='$PREFIX$FILE1'" \
#         -e "file2='$PREFIX$FILE2'" \
#         -e "output_file='$OUTPUT'" accuracy-rounds-two.gnu 

# ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
# rm $OUTPUT".ps"

# ###########

# FILE1="moe_niid_atk2_san_334.txt"
# FILE2="avg_niid_atk2_san_333.txt"

# OUTPUT=$FILE1"_"$FILE2
# gnuplot -persist -e "file1='$PREFIX$FILE1'" \
#         -e "file2='$PREFIX$FILE2'" \
#         -e "output_file='$OUTPUT'" accuracy-rounds-two.gnu 

# ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
# rm $OUTPUT".ps"

# # ###########


# FILE1="fedavg_niid_no_attack_san_330.txt"
# OUTPUT=$FILE1
# gnuplot -persist -e "file1='$PREFIX$FILE1'" \
#         -e "file2='$PREFIX$FILE2'" \
#         -e "output_file='$OUTPUT'" accuracy-rounds-one.gnu 

# ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
# rm $OUTPUT".ps"

###########




##################################################################

# FILE1="moe_atk1_25.txt"
# FILE2="avg_atk1_25.txt"
# O1=`echo $FILE1 | awk -F. '{print $1}'`
# O2=`echo $FILE2 | awk -F. '{print $1}'`
# OUTPUT=$O1"_"$O2
# gnuplot -persist -e "file1='$PREFIX$FILE1'" \
#         -e "file2='$PREFIX$FILE2'" \
#         -e "output_file='$OUTPUT'" accuracy-rounds-two.gnu 

# ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
# rm $OUTPUT".ps"

# # ###########

# FILE1="moe_atk1_50.txt"
# FILE2="avg_atk1_50.txt"
# O1=`echo $FILE1 | awk -F. '{print $1}'`
# O2=`echo $FILE2 | awk -F. '{print $1}'`
# OUTPUT=$O1"_"$O2
# gnuplot -persist -e "file1='$PREFIX$FILE1'" \
#         -e "file2='$PREFIX$FILE2'" \
#         -e "output_file='$OUTPUT'" accuracy-rounds-two.gnu 

# ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
# rm $OUTPUT".ps"

# ###########

# FILE1="moe_atk1_75.txt"
# FILE2="avg_atk1_75.txt"
# O1=`echo $FILE1 | awk -F. '{print $1}'`
# O2=`echo $FILE2 | awk -F. '{print $1}'`
# OUTPUT=$O1"_"$O2
# gnuplot -persist -e "file1='$PREFIX$FILE1'" \
#         -e "file2='$PREFIX$FILE2'" \
#         -e "output_file='$OUTPUT'" accuracy-rounds-two.gnu 

# ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
# rm $OUTPUT".ps"

##################################################################







##################################################################

FILE1="moe_atk1_25_acc_new.txt"
FILE2="avg_atk1_25_acc_new.txt"
O1=`echo $FILE1 | awk -F. '{print $1}'`
O2=`echo $FILE2 | awk -F. '{print $1}'`
OUTPUT=$O1"_"$O2
gnuplot -persist -e "file1='$PREFIX$FILE1'" \
        -e "file2='$PREFIX$FILE2'" \
        -e "output_file='$OUTPUT'" accuracy-rounds-two.gnu 

ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
rm $OUTPUT".ps"

# ###########

FILE1="moe_atk1_50_acc_new.txt"
FILE2="avg_atk1_50_acc_new.txt"
O1=`echo $FILE1 | awk -F. '{print $1}'`
O2=`echo $FILE2 | awk -F. '{print $1}'`
OUTPUT=$O1"_"$O2
gnuplot -persist -e "file1='$PREFIX$FILE1'" \
        -e "file2='$PREFIX$FILE2'" \
        -e "output_file='$OUTPUT'" accuracy-rounds-two.gnu 

ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
rm $OUTPUT".ps"

###########

FILE1="moe_atk1_75_acc_new.txt"
FILE2="avg_atk1_75_acc_new.txt"
O1=`echo $FILE1 | awk -F. '{print $1}'`
O2=`echo $FILE2 | awk -F. '{print $1}'`
OUTPUT=$O1"_"$O2
gnuplot -persist -e "file1='$PREFIX$FILE1'" \
        -e "file2='$PREFIX$FILE2'" \
        -e "output_file='$OUTPUT'" accuracy-rounds-two.gnu 

ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
rm $OUTPUT".ps"


# ############ ############ ###########

FILE1="moe_atk1_25_loss_new.txt"
FILE2="avg_atk1_25_loss_new.txt"
O1=`echo $FILE1 | awk -F. '{print $1}'`
O2=`echo $FILE2 | awk -F. '{print $1}'`
OUTPUT=$O1"_"$O2
gnuplot -persist -e "file1='$PREFIX$FILE1'" \
        -e "file2='$PREFIX$FILE2'" \
        -e "output_file='$OUTPUT'" loss-rounds-two.gnu 

ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
rm $OUTPUT".ps"

###########

FILE1="moe_atk1_50_loss_new.txt"
FILE2="avg_atk1_50_loss_new.txt"
O1=`echo $FILE1 | awk -F. '{print $1}'`
O2=`echo $FILE2 | awk -F. '{print $1}'`
OUTPUT=$O1"_"$O2
gnuplot -persist -e "file1='$PREFIX$FILE1'" \
        -e "file2='$PREFIX$FILE2'" \
        -e "output_file='$OUTPUT'" loss-rounds-two.gnu 

ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
rm $OUTPUT".ps"

###########

FILE1="moe_atk1_75_loss_new.txt"
FILE2="avg_atk1_75_loss_new.txt"
O1=`echo $FILE1 | awk -F. '{print $1}'`
O2=`echo $FILE2 | awk -F. '{print $1}'`
OUTPUT=$O1"_"$O2
gnuplot -persist -e "file1='$PREFIX$FILE1'" \
        -e "file2='$PREFIX$FILE2'" \
        -e "output_file='$OUTPUT'" loss-rounds-two.gnu 

ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
rm $OUTPUT".ps"

##################################################################
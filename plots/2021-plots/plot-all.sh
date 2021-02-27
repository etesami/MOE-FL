#!/bin/bash
PREFIX_ACC="~/data/accuracy/"
PREFIX_LOSS="~/data/train_loss/"

# FILE1="moe_niid_atk1_san_331.txt"
# FILE2="avg_niid_atk1_san_332.txt"

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






##################################################################

# FILE1="moe_niid_no_attack_san_452.txt"
# FILE2="fedavg_niid_no_attack_san_330.txt"
# O1=`echo $FILE1 | awk -F. '{print $1}'`
# O2=`echo $FILE2 | awk -F. '{print $1}'`
# OUTPUT=$O1"_"$O2
# gnuplot -persist -e "file1='$PREFIX_ACC$FILE1'" \
#         -e "file2='$PREFIX_ACC$FILE2'" \
#         -e "output_file='$OUTPUT'" accuracy-rounds-two.gnu 

# ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
# rm $OUTPUT".ps"


# FILE1="moe_iid_no_attack_san_555.txt"
# FILE2="avg_iid_no_attack_san_556.txt"
# O1=`echo $FILE1 | awk -F. '{print $1}'`
# O2=`echo $FILE2 | awk -F. '{print $1}'`
# OUTPUT=$O1"_"$O2
# gnuplot -persist -e "file1='$PREFIX_ACC$FILE1'" \
#         -e "file2='$PREFIX_ACC$FILE2'" \
#         -e "output_file='$OUTPUT'" accuracy-rounds-two.gnu 

# ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
# rm $OUTPUT".ps"

#################################################################










# PURE and Not PURE
# Non IID
##################################################################

# FILE1="moe_atk1_25_acc.txt"
# FILE2="moe_atk1_25_acc_niid_npure.txt"
# FILE3="avg_atk1_25_acc.txt"
# O1=`echo $FILE1 | awk -F. '{print $1}'`
# O2=`echo $FILE2 | awk -F. '{print $1}'`
# O3=`echo $FILE3 | awk -F. '{print $1}'`
# OUTPUT=$O1"_"$O2"_"$O3
# gnuplot -persist -e "file1='$PREFIX_ACC$FILE1'" \
#         -e "file2='$PREFIX_ACC$FILE2'" \
#         -e "file3='$PREFIX_ACC$FILE3'" \
#         -e "output_file='$OUTPUT'" accuracy-rounds-three.gnu 

# ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
# rm $OUTPUT".ps"

# # # ###########

# FILE1="moe_atk1_50_acc.txt"
# FILE2="moe_atk1_50_acc_niid_npure.txt"
# FILE3="avg_atk1_50_acc.txt"
# O1=`echo $FILE1 | awk -F. '{print $1}'`
# O2=`echo $FILE2 | awk -F. '{print $1}'`
# O3=`echo $FILE3 | awk -F. '{print $1}'`
# OUTPUT=$O1"_"$O2"_"$O3
# gnuplot -persist -e "file1='$PREFIX_ACC$FILE1'" \
#         -e "file2='$PREFIX_ACC$FILE2'" \
#         -e "file3='$PREFIX_ACC$FILE3'" \
#         -e "output_file='$OUTPUT'" accuracy-rounds-three.gnu 

# ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
# rm $OUTPUT".ps"

# # ###########

# FILE1="moe_atk1_75_acc.txt"
# FILE2="moe_atk1_75_acc_niid_npure.txt"
# FILE3="avg_atk1_75_acc.txt"
# O1=`echo $FILE1 | awk -F. '{print $1}'`
# O2=`echo $FILE2 | awk -F. '{print $1}'`
# O3=`echo $FILE3 | awk -F. '{print $1}'`
# OUTPUT=$O1"_"$O2"_"$O3
# gnuplot -persist -e "file1='$PREFIX_ACC$FILE1'" \
#         -e "file2='$PREFIX_ACC$FILE2'" \
#         -e "file3='$PREFIX_ACC$FILE3'" \
#         -e "output_file='$OUTPUT'" accuracy-rounds-three.gnu 

# ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
# rm $OUTPUT".ps"


# # # ############ ############ ###########

# FILE1="moe_atk1_25_loss.txt"
# FILE2="moe_atk1_25_loss_niid_npure.txt"
# FILE3="avg_atk1_25_loss.txt"
# O1=`echo $FILE1 | awk -F. '{print $1}'`
# O2=`echo $FILE2 | awk -F. '{print $1}'`
# O3=`echo $FILE3 | awk -F. '{print $1}'`
# OUTPUT=$O1"_"$O2"_"$O3
# gnuplot -persist -e "file1='$PREFIX_LOSS$FILE1'" \
#         -e "file2='$PREFIX_LOSS$FILE2'" \
#         -e "file3='$PREFIX_LOSS$FILE3'" \
#         -e "output_file='$OUTPUT'" loss-rounds-three.gnu 

# ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
# rm $OUTPUT".ps"

# ###########

# FILE1="moe_atk1_50_loss.txt"
# FILE2="moe_atk1_50_loss_niid_npure.txt"
# FILE3="avg_atk1_50_loss.txt"
# O1=`echo $FILE1 | awk -F. '{print $1}'`
# O2=`echo $FILE2 | awk -F. '{print $1}'`
# O3=`echo $FILE3 | awk -F. '{print $1}'`
# OUTPUT=$O1"_"$O2"_"$O3
# gnuplot -persist -e "file1='$PREFIX_LOSS$FILE1'" \
#         -e "file2='$PREFIX_LOSS$FILE2'" \
#         -e "file3='$PREFIX_LOSS$FILE3'" \
#         -e "output_file='$OUTPUT'" loss-rounds-three.gnu 

# ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
# rm $OUTPUT".ps"

# ###########

# FILE1="moe_atk1_75_loss.txt"
# FILE2="moe_atk1_75_loss_niid_npure.txt"
# FILE3="avg_atk1_75_loss.txt"
# O1=`echo $FILE1 | awk -F. '{print $1}'`
# O2=`echo $FILE2 | awk -F. '{print $1}'`
# O3=`echo $FILE3 | awk -F. '{print $1}'`
# OUTPUT=$O1"_"$O2"_"$O3
# gnuplot -persist -e "file1='$PREFIX_LOSS$FILE1'" \
#         -e "file2='$PREFIX_LOSS$FILE2'" \
#         -e "file3='$PREFIX_LOSS$FILE3'" \
#         -e "output_file='$OUTPUT'" loss-rounds-three.gnu 

# ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
# rm $OUTPUT".ps"
# 
##################################################################







# PURE and Not PURE
# IID 
##################################################################

FILE1="moe_atk1_25_acc_iid.txt"
FILE2="moe_atk1_25_acc_iid_npure.txt"
FILE3="avg_atk1_25_acc_iid.txt"
O1=`echo $FILE1 | awk -F. '{print $1}'`
O2=`echo $FILE2 | awk -F. '{print $1}'`
O3=`echo $FILE3 | awk -F. '{print $1}'`
OUTPUT=$O1"_"$O2"_"$O3
gnuplot -persist -e "file1='$PREFIX_ACC$FILE1'" \
        -e "file2='$PREFIX_ACC$FILE2'" \
        -e "file3='$PREFIX_ACC$FILE3'" \
        -e "output_file='$OUTPUT'" accuracy-rounds-three.gnu 

ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
rm $OUTPUT".ps"

# # ###########

FILE1="moe_atk1_50_acc_iid.txt"
FILE2="moe_atk1_50_acc_iid_npure.txt"
FILE3="avg_atk1_50_acc_iid.txt"
O1=`echo $FILE1 | awk -F. '{print $1}'`
O2=`echo $FILE2 | awk -F. '{print $1}'`
O3=`echo $FILE3 | awk -F. '{print $1}'`
OUTPUT=$O1"_"$O2"_"$O3
gnuplot -persist -e "file1='$PREFIX_ACC$FILE1'" \
        -e "file2='$PREFIX_ACC$FILE2'" \
        -e "file3='$PREFIX_ACC$FILE3'" \
        -e "output_file='$OUTPUT'" accuracy-rounds-three.gnu 

ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
rm $OUTPUT".ps"

# ###########

FILE1="moe_atk1_75_acc_iid.txt"
FILE2="moe_atk1_75_acc_iid_npure.txt"
FILE3="avg_atk1_75_acc_iid.txt"
O1=`echo $FILE1 | awk -F. '{print $1}'`
O2=`echo $FILE2 | awk -F. '{print $1}'`
O3=`echo $FILE3 | awk -F. '{print $1}'`
OUTPUT=$O1"_"$O2"_"$O3
gnuplot -persist -e "file1='$PREFIX_ACC$FILE1'" \
        -e "file2='$PREFIX_ACC$FILE2'" \
        -e "file3='$PREFIX_ACC$FILE3'" \
        -e "output_file='$OUTPUT'" accuracy-rounds-three.gnu 

ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
rm $OUTPUT".ps"


# # ############ ############ ###########

FILE1="moe_atk1_25_loss_iid.txt"
FILE2="moe_atk1_25_loss_iid_npure.txt"
FILE3="avg_atk1_25_loss_iid.txt"
O1=`echo $FILE1 | awk -F. '{print $1}'`
O2=`echo $FILE2 | awk -F. '{print $1}'`
O3=`echo $FILE3 | awk -F. '{print $1}'`
OUTPUT=$O1"_"$O2"_"$O3
gnuplot -persist -e "file1='$PREFIX_LOSS$FILE1'" \
        -e "file2='$PREFIX_LOSS$FILE2'" \
        -e "file3='$PREFIX_LOSS$FILE3'" \
        -e "output_file='$OUTPUT'" loss-rounds-three.gnu 

ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
rm $OUTPUT".ps"

# ###########

FILE1="moe_atk1_50_loss_iid.txt"
FILE2="moe_atk1_50_loss_iid_npure.txt"
FILE3="avg_atk1_50_loss_iid.txt"
O1=`echo $FILE1 | awk -F. '{print $1}'`
O2=`echo $FILE2 | awk -F. '{print $1}'`
O3=`echo $FILE3 | awk -F. '{print $1}'`
OUTPUT=$O1"_"$O2"_"$O3
gnuplot -persist -e "file1='$PREFIX_LOSS$FILE1'" \
        -e "file2='$PREFIX_LOSS$FILE2'" \
        -e "file3='$PREFIX_LOSS$FILE3'" \
        -e "output_file='$OUTPUT'" loss-rounds-three.gnu 

ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
rm $OUTPUT".ps"

# ###########

FILE1="moe_atk1_75_loss_iid.txt"
FILE2="moe_atk1_75_loss_iid_npure.txt"
FILE3="avg_atk1_75_loss_iid.txt"
O1=`echo $FILE1 | awk -F. '{print $1}'`
O2=`echo $FILE2 | awk -F. '{print $1}'`
O3=`echo $FILE3 | awk -F. '{print $1}'`
OUTPUT=$O1"_"$O2"_"$O3
gnuplot -persist -e "file1='$PREFIX_LOSS$FILE1'" \
        -e "file2='$PREFIX_LOSS$FILE2'" \
        -e "file3='$PREFIX_LOSS$FILE3'" \
        -e "output_file='$OUTPUT'" loss-rounds-three.gnu 

ps2pdf -dAutoRotatePages=/None -dEPSCrop $OUTPUT".ps"
rm $OUTPUT".ps"

##################################################################










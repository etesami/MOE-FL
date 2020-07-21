#!/bin/bash

DIR="../data_tmp/"
PRFX1="01_avg"
PRFX2="01_opt"
PRFX3="02_attk1_avg"
PRFX4="03_attk1_opt"

PRFX5="04_attk2_avg"
PRFX6="05_attk2_opt"


# FILE=$DIR$PRFX6"80_100_weights"
# TITLE='Attack 1, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE
FILE=$DIR$PRFX6"80_100_test"
echo $FILE && ./plot.sh test $FILE $FILE

# FILE=$DIR$PRFX4"80_train_server"
# echo $FILE && ./plot.sh trainserv $FILE $FILE



# PER="80"
# FILE=$DIR$PRFX4$PER"_weights"
# TITLE='Attack 1, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE
# FILE=$DIR$PRFX4$PER"_train"
# echo $FILE && ./plot.sh train $FILE

# FILE=$DIR$PRFX2"train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX2"train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX2"test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX2"weights"
# echo $FILE && ./plot-weight.sh $FILE
# echo 


# PER="40"
# FILE=$DIR$PRFX4$PER"_weights"
# TITLE='Attack 1, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# PER="50"
# FILE=$DIR$PRFX4$PER"_weights"
# TITLE='Attack 1, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# PER="60"
# FILE=$DIR$PRFX4$PER"_weights"
# TITLE='Attack 1, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# PER="80"
# FILE=$DIR$PRFX4$PER"_weights"
# TITLE='Attack 1, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# #### -----

# PER="20"
# DATA="20"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# DATA="40"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# DATA="60"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# DATA="80"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# #### -----

# PER="40"
# DATA="20"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# DATA="40"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# DATA="60"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# DATA="80"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# #### -----

# PER="50"
# DATA="20"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# DATA="40"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# DATA="60"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# DATA="80"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# #### -----

# PER="60"
# DATA="20"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# DATA="40"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# DATA="60"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# DATA="80"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# #### -----

# PER="80"
# DATA="20"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# DATA="40"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# DATA="60"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE

# DATA="80"
# FILE=$DIR$PRFX6$PER"_"$DATA"_weights"
# TITLE='Attack 2, Workers Weights (Proposed Approach)\n'$PER'% Malicious Workers \\& '$DATA'% Data Alteration'
# echo $FILE && ./plot-weight.sh $FILE "$TITLE" $FILE








############################### old plots

# FILE=$DIR$PRFX1"train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX1"train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX1"test"
# echo $FILE && ./plot.sh test $FILE
# echo 

# FILE=$DIR$PRFX2"train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX2"train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX2"test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX2"weights"
# echo $FILE && ./plot-weight.sh $FILE
# echo 

# FILE=$DIR$PRFX3"train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX3"train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX3"test"
# echo $FILE && ./plot.sh test $FILE
# echo

#######################
#######################
#######################

# FILE=$DIR$PRFX4"train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX4"train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX4"test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX4"weights"
# echo $FILE && ./plot-weight.sh $FILE

# PER="20"
# FILE=$DIR$PRFX5$PER"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX5$PER"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX5$PER"_test"
# echo $FILE && ./plot.sh test $FILE

# PER="40"
# FILE=$DIR$PRFX5$PER"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX5$PER"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX5$PER"_test"
# echo $FILE && ./plot.sh test $FILE

# PER="60"
# FILE=$DIR$PRFX5$PER"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX5$PER"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX5$PER"_test"
# echo $FILE && ./plot.sh test $FILE

# PER="80"
# FILE=$DIR$PRFX5$PER"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX5$PER"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX5$PER"_test"
# echo $FILE && ./plot.sh test $FILE

#######################
#######################
#######################

# PER="20"
# FILE=$DIR$PRFX6$PER"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX6$PER"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX6$PER"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX6$PER"_weights"
# echo $FILE && ./plot-weight.sh $FILE

# PER="40"
# FILE=$DIR$PRFX6$PER"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX6$PER"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX6$PER"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX6$PER"_weights"
# echo $FILE && ./plot-weight.sh $FILE

# PER="60"
# FILE=$DIR$PRFX6$PER"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX6$PER"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX6$PER"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX6$PER"_weights"
# echo $FILE && ./plot-weight.sh $FILE

# PER="80"
# FILE=$DIR$PRFX6$PER"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX6$PER"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX6$PER"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX6$PER"_weights"
# echo $FILE && ./plot-weight.sh $FILE

#######################
#######################
#######################

# PER="20"
# PER_DATA="20"
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# PER_DATA="40"
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# PER_DATA="60"
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# PER_DATA="80"
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE

# PER="40"
# PER_DATA="20"
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# PER_DATA="40"
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# PER_DATA="60"
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# PER_DATA="80"
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE

# PER="60"
# PER_DATA="20"
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# PER_DATA="40"
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# PER_DATA="60"
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# PER_DATA="80"
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE

# PER="80"
# PER_DATA="20"
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# PER_DATA="40"
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# PER_DATA="60"
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# PER_DATA="80"
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX7$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE

#######################
#######################
#######################

# PER="20"
# PER_DATA="20"
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_weights"
# echo $FILE && ./plot-weight.sh $FILE
# PER_DATA="40"
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_weights"
# echo $FILE && ./plot-weight.sh $FILE
# PER_DATA="60"
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_weights"
# echo $FILE && ./plot-weight.sh $FILE
# PER_DATA="80"
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_weights"
# echo $FILE && ./plot-weight.sh $FILE

# PER="40"
# PER_DATA="20"
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_weights"
# echo $FILE && ./plot-weight.sh $FILE
# PER_DATA="40"
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_weights"
# echo $FILE && ./plot-weight.sh $FILE
# PER_DATA="60"
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_weights"
# echo $FILE && ./plot-weight.sh $FILE
# PER_DATA="80"
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_weights"
# echo $FILE && ./plot-weight.sh $FILE

# PER="60"
# PER_DATA="20"
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_weights"
# echo $FILE && ./plot-weight.sh $FILE
# PER_DATA="40"
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_weights"
# echo $FILE && ./plot-weight.sh $FILE
# PER_DATA="60"
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_weights"
# echo $FILE && ./plot-weight.sh $FILE
# PER_DATA="80"
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_weights"
# echo $FILE && ./plot-weight.sh $FILE

# PER="80"
# PER_DATA="20"
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_weights"
# echo $FILE && ./plot-weight.sh $FILE
# PER_DATA="40"
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_weights"
# echo $FILE && ./plot-weight.sh $FILE
# PER_DATA="60"
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_weights"
# echo $FILE && ./plot-weight.sh $FILE
# PER_DATA="80"
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX8$PER"_"$PER_DATA"_weights"
# echo $FILE && ./plot-weight.sh $FILE

#######################
#######################
#######################

# PER="80"
# PER_DATA="80"
# FILE=$DIR$PRFX9$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX9$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX9$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE

# FILE=$DIR$PRFX10$PER"_"$PER_DATA"_train"
# echo $FILE && ./plot.sh train $FILE
# FILE=$DIR$PRFX10$PER"_"$PER_DATA"_train_server"
# echo $FILE && ./plot.sh trainserv $FILE
# FILE=$DIR$PRFX10$PER"_"$PER_DATA"_test"
# echo $FILE && ./plot.sh test $FILE
# FILE=$DIR$PRFX10$PER"_"$PER_DATA"_weights"
# echo $FILE && ./plot-weight.sh $FILE
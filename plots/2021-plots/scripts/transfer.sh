#!/bin/bash



################################
# ./transfer-files.sh accuracy agent-5 20210204_230310_niid_attack_opt_SAN_331 moe_niid_atk1_san_331.txt
# ./transfer-files.sh accuracy agent-5 20210204_234235_niid_avg_atk1_SAN_332 avg_niid_atk1_san_332.txt
# ################################
# ./transfer-files.sh accuracy agent-5 20210204_230310_niid_attack_opt_SAN_331 moe_niid_atk1_5perc_san_331.txt
# ./transfer-files.sh accuracy agent-2 20210205_024716_niid_opt_atk1_SAN_335 moe_niid_atk1_15perc_san_335.txt
################################
# ./transfer-files.sh accuracy agent-2 20210204_234855_niid_opt_atk2_SAN_334 moe_niid_atk2_san_334.txt
# ./transfer-files.sh accuracy agent-3 20210204_234454_niid_avg_atk2_SAN_333 avg_niid_atk2_san_333.txt
################################
# ./transfer-files.sh accuracy agent-3 20210204_200925_rerun_SAN324_SAN330 fedavg_niid_no_attack_san_330.txt
################################

# ./transfer-files.sh accuracy agent-2 20210218_030959_moe_atk1_niid_SAN_353 moe_atk1_50.txt
# ./transfer-files.sh accuracy agent-2 20210218_031042_avg_atk1_niid_SAN_354 avg_atk1_50.txt

# ./transfer-files.sh accuracy agent-3 20210218_033243_moe_atk1_niid_SAN_357 moe_atk1_75.txt
# ./transfer-files.sh accuracy agent-3 20210218_033308_avg_atk1_niid_SAN_358 avg_atk1_75.txt

# ./transfer-files.sh accuracy agent-5 20210218_033412_moe_atk1_niid_SAN_359 moe_atk1_25.txt
# ./transfer-files.sh accuracy agent-5 20210218_033431_avg_atk1_niid_SAN_360 avg_atk1_25.txt

# ################################
# ./transfer-files.sh accuracy agent-6 20210220_213013_moe_atk1_niid_SAN_459 moe_atk1_50_acc_new.txt
# ./transfer-files.sh accuracy agent-6 20210220_213640_avg_atk1_niid_SAN_461 avg_atk1_50_acc_new.txt

# ./transfer-files.sh accuracy agent-7 20210220_220822_moe_atk1_niid_SAN_462 moe_atk1_75_acc_new.txt
# ./transfer-files.sh accuracy agent-7 20210220_220953_moe_atk1_niid_SAN_463 avg_atk1_75_acc_new.txt

# ./transfer-files.sh accuracy agent-8 20210220_224535_moe_atk1_niid_SAN_465 moe_atk1_25_acc_new.txt
# ./transfer-files.sh accuracy agent-8 20210220_224633_avg_atk1_niid_SAN_466 avg_atk1_25_acc_new.txt

./transfer-files.sh train_loss agent-6 20210220_213013_moe_atk1_niid_SAN_459 moe_atk1_50_loss_new.txt
./transfer-files.sh train_loss agent-6 20210220_213640_avg_atk1_niid_SAN_461 avg_atk1_50_loss_new.txt

./transfer-files.sh train_loss agent-7 20210220_220822_moe_atk1_niid_SAN_462 moe_atk1_75_loss_new.txt
./transfer-files.sh train_loss agent-7 20210220_220953_moe_atk1_niid_SAN_463 avg_atk1_75_loss_new.txt

./transfer-files.sh train_loss agent-8 20210220_224535_moe_atk1_niid_SAN_465 moe_atk1_25_loss_new.txt
./transfer-files.sh train_loss agent-8 20210220_224633_avg_atk1_niid_SAN_466 avg_atk1_25_loss_new.txt
################################
#!/bin/bash



# ###############################
# ./transfer-files.sh accuracy agent-5 20210204_230310_niid_attack_opt_SAN_331 moe_niid_atk1_san_331.txt
# ./transfer-files.sh accuracy agent-5 20210204_234235_niid_avg_atk1_SAN_332 avg_niid_atk1_san_332.txt
# ###############################
# ./transfer-files.sh accuracy agent-2 20210204_234855_niid_opt_atk2_SAN_334 moe_niid_atk2_san_334.txt
# ./transfer-files.sh accuracy agent-3 20210204_234454_niid_avg_atk2_SAN_333 avg_niid_atk2_san_333.txt
# ###############################
# No attack version
./transfer-files.sh accuracy agent-3 20210204_200925_rerun_SAN324_SAN330 fedavg_niid_no_attack_san_330.txt
./transfer-files.sh accuracy agent-3 20210220_174623_moe_niid_no_atk_SAN_452 moe_niid_no_attack_san_452.txt
# ###############################

# ##########
# # PURE
# ##########


# # NEW
# # Non IID
# ./transfer-files.sh accuracy agent-6 20210220_213013_moe_atk1_niid_SAN_459 moe_atk1_50_acc.txt
# ./transfer-files.sh accuracy agent-6 20210220_213640_avg_atk1_niid_SAN_461 avg_atk1_50_acc.txt

# ./transfer-files.sh accuracy agent-7 20210220_220822_moe_atk1_niid_SAN_462 moe_atk1_75_acc.txt
# ./transfer-files.sh accuracy agent-7 20210220_220953_moe_atk1_niid_SAN_463 avg_atk1_75_acc.txt

# ./transfer-files.sh accuracy agent-8 20210220_224535_moe_atk1_niid_SAN_465 moe_atk1_25_acc.txt
# ./transfer-files.sh accuracy agent-8 20210220_224633_avg_atk1_niid_SAN_466 avg_atk1_25_acc.txt

# ./transfer-files.sh train_loss agent-6 20210220_213013_moe_atk1_niid_SAN_459 moe_atk1_50_loss.txt
# ./transfer-files.sh train_loss agent-6 20210220_213640_avg_atk1_niid_SAN_461 avg_atk1_50_loss.txt

# ./transfer-files.sh train_loss agent-7 20210220_220822_moe_atk1_niid_SAN_462 moe_atk1_75_loss.txt
# ./transfer-files.sh train_loss agent-7 20210220_220953_moe_atk1_niid_SAN_463 avg_atk1_75_loss.txt

# ./transfer-files.sh train_loss agent-8 20210220_224535_moe_atk1_niid_SAN_465 moe_atk1_25_loss.txt
# ./transfer-files.sh train_loss agent-8 20210220_224633_avg_atk1_niid_SAN_466 avg_atk1_25_loss.txt
# # ################################

# # IID
# ./transfer-files.sh accuracy agent-9 20210221_051154_opt_atk1_iid_SAN_484 moe_atk1_25_acc_iid.txt
# ./transfer-files.sh accuracy agent-9 20210221_051218_avg_atk1_iid_SAN_485 avg_atk1_25_acc_iid.txt

# ./transfer-files.sh accuracy agent-9 20210221_051702_moe_atk1_iid_SAN_487 moe_atk1_50_acc_iid.txt
# ./transfer-files.sh accuracy agent-9 20210221_051859_avg_atk1_iid_SAN_488 avg_atk1_50_acc_iid.txt

# ./transfer-files.sh accuracy agent-8 20210221_053912_moe_atk1_iid_SAN_491 moe_atk1_75_acc_iid.txt
# ./transfer-files.sh accuracy agent-7 20210221_054108_avg_atk1_iid_SAN_492 avg_atk1_75_acc_iid.txt

# ./transfer-files.sh train_loss agent-9 20210221_051154_opt_atk1_iid_SAN_484 moe_atk1_25_loss_iid.txt
# ./transfer-files.sh train_loss agent-9 20210221_051218_avg_atk1_iid_SAN_485 avg_atk1_25_loss_iid.txt

# ./transfer-files.sh train_loss agent-9 20210221_051702_moe_atk1_iid_SAN_487 moe_atk1_50_loss_iid.txt
# ./transfer-files.sh train_loss agent-9 20210221_051859_avg_atk1_iid_SAN_488 avg_atk1_50_loss_iid.txt

# ./transfer-files.sh train_loss agent-8 20210221_053912_moe_atk1_iid_SAN_491 moe_atk1_75_loss_iid.txt
# ./transfer-files.sh train_loss agent-7 20210221_054108_avg_atk1_iid_SAN_492 avg_atk1_75_loss_iid.txt

# ###############################







# ##########
# Not PURE # Only Non IID
# ##########
# ###############################
# ./transfer-files.sh accuracy agent-10 20210221_191701_moe_atk1_niid_npure_SAN_545 moe_atk1_25_acc_niid_npure.txt

# ./transfer-files.sh accuracy agent-10 20210221_191822_moe_atk1_niid_npure_SAN_546 moe_atk1_50_acc_niid_npure.txt

# ./transfer-files.sh accuracy agent-10 20210221_191846_moe_atk1_niid_npure_SAN_547 moe_atk1_75_acc_niid_npure.txt

# ./transfer-files.sh accuracy agent-11 20210221_193141_moe_atk1_iid_npure_SAN_549 moe_atk1_25_acc_iid_npure.txt

# ./transfer-files.sh accuracy agent-11 20210221_193230_moe_atk1_iid_npure_SAN_550 moe_atk1_50_acc_iid_npure.txt

# ./transfer-files.sh accuracy agent-11 20210221_193301_moe_atk1_iid_npure_SAN_551 moe_atk1_75_acc_iid_npure.txt



# ./transfer-files.sh train_loss agent-10 20210221_191701_moe_atk1_niid_npure_SAN_545 moe_atk1_25_loss_niid_npure.txt

# ./transfer-files.sh train_loss agent-10 20210221_191822_moe_atk1_niid_npure_SAN_546 moe_atk1_50_loss_niid_npure.txt

# ./transfer-files.sh train_loss agent-10 20210221_191846_moe_atk1_niid_npure_SAN_547 moe_atk1_75_loss_niid_npure.txt

# ./transfer-files.sh train_loss agent-11 20210221_193141_moe_atk1_iid_npure_SAN_549 moe_atk1_25_loss_iid_npure.txt

# ./transfer-files.sh train_loss agent-11 20210221_193230_moe_atk1_iid_npure_SAN_550 moe_atk1_50_loss_iid_npure.txt

# ./transfer-files.sh train_loss agent-11 20210221_193301_moe_atk1_iid_npure_SAN_551 moe_atk1_75_loss_iid_npure.txt
# ###############################


# ###############################
# ./transfer-files.sh accuracy agent-2 20210221_210026_moe_noatk_iid_SAN_555 moe_iid_no_attack_san_555.txt
# ./transfer-files.sh accuracy agent-2 20210221_210128_avg_noatk_iid_SAN_556 avg_iid_no_attack_san_556.txt
##############################












######################################################################
#### Retrive weights & attackers list
######################################################################



##########
# PURE
##########
# # # ################################

# NEW
# Non IID
# ./transfer-files.sh opt_weights agent-6 20210220_213013_moe_atk1_niid_SAN_459 moe_atk1_50_acc.txt
# ./transfer-files.sh attackers agent-6 20210220_213013_moe_atk1_niid_SAN_459 moe_atk1_50_acc.txt

# ./transfer-files.sh opt_weights agent-7 20210220_220822_moe_atk1_niid_SAN_462 moe_atk1_75_acc.txt
# ./transfer-files.sh attackers agent-7 20210220_220822_moe_atk1_niid_SAN_462 moe_atk1_75_acc.txt

# ./transfer-files.sh opt_weights agent-8 20210220_224535_moe_atk1_niid_SAN_465 moe_atk1_25_acc.txt
# ./transfer-files.sh attackers agent-8 20210220_224535_moe_atk1_niid_SAN_465 moe_atk1_25_acc.txt

# ./transfer-files.sh opt_weights agent-6 20210220_213013_moe_atk1_niid_SAN_459 moe_atk1_50_loss.txt
# ./transfer-files.sh attackers agent-6 20210220_213013_moe_atk1_niid_SAN_459 moe_atk1_50_loss.txt

# ./transfer-files.sh opt_weights agent-7 20210220_220822_moe_atk1_niid_SAN_462 moe_atk1_75_loss.txt
# ./transfer-files.sh attackers agent-7 20210220_220822_moe_atk1_niid_SAN_462 moe_atk1_75_loss.txt

# ./transfer-files.sh opt_weights agent-8 20210220_224535_moe_atk1_niid_SAN_465 moe_atk1_25_loss.txt
# ./transfer-files.sh attackers agent-8 20210220_224535_moe_atk1_niid_SAN_465 moe_atk1_25_loss.txt
# # ################################

# # IID
# ./transfer-files.sh opt_weights agent-9 20210221_051154_opt_atk1_iid_SAN_484 moe_atk1_25_acc_iid.txt
# ./transfer-files.sh attackers agent-9 20210221_051154_opt_atk1_iid_SAN_484 moe_atk1_25_acc_iid.txt

# ./transfer-files.sh opt_weights agent-9 20210221_051702_moe_atk1_iid_SAN_487 moe_atk1_50_acc_iid.txt
# ./transfer-files.sh attackers agent-9 20210221_051702_moe_atk1_iid_SAN_487 moe_atk1_50_acc_iid.txt

# ./transfer-files.sh opt_weights agent-8 20210221_053912_moe_atk1_iid_SAN_491 moe_atk1_75_acc_iid.txt
# ./transfer-files.sh attackers agent-8 20210221_053912_moe_atk1_iid_SAN_491 moe_atk1_75_acc_iid.txt

# ./transfer-files.sh opt_weights agent-9 20210221_051154_opt_atk1_iid_SAN_484 moe_atk1_25_loss_iid.txt
# ./transfer-files.sh attackers agent-9 20210221_051154_opt_atk1_iid_SAN_484 moe_atk1_25_loss_iid.txt

# ./transfer-files.sh opt_weights agent-9 20210221_051702_moe_atk1_iid_SAN_487 moe_atk1_50_loss_iid.txt
# ./transfer-files.sh attackers agent-9 20210221_051702_moe_atk1_iid_SAN_487 moe_atk1_50_loss_iid.txt

# ./transfer-files.sh opt_weights agent-8 20210221_053912_moe_atk1_iid_SAN_491 moe_atk1_75_loss_iid.txt
# ./transfer-files.sh attackers agent-8 20210221_053912_moe_atk1_iid_SAN_491 moe_atk1_75_loss_iid.txt

# ###############################







# ##########
# # Not PURE # Only Non IID
# ##########
# ###############################
# ./transfer-files.sh opt_weights agent-10 20210221_191701_moe_atk1_niid_npure_SAN_545 moe_atk1_25_acc_niid_npure.txt
# ./transfer-files.sh attackers agent-10 20210221_191701_moe_atk1_niid_npure_SAN_545 moe_atk1_25_acc_niid_npure.txt

# ./transfer-files.sh opt_weights agent-10 20210221_191822_moe_atk1_niid_npure_SAN_546 moe_atk1_50_acc_niid_npure.txt
# ./transfer-files.sh attackers agent-10 20210221_191822_moe_atk1_niid_npure_SAN_546 moe_atk1_50_acc_niid_npure.txt

# ./transfer-files.sh opt_weights agent-10 20210221_191846_moe_atk1_niid_npure_SAN_547 moe_atk1_75_acc_niid_npure.txt
# ./transfer-files.sh attackers agent-10 20210221_191846_moe_atk1_niid_npure_SAN_547 moe_atk1_75_acc_niid_npure.txt

# ./transfer-files.sh opt_weights agent-11 20210221_193141_moe_atk1_iid_npure_SAN_549 moe_atk1_25_acc_iid_npure.txt
# ./transfer-files.sh attackers agent-11 20210221_193141_moe_atk1_iid_npure_SAN_549 moe_atk1_25_acc_iid_npure.txt

# ./transfer-files.sh opt_weights agent-11 20210221_193230_moe_atk1_iid_npure_SAN_550 moe_atk1_50_acc_iid_npure.txt
# ./transfer-files.sh attackers agent-11 20210221_193230_moe_atk1_iid_npure_SAN_550 moe_atk1_50_acc_iid_npure.txt

# ./transfer-files.sh opt_weights agent-11 20210221_193301_moe_atk1_iid_npure_SAN_551 moe_atk1_75_acc_iid_npure.txt
# ./transfer-files.sh attackers agent-11 20210221_193301_moe_atk1_iid_npure_SAN_551 moe_atk1_75_acc_iid_npure.txt



# ./transfer-files.sh opt_weights agent-10 20210221_191701_moe_atk1_niid_npure_SAN_545 moe_atk1_25_loss_niid_npure.txt
# ./transfer-files.sh attackers agent-10 20210221_191701_moe_atk1_niid_npure_SAN_545 moe_atk1_25_loss_niid_npure.txt

# ./transfer-files.sh opt_weights agent-10 20210221_191822_moe_atk1_niid_npure_SAN_546 moe_atk1_50_loss_niid_npure.txt
# ./transfer-files.sh attackers agent-10 20210221_191822_moe_atk1_niid_npure_SAN_546 moe_atk1_50_loss_niid_npure.txt

# ./transfer-files.sh opt_weights agent-10 20210221_191846_moe_atk1_niid_npure_SAN_547 moe_atk1_75_loss_niid_npure.txt
# ./transfer-files.sh attackers agent-10 20210221_191846_moe_atk1_niid_npure_SAN_547 moe_atk1_75_loss_niid_npure.txt

# ./transfer-files.sh opt_weights agent-11 20210221_193141_moe_atk1_iid_npure_SAN_549 moe_atk1_25_loss_iid_npure.txt
# ./transfer-files.sh attackers agent-11 20210221_193141_moe_atk1_iid_npure_SAN_549 moe_atk1_25_loss_iid_npure.txt

# ./transfer-files.sh opt_weights agent-11 20210221_193230_moe_atk1_iid_npure_SAN_550 moe_atk1_50_loss_iid_npure.txt
# ./transfer-files.sh attackers agent-11 20210221_193230_moe_atk1_iid_npure_SAN_550 moe_atk1_50_loss_iid_npure.txt

# ./transfer-files.sh opt_weights agent-11 20210221_193301_moe_atk1_iid_npure_SAN_551 moe_atk1_75_loss_iid_npure.txt
# ./transfer-files.sh attackers agent-11 20210221_193301_moe_atk1_iid_npure_SAN_551 moe_atk1_75_loss_iid_npure.txt
###############################


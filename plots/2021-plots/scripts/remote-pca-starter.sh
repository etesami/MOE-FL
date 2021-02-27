#!/usr/bin/env bash

help="Usage: 
    ./remote-pca-starter.sh"
eval "$(~/ehsan/.docopts -A args -h "$help" : "$@")"

# AGENT="agent-6"
# echo $AGENT
# ssh $AGENT <<'ENDSSH'
# ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# p 20210220_213013_moe_atk1_niid_SAN_459 /tmp/pca_niid_p_moe_50 0
# ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# p 20210220_213013_moe_atk1_niid_SAN_459 /tmp/pca_niid_p_moe_50 50
# ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# p 20210220_213013_moe_atk1_niid_SAN_459 /tmp/pca_niid_p_moe_50 100
# ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# p 20210220_213013_moe_atk1_niid_SAN_459 /tmp/pca_niid_p_moe_50 150
# ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# p 20210220_213013_moe_atk1_niid_SAN_459 /tmp/pca_niid_p_moe_50 200
# ENDSSH
# rsync -arz $AGENT:/tmp/pca_niid_p_moe_50 ~/ehsan/data_pca


AGENT="agent-5"
echo $AGENT
# ssh $AGENT <<'ENDSSH'
# ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# p 20210220_213640_avg_atk1_niid_SAN_461 /tmp/pca_niid_p_avg_50 0
# ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# p 20210220_213640_avg_atk1_niid_SAN_461 /tmp/pca_niid_p_avg_50 50
# ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# p 20210220_213640_avg_atk1_niid_SAN_461 /tmp/pca_niid_p_avg_50 100
# ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# p 20210220_213640_avg_atk1_niid_SAN_461 /tmp/pca_niid_p_avg_50 150
# ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# p 20210220_213640_avg_atk1_niid_SAN_461 /tmp/pca_niid_p_avg_50 200
# ENDSSH
# rsync -arz $AGENT:/tmp/pca_niid_p_avg_50 ~/ehsan/data_pca


# AGENT="agent-5"
# echo $AGENT
# ssh $AGENT <<'ENDSSH'
# ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# p 20210221_051859_avg_atk1_iid_SAN_488 /tmp/pca_iid_p_avg_50 0
# ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# p 20210221_051859_avg_atk1_iid_SAN_488 /tmp/pca_iid_p_avg_50 50
# ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# p 20210221_051859_avg_atk1_iid_SAN_488 /tmp/pca_iid_p_avg_50 100
# ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# p 20210221_051859_avg_atk1_iid_SAN_488 /tmp/pca_iid_p_avg_50 150
# ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# p 20210221_051859_avg_atk1_iid_SAN_488 /tmp/pca_iid_p_avg_50 200
# ENDSSH
# rsync -arz $AGENT:/tmp/pca_iid_p_avg_50 ~/ehsan/data_pca




# AGENT="agent-5"
# # # echo $AGENT
# # # ssh $AGENT <<'ENDSSH'
# # # ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# # # iid_p 20210225_201945_pca_iid_avg_atk8_SAN_867 /tmp/pca_iid_avg_8
# # # ENDSSH
# # rsync -arz $AGENT:/tmp/pca_iid_avg_8 ~/ehsan/data_pca

# AGENT="agent-5"
# # # echo $AGENT
# # # ssh $AGENT <<'ENDSSH'
# # # ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# # # iid_p 20210225_202024_pca_iid_avg_atk8_SAN_868 /tmp/pca_iid_avg_15
# # # ENDSSH
# # rsync -arz $AGENT:/tmp/pca_iid_avg_15 ~/ehsan/data_pca

# AGENT="agent-5"
# # # echo $AGENT
# # # ssh $AGENT <<'ENDSSH'
# # # ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# # # iid_p 20210225_202102_pca_iid_avg_atk8_SAN_869 /tmp/pca_iid_avg_23
# # # ENDSSH
# # rsync -arz $AGENT:/tmp/pca_iid_avg_23 ~/ehsan/data_pca



# AGENT="agent-5"
# # # echo $AGENT
# # # ssh $AGENT <<'ENDSSH'
# # # ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# # # iid_p 20210225_210630_pca_iid_avg_atk8_SAN_874 /tmp/pca_iid_moe_8
# # # ENDSSH
# # rsync -arz $AGENT:/tmp/pca_iid_moe_8 ~/ehsan/data_pca

# AGENT="agent-5"
# # # echo $AGENT
# # # ssh $AGENT <<'ENDSSH'
# # # ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# # # iid_p 20210225_210740_pca_iid_moe_atk8_SAN_875 /tmp/pca_iid_moe_15
# # # ENDSSH
# # rsync -arz $AGENT:/tmp/pca_iid_moe_15 ~/ehsan/data_pca




# AGENT="agent-6"
# # # echo $AGENT
# # # ssh $AGENT <<'ENDSSH'
# # # ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# # # iid_p 20210225_202552_pca_iid_avg_atk8_SAN_872 /tmp/pca_iid_moe_23
# # # ENDSSH
# # rsync -arz $AGENT:/tmp/pca_iid_moe_23 ~/ehsan/data_pca

# AGENT="agent-6"
# # echo $AGENT
# # ssh $AGENT <<'ENDSSH'
# # ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# # niid_p 20210225_211746_pca_niid_avg_atk_SAN_877 /tmp/pca_niid_avg_8
# # ENDSSH
# # rsync -arz $AGENT:/tmp/pca_niid_avg_8 ~/ehsan/data_pca

# AGENT="agent-6"
# echo $AGENT
# ssh $AGENT <<'ENDSSH'
# ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# niid_p 20210225_212005_pca_niid_avg_atk_SAN_878 /tmp/pca_niid_avg_15
# ENDSSH
# rsync -arz $AGENT:/tmp/pca_niid_avg_15 ~/ehsan/data_pca



# AGENT="agent-7"
# # # echo $AGENT
# # # ssh $AGENT <<'ENDSSH'
# # # ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# # # niid_p 20210225_212528_pca_niid_avg_atk_SAN_880 /tmp/pca_niid_avg_23
# # # ENDSSH
# # rsync -arz $AGENT:/tmp/pca_niid_avg_23 ~/ehsan/data_pca

# AGENT="agent-7"
# # # echo $AGENT
# # # ssh $AGENT <<'ENDSSH'
# # # ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# # # niid_p 20210225_212807_pca_niid_moe_atk_SAN_881 /tmp/pca_niid_moe_8
# # # ENDSSH
# # rsync -arz $AGENT:/tmp/pca_niid_moe_8 ~/ehsan/data_pca

# AGENT="agent-7"
# # echo $AGENT
# # ssh $AGENT <<'ENDSSH'
# # ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# # niid_p 20210225_212931_pca_niid_moe_atk_SAN_882 /tmp/pca_niid_moe_15
# # ENDSSH
# # rsync -arz $AGENT:/tmp/pca_niid_moe_15 ~/ehsan/data_pca

# AGENT="agent-7"
# # echo $AGENT
# # ssh $AGENT <<'ENDSSH'
# # ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# # niid_p 20210225_213031_pca_niid_moe_atk_SAN_883 /tmp/pca_niid_moe_23
# # ENDSSH
# # rsync -arz $AGENT:/tmp/pca_niid_moe_23 ~/ehsan/data_pca



# AGENT="agent-12"
# # echo $AGENT
# # # ssh $AGENT <<'ENDSSH'
# # # ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# # # niid_np 20210225_214620_pca_niid_np_moe_atk_SAN_884 /tmp/pca_niid_np_moe_8
# # # ENDSSH
# # rsync -arz $AGENT:/tmp/pca_niid_np_moe_8 ~/ehsan/data_pca

# AGENT="agent-12"
# # echo $AGENT
# # ssh $AGENT <<'ENDSSH'
# # ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# # niid_np 20210225_214716_pca_niid_np_moe_atk_SAN_885 /tmp/pca_niid_np_moe_15
# # ENDSSH
# # rsync -arz $AGENT:/tmp/pca_niid_np_moe_15 ~/ehsan/data_pca

# AGENT="agent-12"
# # echo $AGENT
# # ssh $AGENT <<'ENDSSH'
# # ~/ehsan/FederatedLearning/plots/2021-plots/scripts/remote-pca.sh \
# # niid_np 20210225_214811_pca_niid_np_moe_atk_SAN_886 /tmp/pca_niid_np_moe_23
# # ENDSSH
# # rsync -arz $AGENT:/tmp/pca_niid_np_moe_23 ~/ehsan/data_pca



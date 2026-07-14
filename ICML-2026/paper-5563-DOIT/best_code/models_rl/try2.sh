#!/bin/bash

# # Define the list of names
# names=("hopper-medium-replay-v2" "walker2d-medium-replay-v2" "halfcheetah-medium-replay-v2" "walker2d-medium-expert-v2" "hopper-medium-v2" "walker2d-medium-v2" "halfcheetah-medium-v2" "halfcheetah-medium-expert-v2" "hopper-medium-expert-v2")
# # Iterate through the list and copy the file to the new path
# for name in "${names[@]}"; do

#     # Copy the file to the new path
#     cp /workspace/home/huayu/git/Diffusion_Guide/models/${name}0cross_entrophy_alpha3_qa1_fix_dqldecay_16fa_ds15_cuthalf/critic_ckpt100.pth ./${name}0large_actor/
# done


# Define the list of names
names=("antmaze-umaze-v2" "antmaze-medium-play-v2" "antmaze-umaze-diverse-v2" "antmaze-medium-diverse-v2" "antmaze-large-diverse-v2" "antmaze-large-play-v2")
# Iterate through the list and copy the file to the new path
for name in "${names[@]}"; do

    # Copy the file to the new path
    cp /workspace/home/huayu/git/Diffusion_Guide/models/${name}0cross_entrophy_alpha3_qa20_fix_32fa_ds15/critic_ckpt100.pth ./${name}0large_actor/
done


#!/usr/bin/env bash
# This script generates all experiment configurations and instantiates
# these as W&B sweeps to be run somewhere else.

# Parameter grid setup:
#  - seed-repetitions (all)
#  - planning budget (mcts + smc + iterated smc)
#  - iterated-SMC ablations

TEST_MODULE="offline"

# Baseline PPO
#configs=$(ls configs/combined/experiments/ppo)
#for f in $configs; do
#  bash deploy/sweep/compile_sweep.sh \
#    -M $TEST_MODULE \
#    -P "configs/combined/experiments/ppo/$f" \
#    -S configs/sweeps/repetitions_30.yaml \
#    -p PPO_BASELINE \
#    -N "${f%.*}" \
#    $@
#  exit
#done

# Experiments for Baseline MCTS
#configs=$(ls configs/combined/experiments/mcts)
#for f in $configs; do
#  bash deploy/sweep/compile_sweep.sh \
#    -M $TEST_MODULE \
#    -P "configs/combined/experiments/mcts/$f" \
#    -S configs/sweeps/repetitions_5.yaml configs/sweeps/ablations/main/mcts.yaml \
#    -p MCTS_BASELINE_V2 \
#    -N "${f%.*}" \
#    $@
#done
#
#
## Experiments for Baseline SMC
#configs=$(ls configs/combined/experiments/tuning/smc)
#for f in $configs; do
#  bash deploy/sweep/compile_sweep.sh \
#    -M $TEST_MODULE \
#    -P "configs/combined/experiments/smc/$f" \
#    -S configs/sweeps/repetitions_10_other.yaml configs/sweeps/ablations/dsmc_tuning/smc.yaml \
#    -p TEST_SH_SMCTS \
#    -N "spo_${f%.*}" \
#    $@
#done


# Experiments for Our SMC
#configs=$(ls configs/combined/experiments/trt_smc)
#for f in $configs; do
#  bash deploy/sweep/compile_sweep.sh \
#    -M $TEST_MODULE \
#    -P "configs/combined/experiments/trt_smc/$f" \
#    -S configs/sweeps/repetitions_5.yaml configs/sweeps/ablations/main/trt_smc.yaml \
#    -p TRT_SMC_EXPERIMENTS_V2 \
#    -N "main_${f%.*}" \
#    $@
#done


# Ablations for our twisted SMC

# Particle-death fix
#configs=$(ls configs/combined/ablations/trt_smc/particle_death)
#for f in $configs; do
#
#  bash deploy/sweep/compile_sweep.sh \
#    -M $TEST_MODULE \
#    -P "configs/combined/ablations/trt_smc/particle_death/$f" \
#    -S configs/sweeps/repetitions_5.yaml configs/sweeps/ablations/particle_death/pd_smc.yaml \
#    -p TRT_SMC_EXPERIMENTS_V2 \
#    -N "pd_${f%.*}" \
#    $@
#
#done

configs=$(ls configs/combined/experiments/tuning/dsmc/)
for f in $configs; do

  bash deploy/sweep/compile_sweep.sh \
    -M $TEST_MODULE \
    -P "configs/combined/experiments/sh_smcts/$f" \
    -S configs/sweeps/repetitions_10_fourth.yaml configs/sweeps/ablations/dsmc_tuning/dsmc.yaml \
    -p TEST_SH_SMCTS \
    -N "dsmc_${f%.*}" \
    $@

done

#
## Twisting ablations
#configs=$(ls configs/combined/ablations/trt_smc/proposals)
#for f in $configs; do
#
#  # Hard twisting
#  bash deploy/sweep/compile_sweep.sh \
#    -M $TEST_MODULE \
#    -P "configs/combined/ablations/trt_smc/proposals/$f" \
#    -S configs/sweeps/repetitions_5.yaml configs/sweeps/ablations/proposals/trt_smc.yaml \
#    -p TRT_SMC_EXPERIMENTS_V2 \
#    -N "trt_${f%.*}" \
#    $@
#
#  # Soft twisting
#  bash deploy/sweep/compile_sweep.sh \
#    -M $TEST_MODULE \
#    -P "configs/combined/ablations/trt_smc/proposals/$f" \
#    -S configs/sweeps/repetitions_5.yaml configs/sweeps/ablations/proposals/twisted_smc.yaml \
#    -p TRT_SMC_EXPERIMENTS_V2 \
#    -N "twisted_${f%.*}" \
#    $@
#
#done
#
## Root-policy fix ablations
#configs=$(ls configs/combined/ablations/trt_smc/root_policy)
#for f in $configs; do
#
#  bash deploy/sweep/compile_sweep.sh \
#    -M $TEST_MODULE \
#    -P "configs/combined/ablations/trt_smc/root_policy/$f" \
#    -S configs/sweeps/repetitions_5.yaml configs/sweeps/ablations/root_policy/sp_smc.yaml \
#    -p TRT_SMC_EXPERIMENTS_V2 \
#    -N "rp_${f%.*}" \
#    $@
#
#done
#
#
## Value estimator ablations
#configs=$(ls configs/combined/ablations/trt_smc/value_estimator)
#for f in $configs; do
#
#  bash deploy/sweep/compile_sweep.sh \
#    -M $TEST_MODULE \
#    -P "configs/combined/ablations/trt_smc/value_estimator/$f" \
#    -S configs/sweeps/repetitions_5.yaml configs/sweeps/ablations/value_estimator/ve_smc.yaml \
#    -p TRT_SMC_EXPERIMENTS_V2 \
#    -N "ve_${f%.*}" \
#    $@
#
#done

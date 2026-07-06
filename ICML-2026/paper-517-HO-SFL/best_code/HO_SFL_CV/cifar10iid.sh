#!/bin/bash

seeds=$(seq 1 10)
device="cuda:0"

PROJECT="ICML_CIFAR10_IID"
LOGDIR="./logs/cifar10/iid/"

for seed in $seeds; do
  # FSL-SAGE
  python main_v2.py \
    --config-name fsl_sage \
    seed=$seed \
    system.device=$device \
    data.dataset=CIFAR10 \
    logging.project_name="$PROJECT" \
    logging.log_dir="$LOGDIR" 

  # MU-SplitFed
  python main_v2.py \
    --config-name mu_splitfed \
    seed=$seed \
    system.device=$device \
    data.dataset=CIFAR10 \
    logging.project_name="$PROJECT" \
    logging.log_dir="$LOGDIR" 
    

  # HO-SFL
  python main_v2.py \
    --config-name base \
    seed=$seed \
    system.device=$device \
    data.dataset=CIFAR10 \
    logging.project_name="$PROJECT" \
    logging.log_dir="$LOGDIR" \
    algo.zo_mu=1e-3 \
    algo.zo_p=5

  # # SFL
  python main_v2.py \
    --config-name sfl \
    seed=$seed \
    system.device=$device \
    data.dataset=CIFAR10 \
    logging.project_name="$PROJECT" \
    logging.log_dir="$LOGDIR"

  # SFL-ZO
  python main_v2.py \
    --config-name sfl_zo \
    seed=$seed \
    system.device=$device \
    data.dataset=CIFAR10 \
    logging.project_name="$PROJECT" \
    logging.log_dir="$LOGDIR"

done

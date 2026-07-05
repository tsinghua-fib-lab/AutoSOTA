#!/bin/bash

seeds=$(seq 1 10)
device="cuda:1"

PROJECT="ICML_CIFAR10_NONIID"
LOGDIR="./logs/cifar10/noniid/"

for seed in $seeds; do
  # FSL-SAGE
  python main_v2.py \
    --config-name fsl_sage \
    seed=$seed \
    system.device=$device \
    data.dataset=CIFAR10 \
    data.partition.algo=dirichlet \
    +data.partition.alpha=1 \
    logging.project_name="$PROJECT" \
    logging.log_dir="$LOGDIR"

  # python main_v2.py \
    --config-name mu_splitfed \
    seed=$seed \
    system.device=$device \
    data.dataset=CIFAR10 \
    data.partition.algo=dirichlet \
    +data.partition.alpha=1 \
    logging.project_name="$PROJECT" \
    logging.log_dir="$LOGDIR" \
    
  python main_v2.py \
    --config-name base \
    seed=$seed \
    system.device=$device \
    data.dataset=CIFAR10 \
    data.partition.algo=dirichlet \
    +data.partition.alpha=1 \
    logging.project_name="$PROJECT" \
    logging.log_dir="$LOGDIR" \
    algo.zo_mu=1e-3 \
    algo.zo_p=5

  # python main_v2.py \
    --config-name sfl \
    seed=$seed \
    system.device=$device \
    data.dataset=CIFAR10 \
    data.partition.algo=dirichlet \
    +data.partition.alpha=1 \
    logging.project_name="$PROJECT" \
    logging.log_dir="$LOGDIR"

  python main_v2.py \
    --config-name sfl_zo \
    seed=$seed \
    system.device=$device \
    data.dataset=CIFAR10 \
    data.partition.algo=dirichlet \
    +data.partition.alpha=1 \
    logging.project_name="$PROJECT" \
    logging.log_dir="$LOGDIR"
done

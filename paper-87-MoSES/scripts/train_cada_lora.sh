#!/bin/bash

PROJECT_ROOT=./


if [ ${1} == 50 ]; then

commands=(
    "CUDA_VISIBLE_DEVICES=0 python run.py \
    paths.root_dir=$PROJECT_ROOT \
    experiment=cada_lora_single_50.yaml \
    env.generator_params.variant_preset=ovrp \
    model.policy.lora_rank=${2} \
    model.policy.lora_use_linear=${3} \
    "


    "CUDA_VISIBLE_DEVICES=1 python run.py \
    paths.root_dir=$PROJECT_ROOT \
    experiment=cada_lora_single_50.yaml \
    env.generator_params.variant_preset=vrpl \
    model.policy.lora_rank=${2} \
    model.policy.lora_use_linear=${3} \
    "

    "CUDA_VISIBLE_DEVICES=2 python run.py \
    paths.root_dir=$PROJECT_ROOT \
    experiment=cada_lora_single_50.yaml \
    env.generator_params.variant_preset=vrpb \
    model.policy.lora_rank=${2} \
    model.policy.lora_use_linear=${3} \
    "

    "CUDA_VISIBLE_DEVICES=3 python run.py \
    paths.root_dir=$PROJECT_ROOT \
    experiment=cada_lora_single_50.yaml \
    env.generator_params.variant_preset=vrptw \
    model.policy.lora_rank=${2} \
    model.policy.lora_use_linear=${3} \
    "
)

elif [ ${1} == 100 ]; then

commands=(
    "CUDA_VISIBLE_DEVICES=0 python run.py \
    paths.root_dir=$PROJECT_ROOT \
    experiment=cada_lora_single_100.yaml \
    env.generator_params.variant_preset=ovrp \
    model.policy.lora_rank=${2} \
    model.policy.lora_use_linear=${3} \
    "

    "CUDA_VISIBLE_DEVICES=1 python run.py \
    paths.root_dir=$PROJECT_ROOT \
    experiment=cada_lora_single_100.yaml \
    env.generator_params.variant_preset=vrpl \
    model.policy.lora_rank=${2} \
    model.policy.lora_use_linear=${3} \
    "

    "CUDA_VISIBLE_DEVICES=2 python run.py \
    paths.root_dir=$PROJECT_ROOT \
    experiment=cada_lora_single_100.yaml \
    env.generator_params.variant_preset=vrpb \
    model.policy.lora_rank=${2} \
    model.policy.lora_use_linear=${3} \
    "

    "CUDA_VISIBLE_DEVICES=3 python run.py \
    paths.root_dir=$PROJECT_ROOT \
    experiment=cada_lora_single_100.yaml \
    env.generator_params.variant_preset=vrptw \
    model.policy.lora_rank=${2} \
    model.policy.lora_use_linear=${3} \
    "
)


fi



interval=2
for cmd in "${commands[@]}";
do
    eval "$cmd" &
    sleep "$interval"
done

wait



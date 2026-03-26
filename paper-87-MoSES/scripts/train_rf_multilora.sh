#!/bin/bash

PROJECT_ROOT=./

if [ ${1} == 50 ]; then

CUDA_VISIBLE_DEVICES=0 python run.py \
paths.root_dir=$PROJECT_ROOT \
experiment=rf_lora_cross_50.yaml \
env.generator_params.variant_preset=all \
model.policy.lora_rank=${5} \
model.policy.lora_alpha=1.0 \
model.policy.lora_act_func='softplus' \
model.policy.lora_use_trainable_layer=${2} \
model.policy.lora_use_dynamic_topK=${3} \
model.policy.lora_use_basis_variants=${4} \
model.policy.lora_use_basis_variants_as_input=${6} \
model.policy.lora_use_linear=${7} \


elif [ ${1} == 100 ]; then

CUDA_VISIBLE_DEVICES=0 python run.py \
paths.root_dir=$PROJECT_ROOT \
experiment=rf_lora_cross_100.yaml \
env.generator_params.variant_preset=all \
model.policy.lora_rank=${5} \
model.policy.lora_alpha=1.0 \
model.policy.lora_act_func='softplus' \
model.policy.lora_use_trainable_layer=${2} \
model.policy.lora_use_dynamic_topK=${3} \
model.policy.lora_use_basis_variants=${4} \
model.policy.lora_use_basis_variants_as_input=${6} \
model.policy.lora_use_linear=${7} \


fi


#!/bin/bash

PROJECT_ROOT=./



CUDA_VISIBLE_DEVICES=0 python test_vrplib.py \
--log_path $PROJECT_ROOT/logs \
--dataset_path $PROJECT_ROOT/data/vrplib \
--checkpoint $PROJECT_ROOT/pretrained_moses_model/rf/100/multilora_VR1route_softplus.ckpt \
--model_name rf_multilora \
--lora_act_func softplus \
--lora_use_trainable_layer 1 \
--lora_use_dynamic_topK 1 \
--lora_use_basis_variants 0 \
--lora_rank 32 32 32 32 32 \



CUDA_VISIBLE_DEVICES=0 python test_vrplib.py \
--log_path $PROJECT_ROOT/logs \
--dataset_path $PROJECT_ROOT/data/vrplib \
--checkpoint $PROJECT_ROOT/pretrained_moses_model/cada/100/multilora_VR1route_sigmoid.ckpt \
--model_name cada_multilora \
--lora_act_func sigmoid \
--lora_use_trainable_layer 1 \
--lora_use_dynamic_topK 1 \
--lora_use_basis_variants 0 \
--lora_rank 32 32 32 32 32 \



#!/bin/bash

PROJECT_ROOT=./



CUDA_VISIBLE_DEVICES=0 python test.py \
--log_path $PROJECT_ROOT/logs \
--dataset_path $PROJECT_ROOT/data \
--size 50 \
--model_name rf_multilora \
--lora_act_func softplus \
--lora_use_trainable_layer 1 \
--lora_use_dynamic_topK 0 \
--lora_use_basis_variants 0 \
--lora_rank 32 32 32 32 32 \
--checkpoint $PROJECT_ROOT/pretrained_moses_model/rf/50/multilora_denseroute_softplus.ckpt \




CUDA_VISIBLE_DEVICES=0 python test.py \
--log_path $PROJECT_ROOT/logs \
--dataset_path $PROJECT_ROOT/data \
--size 100 \
--model_name rf_multilora \
--lora_act_func softplus \
--lora_use_trainable_layer 1 \
--lora_use_dynamic_topK 0 \
--lora_use_basis_variants 0 \
--lora_rank 32 32 32 32 32 \
--checkpoint $PROJECT_ROOT/pretrained_moses_model/rf/100/multilora_denseroute_softplus.ckpt \



CUDA_VISIBLE_DEVICES=0 python test.py \
--log_path $PROJECT_ROOT/logs \
--dataset_path $PROJECT_ROOT/data \
--size 50 \
--model_name cada_multilora \
--lora_act_func sigmoid \
--lora_use_trainable_layer 1 \
--lora_use_dynamic_topK 0 \
--lora_use_basis_variants 0 \
--lora_rank 32 32 32 32 32 \
--checkpoint $PROJECT_ROOT/pretrained_moses_model/cada/50/multilora_denseroute_sigmoid.ckpt \




CUDA_VISIBLE_DEVICES=0 python test.py \
--log_path $PROJECT_ROOT/logs \
--dataset_path $PROJECT_ROOT/data \
--size 100 \
--model_name cada_multilora \
--lora_act_func sigmoid \
--lora_use_trainable_layer 1 \
--lora_use_dynamic_topK 0 \
--lora_use_basis_variants 0 \
--lora_rank 32 32 32 32 32 \
--checkpoint $PROJECT_ROOT/pretrained_moses_model/cada/100/multilora_denseroute_sigmoid.ckpt \




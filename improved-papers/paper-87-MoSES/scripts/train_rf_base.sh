#!/bin/bash

PROJECT_ROOT=./



CUDA_VISIBLE_DEVICES=0 python run.py \
paths.root_dir=$PROJECT_ROOT \
experiment=rf_base_${1}.yaml \
env.generator_params.variant_preset=cvrp \



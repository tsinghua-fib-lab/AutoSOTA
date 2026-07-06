#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python pta_runner.py     --config configs \
                                                --wandb-log \
                                                --datasets caltech101/dtd/eurosat/fgvc/oxford_flowers/oxford_pets/ucf101/stanford_cars/food101/sun397 \
                                                --backbone RN50
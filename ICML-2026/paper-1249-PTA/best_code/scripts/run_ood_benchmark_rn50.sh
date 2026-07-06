#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python pta_runner.py     --config configs \
                                                --wandb-log \
                                                --datasets I/V/R/S/A \
                                                --backbone RN50
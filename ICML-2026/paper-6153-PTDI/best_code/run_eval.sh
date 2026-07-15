#!/bin/bash

# ! check the seed

# EleutherAI/pythia-6.9b
# huggyllama/llama-7b
python eval.py \
        --dataset WikiMIA \
        --model_path EleutherAI/pythia-6.9b \
        --seed 42

        
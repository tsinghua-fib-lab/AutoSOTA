#!/bin/bash

#SBATCH --job-name=olmo-layer
#SBATCH --output=./logs/layer_%j.out
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
PYTHON_BIN="/home/xxxxxxxxx/anaconda3/bin/python"
SCRIPT="train_exp1_layerwise.py"
RESULTS_DIR="./results_layerwise"

mkdir -p $RESULTS_DIR/block
mkdir -p $RESULTS_DIR/induction
mkdir -p ./logs

echo ">>> Running Block Copy Experiments..."

BLOCK_ARGS="--task block --block_size 5 --vocab_size 100 --num_pairs 4 --steps 20000 --batch_size 256 --seq_len 128"


$PYTHON_BIN $SCRIPT \
    --output_dir $RESULTS_DIR/block \
    --run_id "baseline_block" \
    $BLOCK_ARGS


$PYTHON_BIN $SCRIPT \
    --output_dir $RESULTS_DIR/block \
    --run_id "uniform_0.5_block" \
    --use_scaled_rope --sigma 0.5 \
    $BLOCK_ARGS

$PYTHON_BIN $SCRIPT \
    --output_dir $RESULTS_DIR/block \
    --run_id "gradient_block" \
    --use_scaled_rope --sigma 0.5 \
    --rope_scaling_threshold 2 \
    $BLOCK_ARGS


echo ">>> Running Standard Induction Experiments..."

INDUCT_ARGS="--task standard --vocab_size 100 --num_pairs 4 --steps 10000 --batch_size 256 --seq_len 128"

$PYTHON_BIN $SCRIPT \
    --output_dir $RESULTS_DIR/induction \
    --run_id "baseline_induct" \
    $INDUCT_ARGS


$PYTHON_BIN $SCRIPT \
    --output_dir $RESULTS_DIR/induction \
    --run_id "gradient_induct" \
    --use_scaled_rope --sigma 0.5 \
    --rope_scaling_threshold 2 \
    $INDUCT_ARGS

echo "Layer-wise Experiments Finished."
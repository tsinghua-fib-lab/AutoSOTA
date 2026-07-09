#!/bin/bash -x

#SBATCH --job-name=olmo-exp1-block
#SBATCH --output=./logs/exp1-block_%j.out
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

export PYTHONPATH="/home/xxxxxxxxx/03-proj/PE/OLMo:$PYTHONPATH"
PYTHON_BIN="/home/xxxxxxxxx/anaconda3/bin/python"
SCRIPT="train_exp1_induction.py"
OUTPUT_DIR="./results_exp1/block" 

mkdir -p $OUTPUT_DIR
mkdir -p ./logs

echo "Starting Block Copy Experiment (Regional Attention Test)..."


COMMON_ARGS="--task block --block_size 5 --vocab_size 100 --num_pairs 4 --steps 30000 --batch_size 256 --seq_len 128"

declare -a EXP_SIGMAS=(12.0 15.0 20.0 50.0)        # (2.0 5.0 10.0)

for sigma in "${EXP_SIGMAS[@]}"; do
    run_id="scaled_block_sigma_${sigma}"
    
    echo "Running Scaled RoPE (Exp Decay) with Sigma: $sigma"
    
    $PYTHON_BIN $SCRIPT \
        --output_dir $OUTPUT_DIR \
        --run_id "$run_id" \
        --use_scaled_rope \
        --sigma $sigma \
        --decay_func "exp" \
        $COMMON_ARGS
done

echo "Block Experiment Finished. Data saved to $OUTPUT_DIR"
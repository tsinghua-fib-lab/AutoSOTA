#!/bin/bash

#SBATCH --job-name=exp1-full
#SBATCH --output=./logs/exp1_full_%j.out
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
PYTHON_BIN="/home/xxxxxxxxx/anaconda3/bin/python"

SCRIPT="train_exp1_full.py"  

CHECKPOINT_ROOT="/data/xxxxxxxxx/03-proj/PE/checkpoints_synthetic_full"
mkdir -p $CHECKPOINT_ROOT
mkdir -p ./logs

COMMON_ARGS="--vocab_size 50 --seq_len 64 --num_pairs 4 --steps 100000 --batch_size 64"

MODELS=("60M") #  "20M")

TASKS=("standard") # "block" standard)

SIGMAS=(700.0 100.0 10.0 500.0 250.0 1.0)  # 50.0 200.0 300.0 80.0) # 700.0 100.0 10.0 500.0 250.0 1.0)

THRESHOLDS=(2 3)


for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        
        echo "======================================================="
        echo ">>> Processing Model: $model | Task: $task"
        echo "======================================================="

        RUN_ID="exp1_${task}_${model}_baseline"
        OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}"
        
        echo ">>> Running Baseline..."
        $PYTHON_BIN $SCRIPT \
            --output_dir $OUTPUT_DIR \
            --run_id $RUN_ID \
            --model_size $model \
            --task_mode $task \
            $COMMON_ARGS \


        for sigma in "${SIGMAS[@]}"; do
            for thr in "${THRESHOLDS[@]}"; do
                
                if [ "$thr" -eq -1 ]; then
                    TYPE="uniform"
                else
                    TYPE="grad_thr${thr}"
                fi
                
                RUN_ID="exp1_${task}_${model}_${TYPE}_sig${sigma}"
                OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}"
                
                echo ">>> Running $TYPE (Sigma=$sigma, Thr=$thr)..."
                $PYTHON_BIN $SCRIPT \
                    --output_dir $OUTPUT_DIR \
                    --run_id $RUN_ID \
                    --model_size $model \
                    --task_mode $task \
                    --use_scaled_rope \
                    --sigma $sigma \
                    --rope_scaling_threshold $thr \
                    $COMMON_ARGS
            done
        done
        
    done
done

echo ">>> All exp1 full experiments finished."
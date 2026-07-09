#!/bin/bash
# Sequential experiment runner for paper 5267 SOTA
# Each experiment: train + eval, auto-score on completion
set -e

cd /repo
GPUS="${GPUS:-2,3}"
BASE_CMD="CUDA_VISIBLE_DEVICES=$GPUS python3 -u train_eval.py --dataset math10k --privacy dp --epsilon 6 --base_model /models/google_gemma-3-4b-pt --seed 42 --lora_r 16 --lora_alpha 16 --batch_size 64 --micro_batch_size 4 --steps 300 --lr 3e-4 --dp_max_grad_norm 1.0 --force_train --force_eval --no_resume --num_beams 4 --max_new_tokens 256"

run_exp() {
    local iter=$1; local idea=$2; local title=$3; local extra_args=$4
    local log="/tmp/iter${iter}_${idea}.log"
    echo "=== Iter $iter: $title ==="
    echo "Started: $(date)"
    echo "Log: $log"
    
    $BASE_CMD $extra_args 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    
    echo "Exit: $rc"
    echo "Finished: $(date)"
    
    if [ $rc -eq 0 ]; then
        bash /tmp/auto_score.sh "$log" "$iter" "$idea" "$title" "success" "Auto-scored experiment"
    else
        bash /tmp/auto_score.sh "$log" "$iter" "$idea" "$title" "failed" "Exit code: $rc"
    fi
}

# === Experiment Queue ===
# Uncomment to run specific experiments

# Iter 2: IDEA-08+09 combined (DataLoader seed + debias)
# run_exp 2 "IDEA-08+09" "DataLoader seed fix + dp_debias_second_moment" ""

# Iter 3: IDEA-05 geometry floor mode
# run_exp 3 "IDEA-05-geometry" "PRISM geometry floor mode" "--prism_floor_mode geometry"

# Iter 4: IDEA-02 noise decay
# run_exp 4 "IDEA-02" "Noise multiplier decay at step 200" "--noise_decay_enabled true --noise_decay_start_step 200 --noise_decay_factor 0.8"

# Iter 5: IDEA-05 floor factor tuning
# run_exp 5 "IDEA-05-factor025" "PRISM floor factor 0.25 geometry" "--prism_floor_mode geometry --prism_floor_factor 0.25"

# Iter 6: Combined best
# run_exp 6 "COMBINED" "Combined best settings" "--prism_floor_mode geometry --noise_decay_enabled true"

echo "All experiments completed."

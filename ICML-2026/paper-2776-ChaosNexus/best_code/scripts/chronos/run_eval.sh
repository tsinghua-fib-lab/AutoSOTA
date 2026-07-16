#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir="./checkpoints/t5-mini/run-0/checkpoint-final"

ulimit -n 99999

test_data_dirs=(
    ./data/test
)

# --- START: Modified section (replaces jq) ---
# Goal: Convert the bash array to a JSON array string.
# Example: (path1 path2) -> '["path1","path2"]'
test_data_dirs_json="[" # Start the JSON array string
num_dirs=${#test_data_dirs[@]} # Get the total number of directories
i=0
for dir in "${test_data_dirs[@]}"; do
    test_data_dirs_json+="\"$dir\"" # Add the directory path in quotes
    i=$((i + 1))
    if [ "$i" -lt "$num_dirs" ]; then
        test_data_dirs_json+="," # Add a comma if it's not the last element
    fi
done
test_data_dirs_json+="]" # Close the JSON array string
# --- END: Modified section ---
# run_name=chronos_mini_zeroshot # chronos mini zeroshot
run_name=chronos_t5_mini_ft-0 # newest chronos sft 300k iterations

# Set zero_shot flag based on whether "zeroshot" appears in run_name
if [[ "$run_name" == *"zeroshot"* ]]; then
    zero_shot_flag="true"
else
    zero_shot_flag="false"
fi

use_deterministic=false
model_dirname="chronos"
if [ "$use_deterministic" = false ]; then
    model_dirname="chronos_nondeterministic"
fi
echo "model_dirname: $model_dirname"

python scripts/chronos/evaluate.py \
    eval.checkpoint_path=$checkpoint_dir \
    eval.data_paths_lst=$test_data_dirs_json \
    eval.num_subdirs=null \
    eval.num_test_instances=6 \
    eval.num_samples=10 \
    eval.parallel_sample_reduction=mean \
    eval.window_style=sampled \
    eval.batch_size=32 \
    eval.chronos.deterministic=$use_deterministic \
    chronos.context_length=512 \
    eval.prediction_length=512 \
    eval.limit_prediction_length=false \
    eval.metrics_save_dir=./eval_results/${model_dirname}/CHRONOS-SFT/test_example \
    eval.metrics_fname=metrics \
    eval.overwrite=true \
    eval.device=cuda:6 \
    eval.save_predictions=false \
    eval.save_labels=false \
    eval.save_contexts=false \
    eval.chronos.zero_shot=$zero_shot_flag \
    eval.seed=99 \
    "$@"

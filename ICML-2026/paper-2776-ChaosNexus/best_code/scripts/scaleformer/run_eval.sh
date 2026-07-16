#!/bin/bash
main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir="./checkpoints/Nexus1.0-base/checkpoint-final"

ulimit -n 99999

# # scaling law runs
# run_names_scalinglaw=(
#     pft_chattn_mlm_sys10490_ic2-0
#     pft_chattn_mlm_sys656_ic32-0
#     pft_chattn_mlm_sys164_ic128-0
#     pft_chattn_mlm_sys5245_ic4-0
#     pft_chattn_mlm_sys1312_ic16-0
#     pft_chattn_mlm_sys328_ic64-0
#     pft_chattn_mlm_sys2623_ic8-0
#     pft_chattn_mlm_sys20k_ic1-0
# )
#
# # univariate with old dynamics embedding
# run_names_univariate_kernelemb_old=(
#     pft_emb_equal_param_univariate_from_scratch-0
#     pft_rff_univariate_pretrained-0
# )
#
# # univariate either without dynamics embedding or with the new poly one
# run_names_univariate=(
#     pft_noemb_equal_param_univariate_from_scratch-0
#     pft_vanilla_pretrained_correct-0
#     pft_equal_param_deeper_univariate_from_scratch_noemb-0
# )
#
# # multivariate without dynamics embedding
# run_names_multivariate=(
#     pft_chattn_noembed_pretrained_correct-0
#     pft_stand_chattn_noemb-0
#     pft_chattn_noemb_pretrained_chrope-0
# )
#
# # multivariate with the kernel embedding
# run_names_multivariate_kernelemb=(
#     pft_rff496_proj-0
#     pft_chattn_emb_w_poly-0
#     pft_chattn_fullemb_pretrained-0
# )
#
# # multivariate with linear attention polyfeats dynamics embedding
# run_names_multivariate_linattnpolyemb=(
#     pft_linattnpolyemb_from_scratch-0
# )
#
# run_names_new=(
#     pft_sft-0
# )
#
# run_names=(
#     # ${run_names_new[@]}
#     # ${run_names_scalinglaw[@]}
#     # ${run_names_univariate[@]}
#     # ${run_names_univariate_kernelemb_old[@]}
#     # ${run_names_multivariate[@]}
#     # ${run_names_multivariate_kernelemb_old[@]}
#     ${run_names_multivariate_kernelemb[@]}
#     # ${run_names_multivariate_linattnpolyemb[@]}
# )
#
# echo "run_names: ${run_names[@]}"

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

echo "test_data_dirs: $test_data_dirs_json"

python scripts/scaleformer/evaluate.py \
    eval.mode=predict \
    eval.checkpoint_path=$checkpoint_dir \
    eval.data_paths_lst=$test_data_dirs_json \
    eval.num_subdirs=null \
    eval.num_test_instances=6 \
    eval.window_style=sampled \
    eval.batch_size=128 \
    eval.context_length=512 \
    eval.prediction_length=512 \
    eval.limit_prediction_length=false \
    eval.metrics_save_dir=./eval_results/scaleformer/NEXUS-test/test_example \
    eval.metrics_fname=metrics \
    eval.overwrite=true \
    eval.device=cuda:6 \
    eval.save_labels=false \
    eval.save_predictions=false \
    eval.save_contexts=false \
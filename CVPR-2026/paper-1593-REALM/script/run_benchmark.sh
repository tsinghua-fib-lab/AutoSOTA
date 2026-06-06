#!/usr/bin/env bash
# set -euo pipefail

# 46
# 1) Cleanup old metrics files and result folders for scenes a00002 … a00069
# for i in $(seq -f "a%05g" 46 53); do
#   model_dir="output/benchmark_realm/$i"
  
#   # remove metrics json
#   for metric in metrics_gs_group.json metrics_realm.json; do
#     if [ -f "$model_dir/$metric" ]; then
#       rm "$model_dir/$metric"
#       echo "[INFO] Removed $model_dir/$metric"
#     fi
#   done

#   # remove old result directories
#   for dir in gs_group realm; do
#     if [ -d "$model_dir/$dir" ]; then
#       rm -rf "$model_dir/$dir"
#       echo "[INFO] Removed directory $model_dir/$dir"
#     fi
#   done
# done

#!/usr/bin/env bash
set -euo pipefail

# 2) Process each scene
for i in $(seq -f "a%05g" 33 35); do
  {
    set -e  # abort this block on any error

    data_folder="data/benchmark_realm/$i"
    model_dir="output/benchmark_realm/$i"
    json_path="$data_folder/object_dict.json"

    # If no prompts file, skip THIS SCENE (not exit the whole script!)
    if [ ! -f "$json_path" ]; then
      echo "[WARN] Scene $i: object_dict.json not found, skipping."
      continue    # <- jump to next 'i' in the for‑loop
    fi

    # Loop over each prompt; on first failure, jump to the outer catch
    while IFS="=" read -r name prompt; do
      name=${name#\"}; name=${name%\"}
      prompt=${prompt#\"}; prompt=${prompt%\"}

      echo "[INFO] Scene $i – object: $name"
      echo "[INFO] Prompt: $prompt"

      python reason_seg.py  \
        -m "$model_dir"      \
        --prompt "$prompt"   \
        --object "$name"     \
        --out_repo realm     \
        --itr_finetune 20

      python seg_gsgroup.py \
        -m "$model_dir"      \
        --prompt "$prompt"   \
        --object "$name"     \
        --out_repo gs_group
    done < <(jq -r 'to_entries[] | "\(.key)=\(.value)"' "$json_path")

    # If we got here, all prompts succeeded → run evaluations
    python script/eval_lerf_mask.py "$data_folder" "$model_dir" gs_group
    python script/eval_lerf_mask.py "$data_folder" "$model_dir" realm

    echo "[✓] Scene $i completed successfully."
  } || {
    # Any failure in the block lands here, so skip the rest of that scene
    echo "[ERROR] Scene $i failed, skipping to next."
    continue
  }
done

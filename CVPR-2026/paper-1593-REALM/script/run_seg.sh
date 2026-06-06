# list1='1 2 3 4 5 6 7 8 9 10'
# # list1='9 10'
# list='figurines'


# for j in $list1; do 

#     for i in $list; do
#     {
#         set -e  # abort this block on any error

#         data_folder="data/lerf/$i"
#         model_dir="output/rgb/lerf/$i"
#         json_path="$data_folder/object_dict.json"

#         # If no prompts file, skip THIS SCENE (not exit the whole script!)
#         if [ ! -f "$json_path" ]; then
#         echo "[WARN] Scene $i: object_dict.json not found, skipping."
#         continue    # <- jump to next 'i' in the for‑loop
#         fi

#         # Loop over each prompt; on first failure, jump to the outer catch
#         while IFS="=" read -r name prompt; do
#         name=${name#\"}; name=${name%\"}
#         prompt=${prompt#\"}; prompt=${prompt%\"}

#         echo "[INFO] Scene $i – object: $name"
#         echo "[INFO] Prompt: $prompt"

#         python reason_seg.py  \
#             -m "$model_dir"      \
#             --prompt "$prompt"   \
#             --object "$name"     \
#             --out_repo test     \
#             --itr_finetune 20

#         #   python seg_gsgroup.py \
#         #     -m "$model_dir"      \
#         #     --prompt "$prompt"   \
#         #     --object "$name"     \
#         #     --out_repo gs_group
#         done < <(jq -r 'to_entries[] | "\(.key)=\(.value)"' "$json_path")

#         # If we got here, all prompts succeeded → run evaluations
#         # python script/eval_lerf_mask.py "$data_folder" "$model_dir" gs_groupx
#         python script/eval_lerf_mask.py "$data_folder" "$model_dir" random_12_input_$j 

#         echo "[✓] Scene $i completed successfully."
#     } || {
#         # Any failure in the block lands here, so skip the rest of that scene
#         echo "[ERROR] Scene $i failed, skipping to next."
#         continue
#     }
#     done

# done

list1='1'
# list1='9 10'
list='figurines'


#!/usr/bin/env bash
set -u  

for j in $list1; do
  for i in $list; do
    echo "[SCENE] $i"

    data_folder="data/lerf/$i"
    model_dir="output/rgb/lerf/$i"
    json_path="$data_folder/object_dict.json"

    if [[ ! -f "$json_path" ]]; then
      echo "[WARN] Scene $i: object_dict.json not found, skipping."
      continue
    fi

    while IFS=$'\t' read -r name prompt; do

      name=${name#\"}; name=${name%\"}
      prompt=${prompt#\"}; prompt=${prompt%\"}

      echo "[INFO] Scene $i – object: $name"
      echo "[INFO] Prompt: $prompt"

      python reason_seg.py \
        -m "$model_dir" \
        --prompt "$prompt" \
        --object "$name" \
        --out_repo test \
        --itr_finetune 20 || {
          echo "[WARN] reason_seg.py failed on $i/$name"
          continue
        }

    done < <(jq -r 'to_entries[] | "\(.key)\t\(.value)"' "$json_path")

  done
done

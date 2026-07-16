#!/bin/bash
# ============================================================================
# Run a watermark-removal attack on the watermarked dataset.
#
# A single entry point for every attack: it rewrites the watermarked text,
# detects the watermark on the rewritten text, and computes TPR-at-fixed-FPR
# (all through pipeline/run_attack.py).
#
# Usage:
#   bash scripts/run_attack.sh <attack> [backend]
#
#   attack    BIRA | vanilla_paraphrasing | dipper-1 | dipper-2 | SIRA   (default: BIRA)
#   backend   hf | api   — only used by BIRA / vanilla_paraphrasing      (default: hf)
#               hf  = local model (Llama-3.1-8B-Instruct)
#               api = OpenAI gpt-4o-mini
#
# Examples:
#   bash scripts/run_attack.sh BIRA hf
#   bash scripts/run_attack.sh BIRA api
#   bash scripts/run_attack.sh vanilla_paraphrasing
#   bash scripts/run_attack.sh dipper-2
#   bash scripts/run_attack.sh SIRA
# ============================================================================

attack="${1:-BIRA}"
backend="${2:-hf}"

# ------------------------------- settings -----------------------------------
gpus=0                                          # GPU id(s); use "0,1" for large models (dipper / 70B)
algorithms=(SIR KGW Unigram UPV EWD DIP EXP)    # watermark schemes to attack
num_data=500                                    # samples per watermark scheme

input_path=./watermarked_dataset                # watermarked text (from generate_watermarked_text.sh)
result_save_dir=./experimental_results          # where attacked text + detection scores are saved

# Detectability settings. Each (rule, fpr) is evaluated in the same pass and saved next to the
# attack result: target_fpr -> TPR@FPR (one file per fpr in target_fprs); best -> best-threshold F1.
dataset_path=./dataset/c4/processed_c4.json
human_text_result_save_dir=./experimental_results_human_text
labels="TPR F1"
rules=(target_fpr best)
target_fprs=(0.01 0.1)
# ----------------------------------------------------------------------------

# Per-attack arguments. (BIRA's recommended beta is model-dependent:
# -4.0 for Llama-3.1-8B/70B, -11.0 for gpt-4o-mini.)
extra_args=()
case "$attack" in
  BIRA | vanilla_paraphrasing)
    case "$backend" in
      hf)  extra_args+=(--model_cfg_path ./model_config/llama3.1-8b.yaml --use_sampling); beta=-4.0  ;;
      api) extra_args+=(--model_cfg_path ./model_config/llama3.2-3b.yaml --api_model_name gpt4o-mini); beta=-11.0 ;;
      *)   echo "Unknown backend '$backend' (use: hf | api)"; exit 1 ;;
    esac
    extra_args+=(--backend "$backend")
    if [[ "$attack" == "BIRA" ]]; then
      extra_args+=(--beta "$beta" --percentile 50)
    fi
    ;;
  dipper-1 | dipper-2)
    # DIPPER paraphraser (dipper-paraphraser-xxl) is large — set gpus="0,1" above.
    extra_args+=(--model_name dipper)
    ;;
  SIRA)
    sira_model=meta-llama/Meta-Llama-3.1-8B-Instruct
    extra_args+=(--paraphrase_model_path "$sira_model"
                 --attack_model_path "$sira_model"
                 --threshold 30
                 --model_name "$(basename "$sira_model")")
    ;;
  *)
    echo "Unknown attack '$attack'."
    echo "Choose one of: BIRA | vanilla_paraphrasing | dipper-1 | dipper-2 | SIRA"
    exit 1
    ;;
esac

log_dir="scripts_logs/${attack}_${backend}"
mkdir -p "$log_dir"

echo "Attack: $attack | backend: $backend | gpus: $gpus | watermarks: ${algorithms[*]}"
for algorithm in "${algorithms[@]}"; do
  echo "▶  $algorithm"
  CUDA_VISIBLE_DEVICES=$gpus python pipeline/run_attack.py \
    --attack_algorithms "$attack" \
    --algorithms "$algorithm" \
    --num_data "$num_data" \
    --input_path "$input_path" \
    --result_save_dir "$result_save_dir" \
    --dataset_path "$dataset_path" \
    --human_text_result_save_dir "$human_text_result_save_dir" \
    --labels $labels \
    --rules "${rules[@]}" \
    --target_fprs "${target_fprs[@]}" \
    "${extra_args[@]}" 2>&1 | tee "$log_dir/${algorithm}.log"
done

echo "✅ Done. Attacked text + scores + TPR → $result_save_dir"

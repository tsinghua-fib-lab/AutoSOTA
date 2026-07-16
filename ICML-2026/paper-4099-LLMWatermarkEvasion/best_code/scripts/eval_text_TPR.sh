#!/bin/bash
# ============================================================================
# Re-compute TPR-at-fixed-FPR from already-saved attack detection scores.
#
# The attack run (scripts/run_attack.sh) already reports TPR; use this only to
# re-sweep target_fpr / rules WITHOUT re-running the (expensive) attack. Run it
# on the SAME attack (and backend, for BIRA / vanilla_paraphrasing) so the
# result paths line up.
#
# Usage:
#   bash scripts/eval_text_TPR.sh <attack> [backend]
#
#   attack    BIRA | vanilla_paraphrasing | dipper-1 | dipper-2 | SIRA   (default: BIRA)
#   backend   hf | api   — only used by BIRA / vanilla_paraphrasing      (default: hf)
# ============================================================================

attack="${1:-BIRA}"
backend="${2:-hf}"

gpu=0
algorithms=(SIR KGW Unigram UPV EWD DIP EXP)    # watermark schemes to evaluate
num_data=500
rules=(target_fpr best)                         # detectability rules: TPR@fixed-FPR and best-threshold F1
target_fprs=(0.01 0.1)                           # fixed FPRs at which to report TPR (for the target_fpr rule)
labels="TPR F1"

attack_result_save_dir=./experimental_results
human_text_result_save_dir=./experimental_results_human_text
no_attack_result_save_dir=./experimental_results_no_attack
dataset_path=./dataset/c4/processed_c4.json

# Per-attack model identity (must mirror scripts/run_attack.sh so the result
# paths written by the attack match the paths read here). BIRA's recommended
# beta is model-dependent: -4.0 for Llama-3.1-8B (hf), -11.0 for gpt-4o-mini (api).
beta=""
case "$attack" in
  BIRA | vanilla_paraphrasing)
    case "$backend" in
      hf)  model_cfg=./model_config/llama3.1-8b.yaml; beta=-4.0  ;;
      api) model_cfg=./model_config/gpt4o-mini.yaml;  beta=-11.0 ;;
      *)   echo "Unknown backend '$backend' (use: hf | api)"; exit 1 ;;
    esac
    ;;
  dipper-1 | dipper-2) model_cfg=./model_config/dipper.yaml   ;;
  SIRA)                model_cfg=./model_config/llama3.1-8b.yaml ;;
  *)
    echo "Unknown attack '$attack'."
    echo "Choose one of: BIRA | vanilla_paraphrasing | dipper-1 | dipper-2 | SIRA"
    exit 1
    ;;
esac

log_dir="scripts_logs/tpr_${attack}_${backend}"
mkdir -p "$log_dir"

args=(
  --model_cfg_path "$model_cfg"
  --dataset_path "$dataset_path"
  --attack_result_save_dir "$attack_result_save_dir"
  --human_text_result_save_dir "$human_text_result_save_dir"
  --no_attack_result_save_dir "$no_attack_result_save_dir"
  --algorithms "${algorithms[@]}"
  --attack_algorithms "$attack"
  --num_data "$num_data"
  --labels $labels
  --rules "${rules[@]}"
  --target_fprs "${target_fprs[@]}"
)
if [[ "$attack" == "BIRA" ]]; then
  args+=(--beta "$beta" --percentile 50)
fi

echo "Detectability | rules: ${rules[*]} | target_fprs: ${target_fprs[*]} | attack: $attack | backend: $backend | watermarks: ${algorithms[*]}"
CUDA_VISIBLE_DEVICES=$gpu python evaluation/assess_detectability.py "${args[@]}" \
  2>&1 | tee "$log_dir/tpr.log"

echo "✅ Done. TPR results written next to each attack-result JSON under $attack_result_save_dir."

#!/bin/bash
# ============================================================================
# Evaluate the TEXT QUALITY of a rewriting attack's outputs
# (how well the attacked text preserves the original, per quality metric).
#
# Run an attack first (scripts/run_attack.sh), then this on the SAME attack
# (and backend, for BIRA / vanilla_paraphrasing) so the result paths line up.
#
# Usage:
#   bash scripts/eval_text_quality.sh <attack> [backend]
#
#   attack    BIRA | vanilla_paraphrasing | dipper-1 | dipper-2 | SIRA   (default: BIRA)
#   backend   hf | api   — only used by BIRA / vanilla_paraphrasing      (default: hf)
# ============================================================================

attack="${1:-BIRA}"
backend="${2:-hf}"

gpu=0
algorithms=(SIR KGW Unigram UPV EWD DIP EXP)    # watermark schemes to evaluate
metrics=(nli self_bleu s-bert ppl)              # quality metrics (also available: llm_judge)
num_data=500
result_save_dir=./experimental_results

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

log_dir="scripts_logs/quality_${attack}_${backend}"
mkdir -p "$log_dir"

echo "Text-quality eval | attack: $attack | backend: $backend | metrics: ${metrics[*]}"
for algorithm in "${algorithms[@]}"; do
  for metric in "${metrics[@]}"; do
    echo "▶  $algorithm / $metric"
    args=(
      --algorithms "$algorithm"
      --model_cfg_path "$model_cfg"
      --attack_algorithms "$attack"
      --num_data "$num_data"
      --result_save_dir "$result_save_dir"
      --metric "$metric"
    )
    if [[ "$attack" == "BIRA" ]]; then
      args+=(--beta "$beta" --percentile 50)
    fi
    CUDA_VISIBLE_DEVICES=$gpu python -u evaluation/evaluate_quality.py "${args[@]}" \
      2>&1 | tee "$log_dir/${algorithm}_${metric}.log"
  done
done

echo "✅ Done. Quality results under $result_save_dir."

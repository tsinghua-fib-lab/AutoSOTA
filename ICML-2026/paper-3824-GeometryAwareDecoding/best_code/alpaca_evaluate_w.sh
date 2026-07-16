#!/usr/bin/env bash
set -euo pipefail

LAM=2.2  

MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  # "microsoft/phi-3-mini-4k-instruct"
  # "Qwen/Qwen2.5-3B"
)

TEMPS=(1.0 1.5 2.0)

for MODEL in "${MODELS[@]}"; do
  SAFE_MODEL=${MODEL//\//_}

  for T in "${TEMPS[@]}"; do

    beta_val=$(python3 - <<PY
lam = float("${LAM}")
T = float("${T}")
print(2.8)
PY
)

    OUTPUT="./teseet_${SAFE_MODEL}_T${T}.json"
    OUTDIR="./alpaca_results_w/${SAFE_MODEL}_T${T}"   
    mkdir -p "$OUTDIR"                             
    echo ">>> MODEL=${MODEL}, T=${T}, LAM=${LAM}, BETA=${beta_val}, OUTPUT=${OUTPUT}"

    python3 -u alpaca_generate_w.py \
      --save_address "$OUTPUT" \
      --model_name "$MODEL" \
      --max_new_tokens 1024 \
      --temperature "$T" \
      --do_sample \
      --lam "$LAM" \
      --beta "$beta_val"

      alpaca_eval \
      --annotators_config 'gpt4o' \
      --model_outputs "$OUTPUT" \
      --output_path "$OUTDIR" \
      --precomputed_leaderboard None


  done
done


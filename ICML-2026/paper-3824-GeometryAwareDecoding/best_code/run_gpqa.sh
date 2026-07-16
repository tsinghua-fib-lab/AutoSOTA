#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
MODELS=(
  "Qwen/Qwen2.5-3B"
  "meta-llama/Llama-3.1-8B-Instruct"
  "microsoft/phi-3-mini-4k-instruct"
)
# TASK="gsm8k_cot"
TASK="gpqa_main_generative_n_shot"
BATCH=1
FEWSHOT=8

# LIMIT_ARGS=(--limit 180)
LIMIT_ARGS=()

# required each invocation
: "${T:?set T (e.g., 1.0)}"
: "${m:?set m (e.g., 1200)}"

# grids
SEL_TEMPS=("${T}")
WARM_PS=(0.999)
BETAS=(2.8)
LAMBDAS=(2.2)
# ---------------------------------------------------------------

OUTDIR="temp"
mkdir -p "${OUTDIR}"

ts=$(date +"%Y%m%d-%H%M%S")
GEOM="${TOPW_GEOM_MODE:-NA}"   # geom_mode from env if set

for MODEL in "${MODELS[@]}"; do
  SAFE_MODEL=${MODEL//\//_}

  SUMMARY="${OUTDIR}/all_results_${SAFE_MODEL}_${ts}_T${T}_m${m}_geom${GEOM}.json"

  for Tsel in "${SEL_TEMPS[@]}"; do
    for wp in "${WARM_PS[@]}"; do
      for beta in "${BETAS[@]}"; do
        for lam in "${LAMBDAS[@]}"; do

          tag="${SAFE_MODEL}_T${T}_Tsel${Tsel}_m${m}_lam${lam}_wp${wp}_beta${beta}"
          out_json="${OUTDIR}/${tag}.json"

          echo "[RUN] ${tag} on ${DEVICE} -> ${out_json}"
          PYTHONUNBUFFERED=1 TOPW_PRINT_PARAMS=0 TOPW_PRINT_KEPT=0 TOPW_DEBUG_STEPS=0 \
          python3 -u -m lm_eval --model hf \
            --model_args "pretrained=${MODEL}" \
            --tasks "${TASK}" --device "${DEVICE}" \
            --batch_size "${BATCH}" --num_fewshot "${FEWSHOT}" \
            "${LIMIT_ARGS[@]}" \
            --gen_kwargs "do_sample=True,temperature=${T},selection_temperature=${Tsel},top_m=${m},lambda_geom=${lam},warm_p=${wp},beta=${beta},top_p=1.0,top_k=0,typical_p=1.0" \
            --output_path "${out_json}"

          # lm_eval may append a timestamp to out_json; match on prefix
          base="${out_json%.json}"

          shopt -s nullglob
          run_files=( "${base}"*.json )
          shopt -u nullglob

          if ((${#run_files[@]} > 0)); then
            run_json="${run_files[0]}"

            python3 - "${run_json}" "${MODEL}" "${T}" "${Tsel}" "${m}" "${lam}" "${wp}" "${beta}" "${SUMMARY}" << 'PY'
import json, sys, os

res_path, model, T, Tsel, m, lam, wp, beta, summary_path = sys.argv[1:10]

with open(res_path) as fh:
    d = json.load(fh)

results = d.get("results", {}) or {}

if isinstance(results, dict) and results:
    task_dict = next(iter(results.values()))
else:
    task_dict = {}

strict_val = strict_std = None
flex_val = flex_std = None

for k, v in task_dict.items():
    if "exact_match" in k and "strict-match" in k and "stderr" not in k:
        strict_val = v
    elif "exact_match_stderr" in k and "strict-match" in k:
        strict_std = v
    elif "exact_match" in k and "flexible-extract" in k and "stderr" not in k:
        flex_val = v
    elif "exact_match_stderr" in k and "flexible-extract" in k:
        flex_std = v

print(
    f"[RESULT] model={model} | T={T} Tsel={Tsel} m={m} lam={lam} wp={wp} beta={beta} | "
    f"strict={strict_val}±{strict_std} | flex={flex_val}±{flex_std}"
)

record = {
    "model": model,
    "T": float(T),
    "Tsel": float(Tsel),
    "m": int(m),
    "warm_p": float(wp),
    "lambda": float(lam),
    "beta": float(beta),

    "flexible-extract_value": flex_val,
    "flexible-extract_std": flex_std,
    "strict-match_value": strict_val,
    "strict-match_std": strict_std,
}

agg = []
if os.path.exists(summary_path):
    try:
        with open(summary_path) as fh:
            agg = json.load(fh)
        if not isinstance(agg, list):
            agg = []
    except Exception:
        agg = []

agg.append(record)

with open(summary_path, "w") as fh:
    json.dump(agg, fh, indent=2, sort_keys=True)
PY

            # delete per-run result file; keep only SUMMARY
            rm -f "${run_json}"
          else
            echo "[RESULT] ${tag} | no output file produced matching ${base}*.json"
          fi

        done
      done
    done
  done
done

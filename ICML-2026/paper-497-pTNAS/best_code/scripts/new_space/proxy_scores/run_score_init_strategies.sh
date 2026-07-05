#!/usr/bin/env bash
# Score all models with Xavier and LeCun init, max 30 parallel
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_ROOT="${PROJECT_ROOT}/datasets/nas_bench_tabular"
PYTHON="${PYTHON:-python}"
mkdir -p "${OUT_ROOT}/space_resdnn/proxy_score/init" "${OUT_ROOT}/space_blockmixed/proxy_score/init"

MAX_PARALLEL=10
INIT_METHODS=(xavier kaiming lecun)

# ResNet datasets
RESNET_DATASETS=(event-user-attendance avito-user-clicks avito-ad-ctr hm-user-churn)
# BlockMixed datasets
BM_DATASETS=(event-user-attendance avito-user-clicks avito-ad-ctr hm-user-churn)

GPU=0
running=0

for INIT in "${INIT_METHODS[@]}"; do
  # ResNet
  for DS in "${RESNET_DATASETS[@]}"; do
    CSV="${OUT_ROOT}/space_resdnn/proxy_score/init/score_${DS}_resnet_${INIT}.csv"
    bash -c "
      cd '${PROJECT_ROOT}'
      PYTHONPATH='${PROJECT_ROOT}/src:${PROJECT_ROOT}' '${PYTHON}' scripts/new_space/proxy_scores/score_init_strategies.py \
        --dataset ${DS} --space resnet --init_method ${INIT} \
        --device cuda:$((GPU % 8)) --output_csv '${CSV}'
    " > /dev/null 2>&1 &
    echo "[launch] resnet ${DS} ${INIT} gpu=$((GPU % 8))"
    GPU=$((GPU + 1))
    running=$((running + 1))
    if (( running >= MAX_PARALLEL )); then wait -n 2>/dev/null || true; running=$((running - 1)); fi
  done

  # BlockMixed
  for DS in "${BM_DATASETS[@]}"; do
    CSV="${OUT_ROOT}/space_blockmixed/proxy_score/init/score_${DS}_blockmixed_${INIT}.csv"
    bash -c "
      cd '${PROJECT_ROOT}'
      PYTHONPATH='${PROJECT_ROOT}/src:${PROJECT_ROOT}' '${PYTHON}' scripts/new_space/proxy_scores/score_init_strategies.py \
        --dataset ${DS} --space blockmixed --init_method ${INIT} \
        --device cuda:$((GPU % 8)) --output_csv '${CSV}'
    " > /dev/null 2>&1 &
    echo "[launch] blockmixed ${DS} ${INIT} gpu=$((GPU % 8))"
    GPU=$((GPU + 1))
    running=$((running + 1))
    if (( running >= MAX_PARALLEL )); then wait -n 2>/dev/null || true; running=$((running - 1)); fi
  done
done

echo "[wait] All launched..."
wait
echo "[done] Results in ${OUT_ROOT}/{space_resdnn,space_blockmixed}/proxy_score/init/"

#!/bin/sh
set -eu

ROOT=/home/dataset-assist-0/chenzhibin/s3f+autosota
cd "$ROOT"
PYTHON=/opt/conda/envs/s3f/bin/python
CKPT="$ROOT/ckpt/s2f_pretrain_output/ResidueTypePrediction/CATH/FusionNetwork/null/2026-09-05-09-56-16/model_best.pth"
export LD_LIBRARY_PATH="/opt/conda/envs/s3f/lib:/opt/conda/envs/s3f/cuda11/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_PATH=/opt/conda/envs/s3f/cuda11
export KEOPS_CACHE_FOLDER="$ROOT/keops_cache"
export MPLCONFIGDIR="$ROOT/full217_run/matplotlib_cache"
export S2F_INCLUDE_BINDING=1
export S2F_SCORING_ESM=ESM-1v-ens5
export S2F_ESM_WEIGHT=0.20
export S2F_WT_ESM_WEIGHT=0.30
export S2F_PSSM_WEIGHT=0.25
export S2F_PSSM_ADAPTIVE=1
export S2F_MULTI_WINDOW=1
export S2F_SITE_AGG_ALPHA=0
unset VIRTUAL_ENV
mkdir -p full217_run/logs "$KEOPS_CACHE_FOLDER" "$MPLCONFIGDIR"

RUN_TAG="full217_$(date +%Y%m%d_%H%M%S)"
printf '%s\n' "$RUN_TAG" > full217_run/RUN_TAG.txt
printf 'Started %s\n' "$(date -Is)"
printf 'RUN_TAG=%s\n' "$RUN_TAG"

for WAVE in 0 4; do
    PIDS=''
    for GPU in 0 1 2 3; do
        BATCH=$((WAVE + GPU))
        BATCH2=$(printf '%02d' "$BATCH")
        (
            export SLURM_JOB_ID="${RUN_TAG}_b${BATCH2}"
            "$PYTHON" autosota-util/scripts/s2f_eval.py \
                --subset full \
                --assay-file "full217_run/batches/batch_${BATCH2}.txt" \
                --gpus "$GPU" \
                --method "${RUN_TAG}_b${BATCH2}" \
                --ckpt "$CKPT"
        ) > "full217_run/logs/batch_${BATCH2}.log" 2>&1 &
        PIDS="$PIDS $!"
    done
    FAILED=0
    for PID in $PIDS; do
        wait "$PID" || FAILED=1
    done
    if [ "$FAILED" -ne 0 ]; then
        printf 'FAILED: wave %s; inspect full217_run/logs/\n' "$WAVE" >&2
        exit 1
    fi
    printf 'Completed wave %s at %s\n' "$WAVE" "$(date -Is)"
done

"$PYTHON" full217_run/collect_full217.py
printf 'Completed all 217 assays at %s\n' "$(date -Is)"

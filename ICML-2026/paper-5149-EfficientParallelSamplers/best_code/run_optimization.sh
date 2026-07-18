#!/bin/bash
# Optimization runner: modify gen_kwargs and run evaluation
set -e

INNER_REC=4
STATE_NOISE=0.0
EXIT_T=0.03
FREEZE_STRATEGY=latent-diff
NUM_STEPS=32
EMA_EMBEDS=0.1
MAX_WAVEFRONT=128
LIMIT=${LIMIT:-200}
EXTRA_KWARGS=${EXTRA_KWARGS:-}

cd /repo
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/autosota_cache/hf
export HF_ENDPOINT=https://hf-mirror.com
export MODEL_PATH=${MODEL_PATH:-/models/huginn-0125}

# Build gen_kwargs string
GEN_KWARGS="temperature=0.2,top_p=0.95,do_sample=True,headway=1,inner_recurrence=${INNER_REC},state_noise_mixing=${STATE_NOISE},ema_embeds=${EMA_EMBEDS},exit_t=${EXIT_T},max_wavefront=${MAX_WAVEFRONT},freeze_strategy=${FREEZE_STRATEGY},num_steps=${NUM_STEPS}"
if [ -n "$EXTRA_KWARGS" ]; then
    GEN_KWARGS="${GEN_KWARGS},${EXTRA_KWARGS}"
fi

echo "=== Optimization Run ==="
echo "INNER_REC=${INNER_REC} STATE_NOISE=${STATE_NOISE} EXIT_T=${EXIT_T}"
echo "FREEZE_STRATEGY=${FREEZE_STRATEGY} NUM_STEPS=${NUM_STEPS} LIMIT=${LIMIT}"

# Modify eval_gsm8k_runner.py
cp /repo/eval_gsm8k_runner.py /repo/eval_gsm8k_runner.py.bak
sed -i "s|gen_kwargs = (|gen_kwargs = os.environ.get('GEN_KWARGS_OVERRIDE', |" /repo/eval_gsm8k_runner.py
sed -i "s|gen_kwargs=gen_kwargs,|gen_kwargs=GEN_KWARGS_STR,|" /repo/eval_gsm8k_runner.py

# Add env var support at top of script (after imports)
python3 -c 

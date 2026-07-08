#!/bin/bash
# Quick Start: Full pipeline for Spherical Steering on Qwen-2.5-7B-Instruct
set -e

MODEL="Qwen2.5-7B-Instruct"
LAYER=19
KAPPA=20.0
ALPHA=0.6
BETA=0.4

# Step 1: Extract hidden states
echo ""
echo "========== Step 1: Extract Hidden States =========="
python get_activations.py $MODEL --layer $LAYER

# Step 2: Compute prototypes (2-fold CV)
echo ""
echo "========== Step 2: Compute Prototypes =========="
python get_prototypes.py --feature_file ./features/${MODEL}_layer${LAYER}.npz

# Step 3: MC evaluation (both folds)
echo ""
echo "========== Step 3: MC Evaluation =========="
for FOLD in 0 1; do
    echo ">>> MC fold${FOLD}"
    python evaluate_mc.py $MODEL \
        --prototype_path ./prototypes/${MODEL}_layer${LAYER}_fold${FOLD}.npz \
        --layer $LAYER --kappa $KAPPA --alpha $ALPHA --beta $BETA \
        --output_path results/${MODEL}_l${LAYER}_a${ALPHA}_b${BETA}_fold${FOLD}.json
done

echo ""
echo ">>> MC Average:"
python -c "
import json, glob
mc1, mc2, mc3 = [], [], []
for f in sorted(glob.glob('results/${MODEL}_l${LAYER}_a${ALPHA}_b${BETA}_fold*.json')):
    d = json.load(open(f))
    mc1.append(d['metrics']['MC1']); mc2.append(d['metrics']['MC2']); mc3.append(d['metrics']['MC3'])
    print(f'  {f}: MC1={d[\"metrics\"][\"MC1\"]:.4f} MC2={d[\"metrics\"][\"MC2\"]:.4f} MC3={d[\"metrics\"][\"MC3\"]:.4f}')
print(f'  AVG: MC1={sum(mc1)/len(mc1):.4f}  MC2={sum(mc2)/len(mc2):.4f}  MC3={sum(mc3)/len(mc3):.4f}')
"

# Step 4: LLM Judge evaluation (both folds)
echo ""
echo "========== Step 4: LLM Judge Evaluation =========="
for FOLD in 0 1; do
    echo ">>> Judge fold${FOLD}"
    python evaluate_llm_judge.py $MODEL \
        --prototype_path ./prototypes/${MODEL}_layer${LAYER}_fold${FOLD}.npz \
        --layer $LAYER --kappa $KAPPA --alpha $ALPHA --beta $BETA
done

echo ""
echo ">>> Judge Average:"
python -c "
import pandas as pd, glob, numpy as np
files = sorted(glob.glob('results_llm_judge/${MODEL}_l${LAYER}_fold*_steered_a${ALPHA}_b${BETA}_zeroshot.csv'))
truth, info = [], []
for f in files:
    df = pd.read_csv(f)
    t, i = df['truth_prob'].mean(), df['info_prob'].mean()
    truth.append(t); info.append(i)
    print(f'  {f}: Truth={t:.4f}  Info={i:.4f}  TxI={t*i:.4f}')
t_avg, i_avg = np.mean(truth), np.mean(info)
print(f'  AVG: Truth={t_avg:.4f}  Info={i_avg:.4f}  TxI={t_avg*i_avg:.4f}')
"

echo ""
echo "========== Done! =========="

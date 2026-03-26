#!/bin/bash

export ROOT="experiments/data/gemma-arith-baseline-eval"
mkdir -p $ROOT

# Evaluate the 4 key digit combinations: (1,1), (1,8), (8,1), (8,8)
for dig1 in 1 8; do
  for dig2 in 1 8; do
    python3 -m experiments.baseline gemma-2-2b "$ROOT/arith-${dig1}${dig2}" --batch_size 16         --ndevices 1 --dataset arith --format "few-shot"         --data_params dig1=$dig1 dig2=$dig2 n=1000 op="+" append_ans=False         --format_params shots=8         --max_new_tokens 20         --model_path /models
  done
done

# Compute accuracy from pkl files
python3 - << 'PYEOF'
import re
import pickle
import os

def compute_accuracy(outputs, targets):
    correct = 0
    for output_text, target_text in zip(outputs, targets):
        try:
            # Strip bos/pad tokens from output
            clean_output = re.sub(r'^(<pad>|<bos>)+', '', output_text)
            target_stripped = target_text.lstrip()
            
            # Model generation is after target
            model_gen = clean_output[len(target_stripped):]
            
            # Extract first number from model generation
            m = re.match(r'\s*(-?\d+)', model_gen)
            if not m:
                continue
            model_ans = int(m.group(1))
            
            # Extract the question operands from target
            question_pattern = re.search(r'(\d+)\s*\+\s*(\d+)\s*=\s*$', target_stripped)
            if not question_pattern:
                continue
            op1 = int(question_pattern.group(1))
            op2 = int(question_pattern.group(2))
            correct_ans = op1 + op2
            
            if model_ans == correct_ans:
                correct += 1
        except Exception:
            continue
    return correct / len(outputs) if outputs else 0.0

ROOT = 'experiments/data/gemma-arith-baseline-eval'
for dig1 in [1, 8]:
    for dig2 in [1, 8]:
        fname = f'{ROOT}/arith-{dig1}{dig2}-benchmark.pkl'
        if not os.path.exists(fname):
            print(f'WARNING: {fname} not found')
            continue
        d = pickle.load(open(fname, 'rb'))
        acc = compute_accuracy(d['output'], d['target'])
        key = f'accuracy_{dig1}{dig2}'
        print(f'{key}: {acc:.4f}')

PYEOF

#!/bin/bash
# Evaluate all 4 SP methods on LongBenchV2 (multiple-choice).
# Run each method separately to avoid interference.
#
# Usage: bash scripts/run_eval_longbenchv2.sh

set -e
cd "$(dirname "$0")/.."

MODEL="${MODEL_PATH:-/path/to/Qwen3-14B}"
RATIO=0.15
STRIDE=1
OUTPUT_DIR="results/longbenchv2_sp_eval"

echo "=========================================="
echo "LongBenchV2 evaluation: all SP methods"
echo "=========================================="

# --- SP Guided Recompute (our method) ---
echo ""
echo "[1/4] sp_guided_recompute - LongBenchV2"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks longbenchv2 \
    --methods sp_guided_recompute \
    --recompute_ratio "$RATIO" --output "$OUTPUT_DIR"

# --- SP CacheBlend ---
echo ""
echo "[2/4] sp_cacheblend - LongBenchV2"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks longbenchv2 \
    --methods sp_cacheblend \
    --recompute_ratio "$RATIO" --output "$OUTPUT_DIR"

# --- SP LEGO ---
echo ""
echo "[3/4] sp_lego - LongBenchV2"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks longbenchv2 \
    --methods sp_lego \
    --recompute_ratio "$RATIO" --output "$OUTPUT_DIR"

# --- Ring Attention SP Baseline ---
echo ""
echo "[4/4] ring_attention - LongBenchV2"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks longbenchv2 \
    --methods ring_attention \
    --heads_k_stride "$STRIDE" --output "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "All evaluations complete. Collecting results..."
echo "=========================================="

python3 -c "
import json, glob, os

output_dir = '${OUTPUT_DIR}'
summaries = {}

for summary_path in sorted(glob.glob(os.path.join(output_dir, '**/summary.json'), recursive=True)):
    with open(summary_path) as f:
        summary = json.load(f)
    parts = os.path.relpath(summary_path, output_dir).split(os.sep)
    task_method = parts[0]
    summaries[task_method] = summary

print()
print('=' * 80)
print(f'{\"Task\":<15} {\"Method\":<25} {\"Acc (%)\":<10} {\"F1 (%)\":<10} {\"TTFT (ms)\":<10}')
print('-' * 80)
for key in sorted(summaries.keys()):
    s = summaries[key]
    parts = key.split('_', 1)
    task = parts[0]
    method = parts[1] if len(parts) > 1 else key
    acc = s.get('accuracy', 0)
    f1 = s.get('avg_f1', 0) * 100
    ttft = s.get('avg_ttft_ms', -1)
    ttft_str = f'{ttft:.0f}' if ttft >= 0 else '-'
    print(f'{task:<15} {method:<25} {acc:<10.2f} {f1:<10.2f} {ttft_str:<10}')
print('=' * 80)

combined_path = os.path.join(output_dir, 'all_results.json')
with open(combined_path, 'w') as f:
    json.dump(summaries, f, indent=2)
print(f'\nCombined results saved to: {combined_path}')
"

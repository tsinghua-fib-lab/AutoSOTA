#!/bin/bash
# Full evaluation of all SP methods across all three QA tasks.
# Run each method separately to avoid interference.
#
# Usage: bash scripts/run_eval_all_methods.sh

set -e
cd "$(dirname "$0")/.."

MODEL="${MODEL_PATH:-/path/to/Qwen3-14B}"
RATIO=0.15
STRIDE=1
OUTPUT_DIR="results/all_methods_eval"

echo "=========================================="
echo "Full evaluation: all methods, all tasks"
echo "=========================================="

# --- SP Guided Recompute (our method) ---

echo ""
echo "[1/12] sp_guided_recompute - HotpotQA"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks hotpotqa \
    --methods sp_guided_recompute \
    --recompute_ratio "$RATIO" --output "$OUTPUT_DIR"

echo ""
echo "[2/12] sp_guided_recompute - 2WikiMQA"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks 2wikimqa \
    --methods sp_guided_recompute \
    --recompute_ratio "$RATIO" --output "$OUTPUT_DIR"

echo ""
echo "[3/12] sp_guided_recompute - MuSiQue"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks musique \
    --methods sp_guided_recompute \
    --recompute_ratio "$RATIO" --output "$OUTPUT_DIR"

# --- SP CacheBlend ---

echo ""
echo "[4/12] sp_cacheblend - HotpotQA"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks hotpotqa \
    --methods sp_cacheblend \
    --recompute_ratio "$RATIO" --output "$OUTPUT_DIR"

echo ""
echo "[5/12] sp_cacheblend - 2WikiMQA"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks 2wikimqa \
    --methods sp_cacheblend \
    --recompute_ratio "$RATIO" --output "$OUTPUT_DIR"

echo ""
echo "[6/12] sp_cacheblend - MuSiQue"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks musique \
    --methods sp_cacheblend \
    --recompute_ratio "$RATIO" --output "$OUTPUT_DIR"

# --- SP LEGO ---

echo ""
echo "[7/12] sp_lego - HotpotQA"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks hotpotqa \
    --methods sp_lego \
    --recompute_ratio "$RATIO" --output "$OUTPUT_DIR"

echo ""
echo "[8/12] sp_lego - 2WikiMQA"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks 2wikimqa \
    --methods sp_lego \
    --recompute_ratio "$RATIO" --output "$OUTPUT_DIR"

echo ""
echo "[9/12] sp_lego - MuSiQue"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks musique \
    --methods sp_lego \
    --recompute_ratio "$RATIO" --output "$OUTPUT_DIR"

# --- Ring Attention SP Baseline ---

echo ""
echo "[10/12] ring_attention - HotpotQA"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks hotpotqa \
    --methods ring_attention \
    --heads_k_stride "$STRIDE" --output "$OUTPUT_DIR"

echo ""
echo "[11/12] ring_attention - 2WikiMQA"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks 2wikimqa \
    --methods ring_attention \
    --heads_k_stride "$STRIDE" --output "$OUTPUT_DIR"

echo ""
echo "[12/12] ring_attention - MuSiQue"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks musique \
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
print('=' * 90)
print(f'{\"Task\":<15} {\"Method\":<25} {\"F1 (%)\":<10} {\"Acc (%)\":<10} {\"TTFT (ms)\":<10}')
print('-' * 90)

# Sort by task then method for readability
for key in sorted(summaries.keys()):
    s = summaries[key]
    parts = key.split('_', 1)
    task = parts[0]
    method = parts[1] if len(parts) > 1 else key
    f1 = s.get('avg_f1', 0) * 100
    acc = s.get('accuracy', 0)
    ttft = s.get('avg_ttft_ms', -1)
    ttft_str = f'{ttft:.0f}' if ttft >= 0 else '-'
    print(f'{task:<15} {method:<25} {f1:<10.2f} {acc:<10.2f} {ttft_str:<10}')
print('=' * 90)

combined_path = os.path.join(output_dir, 'all_results.json')
with open(combined_path, 'w') as f:
    json.dump(summaries, f, indent=2)
print(f'\nCombined results saved to: {combined_path}')
"

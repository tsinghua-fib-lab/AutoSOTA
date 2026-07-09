#!/bin/bash
# Evaluate ring_attention (SP baseline) and sp_guided_recompute with heads_k_stride=1
# across all three QA tasks. Run separately to avoid substitute_hf_flash_attn interference.
#
# Usage: bash scripts/run_eval_stride1.sh

set -e
cd "$(dirname "$0")/.."

MODEL="${MODEL_PATH:-/path/to/Qwen3-14B}"
STRIDE=1
RATIO=0.15
OUTPUT_DIR="results/stride1_eval"

echo "=========================================="
echo "Evaluating with heads_k_stride=${STRIDE}"
echo "=========================================="

# --- SP Guided Recompute (our method) ---

echo ""
echo "[1/6] sp_guided_recompute - HotpotQA"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks hotpotqa \
    --methods sp_guided_recompute \
    --recompute_ratio "$RATIO" --output "$OUTPUT_DIR"

echo ""
echo "[2/6] sp_guided_recompute - 2WikiMQA"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks 2wikimqa \
    --methods sp_guided_recompute \
    --recompute_ratio "$RATIO" --output "$OUTPUT_DIR"

echo ""
echo "[3/6] sp_guided_recompute - MuSiQue"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks musique \
    --methods sp_guided_recompute \
    --recompute_ratio "$RATIO" --output "$OUTPUT_DIR"

# --- Ring Attention SP Baseline ---

echo ""
echo "[4/6] ring_attention - HotpotQA"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks hotpotqa \
    --methods ring_attention \
    --heads_k_stride "$STRIDE" --output "$OUTPUT_DIR"

echo ""
echo "[5/6] ring_attention - 2WikiMQA"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks 2wikimqa \
    --methods ring_attention \
    --heads_k_stride "$STRIDE" --output "$OUTPUT_DIR"

echo ""
echo "[6/6] ring_attention - MuSiQue"
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
    --model "$MODEL" --tasks musique \
    --methods ring_attention \
    --heads_k_stride "$STRIDE" --output "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "All evaluations complete. Collecting results..."
echo "=========================================="

# Collect all summary.json files into a single report
python3 -c "
import json, glob, os

output_dir = '${OUTPUT_DIR}'
summaries = {}

for summary_path in sorted(glob.glob(os.path.join(output_dir, '**/summary.json'), recursive=True)):
    with open(summary_path) as f:
        summary = json.load(f)
    # Extract task and method from path: output_dir/task_method/timestamp/summary.json
    parts = os.path.relpath(summary_path, output_dir).split(os.sep)
    task_method = parts[0]  # e.g. 'hotpotqa_sp_guided_recompute'
    summaries[task_method] = summary

# Print summary table
print()
print('=' * 80)
print(f'{\"Task\":<20} {\"Method\":<25} {\"F1 (%)\":<10} {\"Acc (%)\":<10} {\"TTFT (ms)\":<10}')
print('-' * 80)
for key, s in summaries.items():
    # Parse task_method key
    parts = key.split('_', 1)
    task = parts[0]
    method = parts[1] if len(parts) > 1 else key
    f1 = s.get('avg_f1', 0) * 100
    acc = s.get('accuracy', 0)
    ttft = s.get('avg_ttft_ms', -1)
    ttft_str = f'{ttft:.0f}' if ttft >= 0 else '-'
    print(f'{task:<20} {method:<25} {f1:<10.2f} {acc:<10.2f} {ttft_str:<10}')
print('=' * 80)

# Save combined
combined_path = os.path.join(output_dir, 'all_results_stride${STRIDE}.json')
with open(combined_path, 'w') as f:
    json.dump(summaries, f, indent=2)
print(f'\nCombined results saved to: {combined_path}')
"

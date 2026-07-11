#!/bin/bash
# REVIS CHAIR Evaluation Script
# Reproduces CHAIRS and CHAIRI metrics for paper 3190
set -e

cd /repo
export HF_HOME=/autosota_cache/hf
export NLTK_DATA=/repo/utils/nltk_data

echo "=== REVIS CHAIR Evaluation ==="
echo "Model: Qwen2.5-VL-7B-Instruct"
echo "Config: alpha=1.6, layer=27, images=100, max_tokens=512"

# Run REVIS evaluation
python3 main_all_visual_only.py run \
    --model_path /models/Qwen2.5-VL-7B-Instruct \
    --benchmark chair \
    --vector_file vector/qwen2.5vl_none_image.pt \
    --results_dir results \
    --question_dir data/chair_val2014_100.jsonl \
    --image_folder /datasets/coco/val2014 \
    --gt_dir /datasets/coco \
    --vis_layers 27 \
    --alpha_visual 1.6 \
    --tau_low 0.2 \
    --risk_gamma 1.0

# Compute CHAIR metrics from saved captions
python3 -c "
import json, os, sys
sys.path.insert(0, /repo)
from utils.chair import CHAIR, read_jsonl

# REVIS results
response_file = results/chair/Vis27_aV1.6_Gamma1.0_TauL0.2/chair_captions.jsonl
annotation_dir = /datasets/coco/annotations

generated_data_raw = read_jsonl(response_file)
generated_data = [{image_id: int(item.get(image_id, item.get(question_id, -1))), caption: item.get(text, item.get(caption, ))} for item in generated_data_raw]
unique_data = list({item[image_id]: item for item in generated_data}.values())
img_ids = sorted([item[image_id] for item in unique_data])

evaluator = CHAIR(imids=img_ids, coco_annotation_path=annotation_dir)
evaluator.get_annotations()
scores = evaluator.compute_chair(unique_data)

results = {CHAIRs: round(scores[CHAIRs] * 100, 2), CHAIRi: round(scores[CHAIRi] * 100, 2)}
print(json.dumps(results, indent=2))

# Save
with open(response_file.replace(.jsonl, _chair_results.json), w) as f:
    json.dump(results, f, indent=2)
"

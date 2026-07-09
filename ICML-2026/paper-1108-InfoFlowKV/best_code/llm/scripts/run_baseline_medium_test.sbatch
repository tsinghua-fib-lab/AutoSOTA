#!/bin/bash
#SBATCH --job-name=baseline_med
#SBATCH --account=torch_pr_637_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200"
#SBATCH --nodes=1 --tasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=128GB --time=2:00:00
#SBATCH --output=slurm-%j.out

source ${CONDA_BASE:-$HOME/miniconda3}/bin/activate
conda activate qwen3
cd ${REPO_ROOT:-$HOME/kv-cache-dev}

echo "=== Baseline medium test (20 samples) ==="

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -c "
import sys; sys.path.insert(0, '.')
import torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from benchmarks import get_dataset

import scripts.inference_with_recompute_kv as script
script._CUSTOM_QUERY_TEMPLATE = 'Read the above context and answer the multiple-choice question. Only output the letter of the correct answer (A, B, C, or D).\n{query}\nAnswer:'
from scripts.inference_with_recompute_kv import build_full_prompt, get_chat_suffix

model_path = '${MODEL_PATH:-/path/to/Qwen3-14B}'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

dataset = get_dataset('longbenchv2', input_dir='inputs', device='cuda:0')
dataset.load(length_filter=['medium'])
print(f'Medium samples: {len(dataset.data)}')

model = AutoModelForCausalLM.from_pretrained(model_path, device_map={'': 'cuda:0'}, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation='flash_attention_2')
model.eval()
device = torch.device('cuda:0')

correct = 0
total = min(20, len(dataset.data))
for i in range(total):
    s = dataset.data[i]
    context = s['context']
    query = s['input']
    answer = s['answer']

    full_prompt = build_full_prompt(context, query, 'qwen', tokenizer)
    input_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    max_len = 131072 - 32
    orig_len = len(input_ids)
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len // 2] + input_ids[-max_len // 2:]
    inputs = {'input_ids': torch.tensor([input_ids]).to(device), 'attention_mask': torch.ones(1, len(input_ids), dtype=torch.long).to(device)}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False, temperature=1.0)
    pred = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    pred_letter = dataset.parse_generation(pred)
    is_correct = pred_letter == answer.strip().upper()
    if is_correct:
        correct += 1
    trunc = f'trunc {orig_len}->{len(input_ids)}' if orig_len > max_len else f'no trunc ({orig_len})'
    print(f'[{i+1}/{total}] pred={repr(pred[:60]):<45} letter={pred_letter} answer={answer} {\"OK\" if is_correct else \"X\"} ({trunc})')

    del inputs, out
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

print(f'\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%')
"

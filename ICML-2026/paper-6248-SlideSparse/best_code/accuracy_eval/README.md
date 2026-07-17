# SlideSparse Accuracy Evaluation

Pruning and evaluation scripts for reproducing SlideSparse accuracy results. Based on [Wanda](https://github.com/locuslab/wanda) and [SparseGPT](https://github.com/IST-DASLab/sparsegpt), with modifications to support arbitrary N:M sparsity patterns and newer model architectures (e.g., Qwen2.5).

## Environment

- PyTorch 2.9.0+cu129
- CUDA 12.9
- transformers 4.57.3
- accelerate 1.12.0
- lm-eval 0.4.11
- datasets 4.8.4
- bitsandbytes 0.45.0 (for INT8 evaluation)

## Usage

### Pruning (Wanda)
```bash
cd wanda
# 6:8 sparsity on Qwen2.5-14B (prune 2 out of every 8)
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model Qwen/Qwen2.5-14B \
  --prune_method wanda \
  --sparsity_type 6:8 \
  --prune_n 2 --prune_m 8 \
  --save /path/to/results/14b/wanda_6_8 \
  --save_model /path/to/results/14b/wanda_6_8/model
```

### Pruning (SparseGPT)
```bash
cd sparsegpt
# 6:8 sparsity on Qwen2.5-7B (prune 2 out of every 8)
CUDA_VISIBLE_DEVICES=0 python llama.py Qwen/Qwen2.5-7B wikitext2 \
  --prunen 2 --prunem 8 \
  --save /path/to/results/7b_sparsegpt/sgpt_6_8
```

### N:M Pattern Reference

**Important**: In our notation, N:M means "keep N non-zeros out of every M elements". The number of **pruned (zeroed) weights** per group = M - N = 2 for all patterns below. You must pass `--prune_n 2 --prune_m M` explicitly.

| Pattern | Keep:Total | Pruned | Sparsity | Wanda args | SparseGPT args |
|---------|-----------|--------|----------|------------|----------------|
| 2:4 | 2:4 | 2 | 50% | `--sparsity_type 2:4 --prune_n 2 --prune_m 4` | `--prunen 2 --prunem 4` |
| 4:6 | 4:6 | 2 | 33% | `--sparsity_type 4:6 --prune_n 2 --prune_m 6` | `--prunen 2 --prunem 6` |
| 6:8 | 6:8 | 2 | 25% | `--sparsity_type 6:8 --prune_n 2 --prune_m 8` | `--prunen 2 --prunem 8` |
| 8:10 | 8:10 | 2 | 20% | `--sparsity_type 8:10 --prune_n 2 --prune_m 10` | `--prunen 2 --prunem 10` |

**Key**: For all patterns, we prune 2 weights per group. `prunen`/`prune_n` = weights to zero.

### Batch Evaluation Script
```bash
# All 9 benchmarks in one command
./run_eval.sh /path/to/pruned_model /path/to/results 0 auto

# INT8 mode
INT8=1 ./run_eval.sh Qwen/Qwen2.5-14B /path/to/results/dense_int8 0 4
```

### Evaluation (lm-eval)
```bash
# BF16 pruned model
lm_eval --model hf \
  --model_args pretrained=/path/to/pruned_model,dtype=float16 \
  --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,boolq,openbookqa \
  --batch_size 4 --output_path /path/to/results/commonsense

# INT8 quantized (bitsandbytes LLM.int8(), dynamic quantization)
lm_eval --model hf \
  --model_args pretrained=/path/to/pruned_model,load_in_8bit=True \
  --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,boolq,openbookqa \
  --batch_size 4 --output_path /path/to/results/commonsense_int8

# MMLU (5-shot)
lm_eval --model hf \
  --model_args pretrained=/path/to/pruned_model,dtype=float16 \
  --tasks mmlu --num_fewshot 5 \
  --batch_size 4 --output_path /path/to/results/mmlu

# GSM8K (5-shot)
lm_eval --model hf \
  --model_args pretrained=/path/to/pruned_model,dtype=float16 \
  --tasks gsm8k --num_fewshot 5 \
  --batch_size 4 --output_path /path/to/results/gsm8k
```

### INT8 Evaluation Notes
- We use **bitsandbytes LLM.int8()** (dynamic mixed INT8+FP16 quantization at inference time)
- NOT W8A8 static quantization — no pre-quantized checkpoints needed
- Dense and sparse models use identical quantization method for fair comparison
- Add `load_in_8bit=True` to `--model_args` to enable INT8 mode

### Docker Environment
```bash
# Base image with all dependencies
docker run -d --name eval --gpus all --shm-size=32g \
  -v /path/to/data:/workspace \
  bcacdwk/vllmbench:universal tail -f /dev/null

# Install additional deps inside container
docker exec eval pip install lm-eval datasets bitsandbytes
```

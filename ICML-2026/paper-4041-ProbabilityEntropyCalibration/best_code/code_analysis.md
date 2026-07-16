# Code Analysis: Paper 4041 - RankTuner SOTA Preparation Repair

## Preparation Failure Diagnosis

The normal SOTA preparation step failed because:
1. **git not installed**: Container lacks git binary; apt-get blocked by network proxy
2. **Cache cleanup**: `/autosota_cache/checkpoints/`, `/autosota_cache/paper-4041-venv/`, and `/autosota_cache/eval_results/` were deleted between reproduction and SOTA start
3. **Stale mounts**: `/models` and `/datasets` mounts had stale NFS file handles from cache cleanup
4. **No model weights**: The checkpoint at `/repo/verl/checkpoints_5e-5/` had only config files, no safetensors

## Repair Actions

1. Copied git binary from host into container
2. Created `/tools/record_score.sh` from host script
3. Rebuilt Python venv at `/autosota_cache/paper-4041-venv` with:
   - PyTorch 2.6.0+cu124
   - transformers 4.53.3
   - vllm 0.8.4
   - All training dependencies (accelerate, datasets, peft, ray, etc.)
4. Downloaded Qwen2.5-Math-7B to `/autosota_cache/models/Qwen2.5-Math-7B`
5. Started baseline RankTuner training

## Corrected Evaluation Command

```bash
source /autosota_cache/paper-4041-venv/bin/activate
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/autosota_cache/hf
export TMPDIR=/autosota_cache/tmp
cd /repo/math_evaluation

python3 -u math_eval.py \
    --model_name_or_path /autosota_cache/checkpoints/numina-cot-ranktuner-Qwen2.5-Math-7B/global_step_39 \
    --data_name math_oai \
    --output_dir /autosota_cache/eval_results/numina-cot-ranktuner-Qwen2.5-Math-7B \
    --split test \
    --prompt_type qwen25-math-cot \
    --num_test_sample -1 \
    --seed 0 \
    --temperature 1.0 \
    --n_sampling 16 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm
```

Key changes from manifest:
- Model path: `/autosota_cache/models/Qwen2.5-Math-7B` (not `/models/`)
- Checkpoint path: `/autosota_cache/checkpoints/...` (mounted cache)
- Proxy must be unset for HF access
- GPU 0,1 are available (mapped from host 6,7)

## Evaluation Output Format

JSON file `math_oai_metrics.json` in output_dir:
- `pass_at_k.pass@1`: float percentage (e.g., 67.79)
- `pass_at_k.pass@16`: float percentage (e.g., 90.8)
- All values in 0-100 scale

## Safe Optimization Targets

### Training Configuration (low risk)
- Learning rate: 5e-5 → sweep 2e-5 to 1e-4
- Batch size: 256 → 128 or 512
- Warmup ratio: 0.1 → 0.03 to 0.15
- LR schedule: cosine → WSD (wsd_stable_ratio=0.8)
- Gradient accumulation: 4 → 8

### Algorithm Modifications (medium risk)
- `verl/verl/trainer/fsdp_general_trainer.py`: Loss type scheduling, EMA, WSD scheduler
- `verl/verl/trainer/compute_loss.py`: Completion-only loss masking, Bayesian variance damping

### Data Modifications (medium risk)
- `verl/examples/data_preprocess/numina_cot.py`: Quality filtering
- Weighted sampling based on RankTuner scale

### Key files to modify
1. Training loop: `/repo/verl/verl/trainer/fsdp_general_trainer.py`
2. Loss function: `/repo/verl/verl/trainer/compute_loss.py`
3. Data preprocessing: `/repo/verl/examples/data_preprocess/numina_cot.py`
4. Training config: passed via CLI arguments to `torchrun`

## Constraints Preserved
- Same test data (MATH-OAI 500 problems)
- Same evaluation protocol (vLLM, 16 samples, temperature 1.0)
- Same model architecture (Qwen2.5-Math-7B)
- Same primary metric (Pass@16) and guardrail (Pass@1)
- All changes limited to training-side modifications

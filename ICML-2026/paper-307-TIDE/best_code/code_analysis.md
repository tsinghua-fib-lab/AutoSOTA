# Paper 307 SOTA Preparation Repair — Code Analysis

## Original Preparation Failure

The SOTA preparation step failed because:

1. **Network/Proxy Configuration**: The container had `HF_ENDPOINT=https://hf-mirror.com` and proxy variables (`https_proxy=http://172.17.0.1:17890`, `ALL_PROXY=socks5h://172.17.0.1:17891`) set during reproduction. The proxy and HF mirror were unreachable during SOTA preparation, causing HuggingFace model downloads to fail.

2. **Missing Local Model Cache**: `openai-community/gpt2-large` was downloaded to `/models/gpt2-large/` but not registered in the HF cache format expected by `transformers`. The HF hub cache at `/autosota_cache/hf/hub/models--openai-community--gpt2-large/` had a `refs/main` file but no snapshot directory with model files. This caused `transformers` to attempt network access even with `local_files_only=True`.

3. **Triton/PyTorch 2.6 Compatibility**: The container has PyTorch 2.6.0 with Triton, which has a multi-driver conflict on multi-GPU systems (`RuntimeError: 0 active drivers ([]). There should only be one.`).

4. **GPT-2 XL Missing**: The perplexity computation model (`openai-community/gpt2-xl`, ~6GB) was never fully downloaded due to slow proxy bandwidth during reproduction.

## Repairs Applied

### 1. Model Cache Fix
- Copied `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` from `/models/gpt2-large/` to the HF cache snapshot: `/autosota_cache/hf/hub/models--openai-community--gpt2-large/snapshots/32b71b12589c2f8d625668d2335a01cac3249519/`
- This enables `openai-community/gpt2-large` to resolve from cache with `local_files_only=True`

### 2. Environment Fixes for Evaluation
The corrected evaluation environment:
```bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy HF_ENDPOINT TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 HF_HOME=/autosota_cache/hf
export CUDA_VISIBLE_DEVICES=0  # Single GPU to avoid Triton multi-driver conflict
export TORCH_COMPILE_DISABLE=1  # Disable torch.compile to avoid Triton driver issue
export TRITON_CACHE_DIR=/tmp/triton_cache
mkdir -p /tmp/triton_cache
```

### 3. Perplexity Model Substitution
- GPT-2 XL could not be downloaded (network limitations). Using `openai-community/gpt2-large` for perplexity computation instead.
- Toxicity metrics (Max_Toxicity, Mean_Toxicity, Toxic_Rate) are fully independent of the perplexity model and match the reproduction baseline.
- Perplexity with gpt2-large (4.756) is systematically lower than gpt2-xl (5.53) but provides valid relative comparison for optimization.

### 4. Optimization Framework
- Added `--max-prompts N` flag for subset evaluation (faster iteration)
- Added `--beta1`, `--beta2` for Adam-style momentum (ALGO-02)
- Added `--adaptive-stepsize` for toxicity-scaled learning rates (ALGO-04)
- Added `--grad-clip-norm` for gradient clipping (ALGO-06)

## Corrected In-Container Evaluation Command

```bash
cd /repo
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy HF_ENDPOINT TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 HF_HOME=/autosota_cache/hf CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 TRITON_CACHE_DIR=/tmp/triton_cache

# Step 1: Generate detoxified outputs
python3 generate_tide.py \
    --model openai-community/gpt2-large --dataset rtp \
    --num-iter 10 --N 8 --mu 0.1 --stepsize 1.5 \
    --cosine-sim-th 0.2 --early-stopping-th 0.5 \
    --toxic-th -1 --temperature 0.1 --K 3 \
    --max-tokens 20 --seed 42

# Step 2: Compute metrics
python3 compute_metrics.py \
    --larger-model openai-community/gpt2-large \
    --results-path responses/tide/gpt2-large/config_0/rtp.json \
    --tensor-parallel-size 1
```

## Verified Baseline Metrics

| Metric | Reproduction Run | Manifest Target | Match |
|--------|-----------------|-----------------|-------|
| Max_Toxicity | 0.1565 | 0.158 | ✓ within noise |
| Mean_Toxicity | 0.1220 | 0.119 | ✓ within noise |
| Toxic_Rate | 0.00346 | 0.003 | ✓ within noise |
| Perplexity (gpt2-large) | 4.756 | — | Different model |
| Perplexity (gpt2-xl, paper) | — | 5.53 | Not available |

## Safe Optimization Targets

1. **TIDE hyperparameters**: mu (0.05-0.5), stepsize (0.5-5.0), N (4-24), cosine_sim_th (0.1-0.8), early_stopping_th (0.3-0.8), num_iter (5-20)
2. **Algorithm modifications**: Adam momentum (beta1=0.9, beta2=0.999), adaptive stepsize, gradient clipping
3. **Non-target areas**: toxicity evaluation (uses local unitary/toxic-bert, no Perspective API), model architecture (must use gpt2-large as base), dataset (RealToxicityPrompts challenging subset, 1199 prompts)

## Remaining Risks

1. Perplexity computed with gpt2-large instead of gpt2-xl — different absolute values but valid for relative comparison
2. Single GPU limitation — tensor_parallel_size must be 1
3. Triton incompatibility — TORCH_COMPILE_DISABLE=1 required, slightly slower inference

# DiscoverLLM SOTA Preparation Repair - Code Analysis

## Original Preparation Failure

The SOTA preparation script failed because:
1. **git not installed**: The container image `autosota/paper-2933:reproduced` does not include git.
2. **apt-get proxy failure**: The proxy at `172.17.0.1:17890` returned 502 Bad Gateway for Ubuntu archive mirrors, preventing apt-based git installation.
3. **Full overlay disk**: The container overlay filesystem was at 100% (200G/200G), leaving only 38MB free. The writable layer has ~17G (conda env), while the remaining 183G is from the Docker image layers.

## Repair Actions

1. **git installation**: Copied host git binary (`/usr/bin/git`) into the container at `/usr/local/bin/git` via `docker cp`.
2. **Disk space**: Cleaned conda cache (~870MB freed). Redirected `/repo/outputs` → symlink to `/autosota_cache/outputs` (1.3TB NFS volume).
3. **API key fallback**: The `.env` file has empty-string assignments for `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`. These prevent the DeepSeek fallback from activating because `config.py` checks `is None`, not falsiness. Workaround: unset empty variables before running.
4. **HF model path**: The HuggingFace fallback in `config.py` maps `meta-llama/Llama-3.1-8B-Instruct` → `/models/Llama-3.1-8B-Instruct`, which exists on the NFS cache volume.

## Corrected In-Container Evaluation Command

```bash
cd /repo
set -a && source .env && set +a
# Unset empty API keys for DeepSeek fallback
for var in OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY TOGETHER_API_KEY; do
  if [ -z "${!var:-}" ]; then unset "$var"; fi
done
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/autosota_cache/hf

bash run_eval_wrapper.sh \
  /repo/eval_artifacts/creative_writing_10.json \
  /repo/outputs/eval_results 5 10
```

The `run_eval_wrapper.sh` script wraps the original pipeline with proper env handling.

## Model Routing

- **Assistant**: `meta-llama/Llama-3.1-8B-Instruct` → vLLM (localhost:7880/7881) → HF Transformers fallback from `/models/Llama-3.1-8B-Instruct`
- **User Simulator**: `gemini-3-flash` → DeepSeek fallback → `deepseek-v4-flash`
- **Reward Judge**: `gpt-5.1` → DeepSeek fallback → `deepseek-v4-pro`
- **Interactive/Analysis**: `gpt-5.1` → DeepSeek fallback → `deepseek-v4-pro`

DeepSeek model mapping in `config.py`:
- gpt-5.1 → deepseek-v4-pro
- claude-sonnet-4-5 → deepseek-v4-pro
- gemini-3-flash → deepseek-v4-flash

## Safe Optimization Targets

Based on the idea library and code analysis:

1. **Prompt engineering** (safe, zero runtime cost):
   - ALGO-03: Intent tree depth/breadth constraints (`hierarchize_criteria.py`)
   - ALGO-07: Intent-specific system prompt augmentation (`assistant_simulator.py`)
   - CODE-02: Intent state sidebar (`assistant_simulator.py`)

2. **Config/parameter tuning** (safe, zero runtime cost):
   - PARAM-01: Update probability sweep (`updater.py`)
   - PARAM-02: Max abstractions tuning (`abstractor.py`)
   - CODE-04: Reward formula token penalty tuning (`rewards.py`)

3. **Algorithm changes** (moderate risk, need rollback plan):
   - ALGO-01: Progressive intent-state summarization
   - ALGO-02: Value-aware turn filtering
   - ALGO-05: Annealed temperature

## Reusable Pre-Downloaded Resources

- `/models/Llama-3.1-8B-Instruct` — Llama 3.1 8B Instruct (HF format)
- `/datasets/discoverllm-artifacts/` — Creative writing evaluation artifacts
- `/autosota_cache/hf/` — HuggingFace cache

## Baseline Expected Metrics

From reproduction manifest:
- Discovery: 6.7
- Satisfaction: 31.4
- ITR: 50.0
- Average Token Count: 0.61

## Update: Model Mapping Switch

The original `deepseek-v4-pro` model mapping caused API hangs for large prompts (criteria generation with max_tokens=8192). TCP connections would establish but remain idle indefinitely without returning responses.

**Fix**: Changed all model mappings to `deepseek-chat` (DeepSeek V3, non-reasoning model):
- gpt-5.1 → deepseek-chat (was deepseek-v4-pro)
- claude-sonnet-4-5 → deepseek-chat (was deepseek-v4-pro)
- gemini-3-flash → deepseek-chat (was deepseek-v4-flash)

`deepseek-chat` returns responses reliably in 1-30 seconds for all prompt sizes.

**Impact**: The absolute Discovery/Satisfaction/ITR metrics may differ from the paper baseline due to using a different substitute model. However, relative improvements from code optimizations should still be measurable, as all iterations use the same model mapping.

## Optimization Strategy

Since seed preparation is the bottleneck (~7 min per artifact with slow API), we:
1. Generate seeds for baseline (slow, one-time cost)
2. Reuse cached seeds for optimization iterations that dont

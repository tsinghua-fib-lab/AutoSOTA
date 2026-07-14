# Code Analysis: Paper 5253 SOTA Preparation Repair

## Original Preparation Failure

1. **Reusable container `autosota_repro_paper_5253`**: `git` not installed and `dpkg` locked by another `apt-get` process. Could not init git repo or create `_baseline` tag.

2. **New container `autosota_sota_paper_5253`**: First `docker run` failed due to Docker authorization policy blocking `--network host`. Second attempt with simpler network config succeeded but `dpkg` was in interrupted state and `git` was not installed.

3. **Root cause**: The `autosota/paper-5253:reproduced` image, based on `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`, does not include `git` by default. The preparation script's fallback to install git failed due to dpkg state and proxy issues.

## Repair Applied

1. Fixed `dpkg` with `dpkg --configure -a`
2. Installed `git` via `apt-get` (without proxy — direct connection works for apt)
3. Created `/tools/record_score.sh` from host script
4. Initialized git repo in `/repo`, created baseline commit and `_baseline` tag
5. Confirmed `/autosota_artifacts/paper-5253/sota/` is writable

## Corrected In-Container Evaluation Command

```bash
cd /repo && export HF_ENDPOINT=https://hf-mirror.com && export HF_HOME=/autosota_cache/hf && CUDA_VISIBLE_DEVICES=0 python3 aggregate.py --aggregator QwenR1 --task GPQA --gpus 1 --seed 0
```

## Baseline Verification

- **Expected**: Accuracy 54.04% (manifest baseline)
- **Observed**: `acc: 54.04 | dataset: GPQA | aggregator: QwenR1 | seed: 0`
- **Match**: Exact. ✅

## Available Infrastructure

- **GPUs**: 2x NVIDIA A100-SXM4-80GB (81 GB free each)
- **Complete models**: Qwen2.5-7B-Instruct (`/models/Qwen`), DeepSeek-R1-Distill-Qwen-7B (`/models/deepseek-ai`)
- **Pre-computed profiles**: QwenR1_profile_51.41.json, Qwen_profile_35.34.json
- **Pre-computed expert responses**: `skills/GPQA/round0_seed0.csv` (198 questions, 6 responses each)
- **SentenceTransformer**: all-MiniLM-L6-v2 at `/autosota_cache/hf/models/`
- **Test data**: GPQA Diamond (198 samples) at `/repo/test_data/GPQA_test.json`

## Pipeline Structure

1. `annotate_keywords.py` — annotates keywords per question using Qwen
2. `create_profile.py` — builds skill profiles per model on GPQA train
3. `recruit_agents.py` — selects top-k experts per question via keyword-skill matching
4. `expert_inference.py` — runs expert models on test questions
5. `aggregate.py` — majority vote → QwenR1 aggregation → final answer

## Optimization Targets

With only 2 models available, the optimization surface is concentrated on the aggregation step:

1. **ALGO-02**: Weighted voting by skill-profile scores (QwenR1 weight 51.41 vs Qwen 35.34)
2. **ALGO-04**: Consensus bypass — skip QwenR1 when experts unanimously agree
3. **CODE-02**: Try Qwen as aggregator instead of QwenR1
4. **Multi-seed**: Average across seeds 0,1,2 for robustness
5. **ALGO-01**: Adaptive k — requires more models; not applicable with 2 models
6. **CODE-01**: LRU model cache — not applicable since only 2 models are available

## Red-line Compliance

- No test data, labels, or benchmark protocols are modified
- No hard-coded metrics or predictions
- All changes are to inference/aggregation logic only
- Evaluation uses the same GPQA Diamond test set (198 samples)

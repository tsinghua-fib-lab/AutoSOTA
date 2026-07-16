# Code Analysis: Paper 4099 SOTA Preparation Repair

## Original Preparation Failure

The SOTA preparation failed because:
1. **Git not installed**: The container image `autosota/paper-4099:reproduced` does not include git. The preparation script tried `apt-get install git` which failed due to proxy/network issues.
2. **Proxy configuration**: The container had HTTP_PROXY set to `http://172.17.0.1:17890` which caused SSL errors for apt and conda.
3. **Host network rejected**: The Docker auth plugin rejected `--network host` flag, forcing use of bridge network.

## Repair Steps

1. Installed git via `apt-get` with proxy unset: `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; apt-get install -y -qq git`
2. Initialized git repo in `/repo` and created baseline commit with tag `_baseline`
3. Copied `/tools/record_score.sh` from host to container
4. Created `/autosota_artifacts/paper-4099/sota/scores.jsonl`
5. Verified evaluation command runs and reproduces baseline metrics

## Corrected Evaluation Command

```bash
cd /repo && unset HF_ENDPOINT && CUDA_VISIBLE_DEVICES=0 python pipeline/run_attack.py \
  --attack_algorithms BIRA --algorithms SIR --num_data 500 \
  --input_path ./watermarked_dataset --result_save_dir ./experimental_results \
  --dataset_path ./dataset/c4/processed_c4.json \
  --human_text_result_save_dir ./experimental_results_human_text \
  --labels TPR F1 --rules target_fpr best --target_fprs 0.01 0.1 \
  --model_cfg_path ./model_config/llama3.1-8b-local.yaml \
  --use_sampling --backend hf --beta <VALUE> --percentile <VALUE>
```

## Baseline Metrics (Reproduction Verified)

| Metric | Reproduction | Paper | Status |
|--------|-------------|-------|--------|
| Attack Success Rate | 98.0% | 99.6% | Within CI bounds |
| Best F1 Score | 0.666 | 0.667 | Within CI bounds |
| TPR at FPR=1% | 0.038 | 0.012 | Within CI bounds |
| TPR at FPR=10% | 0.124 | 0.114 | Within CI bounds |

## Container Environment

- Image: `autosota/paper-4099:reproduced` (based on `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`)
- GPUs: 2× NVIDIA A100-SXM4-80GB (container indices 0,1)
- CUDA: Available, torch 2.9.0+cu128
- Key paths: `/repo`, `/models/`, `/datasets/`, `/autosota_cache/`

## Key Code Paths

- Attack entry: `pipeline/run_attack.py`
- BIRA implementation: `attack_utils/attacks.py`
- Model wrapper: `attack_utils/utils.py`
- Model config: `model_config/llama3.1-8b-local.yaml`
- SIR watermark: `watermark/` directory
- Evaluation output: `experimental_results/BIRA/Llama-3.1-8B-Instruct/SIR/BIRA_beta_<B>_percentile_<P>_num_data_<N>.json`

## Safe Optimization Targets

1. **Beta parameter** (`--beta`): Baseline -4.0. Increasing magnitude (more negative) applies stronger logit bias. Risk: degeneration at extreme values. Mitigation: adaptive loop reduces beta when degeneration detected.
2. **Percentile** (`--percentile`): Baseline 50. Controls which tokens get biased (top percentile by self-information). Lower values = fewer tokens biased = more targeted.
3. **Repetition penalty** (`repetition_penalty` in model config): Baseline 1.0 (disabled). Enabling at 1.1 prevents n-gram repetition, reducing degeneration at aggressive betas.
4. **No-repeat ngram size** (`no_repeat_ngram_size`): Baseline 0 (disabled). Setting to 3 prevents 3-gram repeats.
5. **Learning rate** (`--learning_rate`): Baseline 0.125. Controls adaptive beta reduction step size.
6. **BIRA rewrite logic**: `BIRAAttack.rewrite()` in `attacks.py:508-564`. Core evasion loop.
7. **Logit processor**: `HighSurprisalLogitProcessor.__call__()` in `attacks.py:174-191`. Applies bias.
8. **Self-information**: `SelfInformationCalculator` in `attacks.py:122-171`. Identifies high-surprisal tokens.

## Constraints

- Do not modify metric definitions, scoring scripts, test data, labels, or dataset splits
- Do not hard-code predictions or metrics
- All changes must be inside container `/repo`
- Use `/tools/record_score.sh` for all score records

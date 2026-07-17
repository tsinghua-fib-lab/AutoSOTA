# Code Analysis: Paper 1798 SOTA Preparation Repair

## Original Preparation Failure

The preparation failed because:
1. **git not installed**: The container `autosota_repro_paper_1798` and the new `autosota_sota_paper_1798` both lacked `git`, which is required for baseline commit/tag operations.
2. **apt proxy failure**: The default apt configuration used a proxy (`http://127.0.0.1:7890`) that returned 502 Bad Gateway errors, preventing `apt-get install git`.
3. **Docker run rejection**: First attempt to start a new container with `--network host` was rejected by the Docker authorization plugin.

## Repair Actions

1. Installed `git` inside `autosota_sota_paper_1798` using `apt-get` with `-o Acquire::http::Proxy=false` to bypass the broken proxy.
2. Initialized git repo at `/repo` with baseline commit and `_baseline` tag.
3. Copied `record_score.sh` to `/tools/record_score.sh` in the container.
4. Ensured artifact directory `/autosota_artifacts/paper-1798/sota/` was writable.

## Corrected In-Container Evaluation Command

```bash
cd /repo/steering_watermark && python3 run_reproduction_v2.py
```

This runs:
- Generation of 1000 steered-A texts + 1000 steered-B texts using Llama-3.2-1B-Instruct
- Activation gathering from steering layer 15
- MLP training (1 epoch, Adam lr=0.001)
- Token-level and text-level F1 evaluation

## Baseline Verification

The baseline reproduced the manifest metrics exactly:
- Token-level F1: 65.5% (manifest: 65.5)
- Text-level F1: 75.9% (manifest: 75.9)

## Optimization Targets and Key Files

| File | Purpose | Safe to modify |
|------|---------|---------------|
| `run_reproduction_v2.py` | Main evaluation script, config, evaluation | Yes (config, voting) |
| `src/text_generation.py:generate_noise()` | Steering vector generation | Yes |
| `src/ml_model.py:SimpleMLP` | MLP classifier architecture and training | Yes |
| `src/llm_wrapper.py:SteeringHook` | Steering application to hidden states | Yes |
| `src/data_processing.py` | Data splitting and preprocessing | Caution |
| `src/detection.py` | Detection/evaluation | Caution |

## Results Summary

Block-sparse steering vectors (ALGO-4), aligned with transformer attention head dimensions (64-dim blocks matching 32-head Llama architecture), produced dramatic improvements:

| Iter | Idea | Token F1 | Text F1 | Alpha |
|------|------|----------|---------|-------|
| 0 | Baseline (random sparse 0.3%) | 65.5% | 75.9% | 5 |
| 2 | block_sparse_1 (1 head) | 97.4% | 99.3% | 5 |
| 3 | block_sparse_2 (2 heads) + conf voting | 99.2% | 99.5% | 5 |
| 4 | block_sparse_1 | 89.2% | 96.6% | 3 |
| 5 | block_sparse_1 | 76.3% | 86.8% | 2 |

The key insight: structured steering vectors aligned with transformer architecture create coherent, survivable signals that are far more detectable than random-sparse vectors.

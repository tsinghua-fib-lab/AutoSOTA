# Code Analysis: Paper 3105 - AI Engram SOTA Preparation Repair

## Original Failure

The SOTA preparation failed because:
1. **git not installed** in the `autosota/paper-3105:reproduced` Docker image
2. **apt-get failed** via proxy (172.17.0.1:17890 returned 502 for Ubuntu archive URLs)
3. The git init/commit/tag commands essential for the SOTA workflow could not execute

## Repair Applied

- Installed git using `apt-get` without proxy (direct network access works)
- Initialized git repo at `/repo`, committed baseline state, tagged `_baseline`
- Copied `/tools/record_score.sh` into container via `docker cp`

## Corrected In-Container Evaluation Command

```bash
cd /repo && python3 reproduce_v2.py
```

Runs inside container `autosota_sota_paper_3105`. No `docker exec` wrapper needed.

## Baseline Verification

- **ToW: 0.9607** — exact match to manifest baseline
- **forget_class_accuracy: 0.018** — matches manifest (0.018)
- **retain_class_accuracy: 0.9316** — matches manifest (0.9316)
- **test_accuracy: 0.8402** — consistent with original eval

Baseline recorded as iteration 0 in scores.jsonl.

## Pre-downloaded Resources (/paper_data)

| Resource | Path | Status |
|----------|------|--------|
| ResNet-18 CIFAR-10 weights | `/paper_data/resnet18_cifar10/pytorch_model.bin` | Used by eval |
| CIFAR-10 dataset | `/datasets/cifar10` | Used by eval |
| Engram checkpoints | `/paper_data/engram_checkpoints` | Available if needed |

## Safe Optimization Targets

The evaluation uses:
1. `notebook_engram.py:EngramEditor.compute_engram_weights()` — computes P = C_target @ pinv(C_total), then W_engram = W @ P
2. `notebook_engram.py:EngramEditor.apply_engram_weights()` — W_new = W - alpha * W_engram
3. `reproduce_v2.py:compute_tow()` — computes ToW metric

Safe modifications (no red-line violations):
- **Covariance shrinkage** (ALGO-02): Regularize C_total before pinv
- **Per-layer norm scaling** (CODE-01): Scale edit strength per layer by ||P||/||W||
- **Contrastive covariance** (ALGO-06): Use retain-only classes for C_total
- **Fisher importance** (ALGO-01): Scale edits by Fisher information ratio
- **Alpha sweep**: Vary alpha with best scaling combination

## Constraints

- Do NOT modify: metric definitions, test data, labels, dataset splits
- Do NOT hard-code predictions or metrics
- All code changes inside `/repo` in container

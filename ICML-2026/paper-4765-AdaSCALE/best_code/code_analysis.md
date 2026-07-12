# Code Analysis - Paper 4765 (AdaSCALE)

## Preparation Failure

### Root Cause
The original preparation failed because git was not installed in the autosota/paper-4765:reproduced image. The orchestrator tried to install git via apt-get in the reusable container but the dpkg lock was held. When falling back to a new container from the reproduced image, git was still missing and apt-get was in a broken state (dpkg interrupted).

### Repair Steps
1. apt-get update && apt-get install -y git in the SOTA container
2. git init and baseline commit/tag inside /repo
3. Created /tools/ and copied record_score.sh
4. Created /autosota_artifacts/paper-4765/sota/
5. Ran baseline evaluation and verified metrics match manifest

## Corrected Evaluation Command

    cd /repo && python3 scripts/eval_ood_imagenet.py --tvs-pretrained --postprocessor adascale_a --batch-size 64 --save-csv

### Verified Baseline
- FPR95 (nearood): 52.28 - matches manifest
- AUROC (nearood): 82.27 - matches manifest
- Datasets: ImageNet-1k val (29,461 images) + openimage_o OOD (17,632 images)
- Model: torchvision ResNet-50 IMAGENET1K_V1 (auto-downloaded)

## Container and Environment
- Container: autosota_sota_paper_4765
- Image: autosota/paper-4765:reproduced
- GPUs: 2 (indices 6,7 mapped inside container)
- PyTorch 2.1.2, CUDA 12.1
- Repo: /repo (https://github.com/sudarshanregmi/AdaSCALE, commit ed5f639)

## Key Source Files
- openood/postprocessors/adascale_postprocessor.py - Core postprocessor
- scripts/eval_ood_imagenet.py - Evaluation script
- configs/postprocessors/adascale_a.yml - AdaSCALE-A config
- configs/postprocessors/adascale_l.yml - AdaSCALE-L config

## Safe Optimization Targets
1. PARAM-01: Extend percentile range sweep
2. CODE-02: Increase num_samples for eCDF
3. ALGO-05: KDE-smoothed CDF for percentile mapping
4. ALGO-02: Pre-pooling channel statistics in Q
5. ALGO-04: BN-calibrated per-channel perturbation
6. ALGO-01: Score fusion (AdaSCALE-A x AdaSCALE-L)
7. CODE-04: FP16 perturbation forward pass
8. k1/k2 swap bug investigation

## Reproducibility Notes
- OOD datasets use openimage_o as substitute
- NFS cleanup errors at end of eval are harmless
- ACC=0.11 for AdaSCALE-A is a known reporting artifact (CODE-01)

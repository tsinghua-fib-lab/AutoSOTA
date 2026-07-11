# SOTA Preparation Repair — Paper 3508

## Original Failure

The preparation script failed because:
1. Git not installed in container (Ubuntu focal base image)
2. Apt proxy blocked archive.ubuntu.com (502 errors through proxy)

## Repair Applied

- Installed git via apt-get with NO_PROXY bypass
- Initialized git repo, created baseline commit and _baseline tag
- Created /tools/ and copied record_score.sh
- Created scores.jsonl at /autosota_artifacts/paper-3508/sota/

## Corrected Evaluation Command

    cd /repo/externalAerodynamics && python3 run_hsd_only.py

Same as manifest eval_command, runs inside autosota_sota_paper_3508.

## Baseline Verification

All metrics within numerical noise of manifest values. Baseline confirmed.

## Optimization Targets

- config.py: K_EIGENS, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, HSD_FNO_*
- training.py: train_HSD() loss weights, warmup, scheduler, EMA
- models.py: SpectralAmp, FNO capacity, spectral gate, mode weights
- Topology metrics and evaluation protocol MUST NOT change

# Code Analysis for Paper 698: CurriSeg SOTA Optimization

## Evaluation Path
- Eval script: /repo/eval_repro.py
- Command: CUDA_VISIBLE_DEVICES=0 python3 eval_repro.py
- Checkpoint: /models/pretrained/Curriseg/model.pth (297MB, author-provided)
- Model: Network(channels=192) with ResNet50 backbone from lib/Network.py
- Test data: /datasets/TestDataset/COD10K/ - 2026 image/GT pairs
- Output: Stdout line M: <float> - MAE with 6 decimal places
- Metric computation: Per-image MAE averaged over all images
- PySODMetrics available: v1.6.2 - can compute Fbeta, Ephi, Salpha, MAE

## Training Path
- Script: /repo/Train.py (curriculum + TSSW + PUE)
- Anti-curriculum: /repo/anti_curri_stage.py (SBFT + hard sample fine-tuning)
- Data loader: utils/data_val.py - PolypObjDataset with augmentations
- Training data: NOT available (/datasets/TrainDataset/ missing)
- Network: lib/Network.py (FEDER architecture), lib/Modules.py (GPM, REM11, GCM3)
- Backbone: ResNet50 pretrained at /models/resnet50_imagenet1k_v1.pth

## Config Path
- CLI args in Train.py control all hyperparameters
- TSSW params: tssw_K, tssw_wmin, tssw_sigma_star, tssw_gamma
- PUE params: pue_wmin, pue_Tc
- Anti-curriculum: hard_ratio, use_sbft, sbft_prob, sbft_kernel, sbft_sigma

## Safe Modification Targets (inference-only, no training needed)
1. eval_repro.py - TTA, resolution sweep, BN recalibration, resize method
2. Train.py - loss functions, curriculum schedule, TSSW params (requires training data)
3. anti_curri_stage.py - SBFT params, hard ratio schedule (requires training data)
4. lib/Network.py - normalization layers (requires retraining)
5. utils/data_val.py - augmentations (requires retraining)

## Risky Files (do not modify)
- Test data: /datasets/TestDataset/COD10K/ - immutable
- Scoring: /tools/record_score.sh - immutable

## Current Baseline
- M: 0.0307 (paper: 0.030)
- Fbeta: ~0.738 (paper: 0.736)
- Ephi: ~0.905 (paper: 0.910)
- Salpha: ~0.909 (paper: 0.818, MATLAB vs Python diff)

## Key Constraints
- Training data unavailable - only inference-time optimizations feasible
- Max 15 min eval timeout
- GPU devices: 2,3

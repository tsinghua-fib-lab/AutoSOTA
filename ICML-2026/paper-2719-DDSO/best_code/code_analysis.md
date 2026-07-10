# Code Analysis - Paper 2719 HICALD SOTA

## Key Files
-  — Main evaluation script. Trains ICALD_Classifier on Heart-Disease for 5 runs.
-  — Model definitions: ICALD_Classifier, ALD_Classifier, Softmax_Classifier, etc.
-  — Evaluation metrics: Accuracy, ECE, KCE, GroupX_ECE, GroupY_ECE, AP, AUC.
-  — Data loading: Heart-Disease (23 features, 5 classes), 299 samples after preprocessing.

## Baseline Parameters
- lambda_reg=0.9, t=0.5, K=5, lr=1e-3, batch_size=128, epochs=600, patience=50
- Early stopping: score = ACC - ECE, min_epochs=20, patience=50
- mc_samples: 100 (early stopping), 2000 (final eval)

## Architecture
- ICALD_Classifier: 3-layer MLP with residual connections
  - layer1: Linear(input_dim, n_hidden) + residual projection
  - layer2_*: 3 branches (theta, sigma, kappa): Linear(n_hidden + q_dim, n_hidden//2)
  - layer3_*: 3 branches: Linear(n_hidden//2 + q_dim, output_dim)
  - q embedding: Linear(1, 2)→Linear(2, 1)→exp
  - Dropout 0.5, NO BatchNorm, NO LayerNorm

## Modifiable Targets
1. ICALD_Classifier in model_all.py — architecture changes (BN, more layers, etc.)
2. Training loop in run_reproduction_fast.py — LR schedule, loss functions, sampling
3. Post-hoc calibration wrapper — temperature scaling, ensemble averaging

## Risky Files (DO NOT MODIFY)
- utils.py evaluate() function — metric definitions
- datasets.py data loading — test splits
- /tools/record_score.sh — scoring script

## Reusable Resources
- Container: autosota_repro_paper_2719 with PyTorch 2.1.2, CUDA 12.1
- Dataset: /repo/datasets/classification/heart_disease/heart_disease_uci.csv
- Cache: /autosota_cache, /datasets, /models

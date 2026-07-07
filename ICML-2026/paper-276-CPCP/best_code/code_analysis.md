# Code Analysis — Paper 276: CPCP (Colorful Pinball Conformal Prediction)

## Evaluation Path
- **Script**: run_naval_cpcp.py
- **Entry**: main() — runs 20 seeds, each training CPCP-Clip+Mix and RCP-Pinball
- **Data split**: 60:20:20 (train:cal:test) via rcp_protocol_split(), then calibration split into 40:40:20 for three CPCP phases
- **Metrics computed**: Cov, Size, WSC, MSCE_10, MSCE_30, L1-ERT, L2-ERT via get_metrics_nd()

## Train/Inference Path
1. Mean network: Net(16,2) trained with HuberLoss, 100 epochs, LR=2e-3
2. Quantile network: MonotonicThreeHeadNet trained with pinball loss, 200 epochs (Phase 1)
3. Fine-tuning: finetune_main_head_improved() — only head_main trained with density-weighted pinball loss, 200 epochs (Phase 2)
4. Conformalization: Scores computed on X_score, quantile q determined

## Config Path
- Hyperparameters: Defined inline in run_naval_cpcp.py (alpha=0.1, epsilon=0.02, clip_max=5.0, mix_ratio=0.5)
- Architecture: models.py — 3-layer MLP, hidden_dim=256, Softplus for gap activations
- Training: trainers.py — Adam LR=2e-3, batch_size=1024, no LR schedule

## Metric Parser
- get_metrics_nd() in metrics.py returns dict: Cov, Size, WSC, MSCE_30, MSCE_10, L1-ERT, L2-ERT
- Size = mean per-dimension log volume (Volume metric)
- MSCE_10 = K-means with K=10 partitions (primary MSCE)
- MSCE_30 = K-means with K=30 partitions
- WSC = Worst-Slice Coverage with M=1000 projections
- ERT = Excess Risk Test with logistic regression, 5-fold CV

## Reusable Resources
- Datasets: /repo/Datasets/naval.csv (1.4MB)
- Cache mounts: /autosota_cache, /datasets, /models

## Risky Files (DO NOT MODIFY)
- metrics.py — metric definitions (red-line constraint)
- data_utils.py — data loading
- losses.py — pinball loss definition (core method)

## Safe Modification Targets
- trainers.py — training loops, LR schedules, weight computation, fine-tuning
- models.py — network architecture, activation functions
- methods.py — conformalization procedure, calibration splits
- run_naval_cpcp.py — hyperparameters (epsilon, epochs)

## Key Findings
1. NON-DETERMINISM: methods.py:151 uses np.random.permutation(n) without seeding
2. HARD CLAMP: trainers.py uses torch.clamp(q_diff, min=1e-4) destroying weight ranking
3. FIXED LR: All trainers use constant LR=2e-3 with no schedule
4. STATIC WEIGHTS: Density weights computed once before fine-tuning
5. SOFTPLUS: Produces near-zero gaps causing exploding density weights

## Evaluation Command (in-container)
cd /repo && CUDA_VISIBLE_DEVICES=1 python3 run_naval_cpcp.py

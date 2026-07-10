# Code Analysis — IMPACT (Paper 2344)

## Evaluation Path
- **Entry**: `main.py` → `IMPACT.fit()` → `IMPACT.decision_function()` → `evaluation.metrics.get_metrics()`
- **Output format**: `IMPACT-SMD, AUC: X.XXXX` per run; average line format: `IMPACT-SMD, AUC: X.XXXX, Time: ... avg`
- **Primary metric**: `AUC_ROC` (first element from `get_metrics()`) — interval-level ROC-AUC
- **All metrics returned**: AUC_ROC, AUC_PR, PointF1, Precision, Recall

## Training Path
- `IMPACT.fit()` → `_training()` → `training_forward()` per batch
- **Epochs 0-8**: Simple deviation loss on seen_head outputs (normal vs labeled anomalous)
- **Epoch >= 9**: Influence computation via HVP → pseudo-anomaly generation → unseen loss via pseudo_head
- **Influence flow**: val set reference → hvp → per-sample gradient → influence score → top-K harmful (perturb) + helpful (reference) samples
- **Inference**: TCN feature → seen_head + pseudo_head scores + DSADLoss distance → max score

## Config Path
- **CLI args**: `main.py` argparse (data, num_epochs, epoch_steps, batch_size, lr, rep_dim, hidden_dims, k, lambd, runs)
- **Model configs**: Dict passed to IMPACT constructor
- **Hardcoded values**: alpha=0.02 (perturbation strength in impact.py line with `0.02 * torch.sign`), max_con_num=5, max_per_num=k, max_ref_num=k

## Metric Parser
- stdout line: `IMPACT-SMD, AUC: X.XXXX` → extract AUC value, multiply by 100 for percentage
- Average line: `IMPACT-SMD, AUC: X.XXXX, Time: Y.YYs, avg` → this is the primary metric value

## Reusable /paper_data
- Not mounted. SMD data is preprocessed from OmniAnomaly source and stored in `/repo/datasets/SMD/`

## Risky Files (do not modify)
- `evaluation/metrics.py` — metric computation
- `evaluation/basic_metrics.py` — core metric functions
- `evaluation/cal_vus.py` — VUS computation
- `utils.py` — data loading (dataset splits)

## Safe Modification Targets
- `model/network.py` — HolisticHead (BN→GN), TCNEncoder, IMPACTNet architecture
- `model/impact.py` — training_forward (alpha, influence, epoch threshold), _training (scheduler), loss functions
- `main.py` — CLI defaults, config passing

## Key Architecture Details
- TCN: uses `weight_norm` (NOT BatchNorm) — verified in TcnResidualBlock
- BatchNorm1d: only in HolisticHead.fc1 (both seen_head and pseudo_head)
- Feature dim: rep_dim=64 → TCN output is 64-dim L2-normalized vectors
- HolisticHead: Linear(64,64) → BatchNorm1d → ReLU → Dropout → Linear(64,3)
- Perturbation: gradient-sign in feature space, alpha=0.02, applied at epoch >= 9
- Reference set: collected from normal_helpful samples, used in inference for DSADLoss center

## Current Baseline
- AUC: 74.84% (paper: 75.97%)
- Evaluation: `CUDA_VISIBLE_DEVICES=0,1 python3 main.py --data SMD --setting general --num_epochs 10 --batch_size 64 --lr 0.0003 --rep_dim 64 --hidden_dims 64 --k 5 --lambd 1.0 --runs 5`

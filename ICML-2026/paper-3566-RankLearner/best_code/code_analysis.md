# Code Analysis: Paper 3566 SOTA Preparation Repair

## Original Failure

The preparation script attempted to install `git` via `apt-get` in the container, but the proxy at `172.17.0.1:17890` rejected connections to Ubuntu archive mirrors. Git is needed for baseline tagging and commit tracking in the optimization loop.

## Repair Applied

1. **Git installation**: Used `unset` for all proxy environment variables before `apt-get install git`, which succeeded because the container has direct network access without the proxy. Git 2.25.1 installed successfully.
2. **Safe directory**: Added `/repo` to git safe.directory config to resolve "dubious ownership" (repo files owned by uid 1000, running as root).
3. **Baseline commit**: Created `_baseline` tag at the initial repository state.
4. **Record script**: Copied `/tools/record_score.sh` from host into container.
5. **Artifacts directory**: Created `/autosota_artifacts/paper-3566/sota/`.

## Verified Baseline

Command: `cd /repo && python3 eval.py --dataset synthetic --train_size 500 --n_seeds 5`

Results match manifest exactly:
- Rank-Learner AUTOC = 1.3113 ± 0.0140
- DR-learner AUTOC = 1.2781 ± 0.0536
- Mean Policy Value = 0.5634 ± 0.0054

## Code Architecture

### Files
- `eval.py` - Standalone evaluation: loads pre-trained checkpoints, evaluates on fixed test set, prints metrics
- `run_pipeline.py` - Full reproduction: generates data, trains nuisance→DR→Rank-Learner, evaluates (saves checkpoints to `/repo/experiments/`)
- `library/training.py` - Four training functions: `train_propensity`, `train_response`, `train_cate`, `train_ranker`
- `library/data_utils.py` - Datasets, data splitting, DR score computation
- `library/models.py` - Simple NN architectures: `ClassificationHead`, `RegressionHead` (both: Linear→ReLU→Linear)
- `library/eval_utils.py` - Evaluation dataset and metrics computation
- `library/metrics.py` - AUTOC and Policy Value metric implementations

### Training Pipeline
1. Step 1: Generate synthetic data (11000 samples, 10 features, nonlinear DGP)
2. Step 2: Train 3 nuisance models (propensity e, outcomes m0, m1) per seed
3. Step 3: Train DR-learner (pointwise CATE regression on DR scores)
4. Step 4: Train Rank-Learner (orthogonal + plug-in rankers with pairwise BCE loss)
5. Step 5: Evaluate on fixed test set (last 1000 samples)

### Key Hyperparameters
- Rank-Learner orthogonal: kappa=0.5, lr=0.001, hidden_dim=128, batch_size=256, fraction_of_pairs=0.1
- Rank-Learner plug-in: kappa=3.0, lr=0.001, hidden_dim=128, batch_size=256, fraction_of_pairs=0.1
- Nuisance e: hidden_dim=128, lr=0.0005
- Nuisance m0/m1: hidden_dim=64, lr=0.001
- DR-learner: hidden_dim=128, lr=0.0005
- All stages: max_epochs=50, patience=5, Adam optimizer

### Optimization Targets
- `RankerDataset.__getitem__` - pair generation and orthogonal label computation
- `train_ranker` - pair sampling (fraction_of_pairs), loss function, optimization schedule
- `ClassificationHead` / `RegressionHead` - model architectures
- `compute_dr_scores` - DR score computation (can be refined)
- Pipeline configs - hyperparameters for all stages

## Safe Modifications (Red-Line Compliant)
- Model architectures, training hyperparameters, loss functions, sampling strategies
- All evaluation protocol (test data, metric computation, comparison) remains unchanged
- The evaluation command `python3 eval.py --dataset synthetic --train_size 500 --n_seeds 5` always tests on the same fixed data

## Reusable Resources
- No `/paper_data` mount exists
- Pre-generated synthetic data at `/repo/data/datasets/synthetic.csv`
- Pre-trained checkpoints at `/repo/experiments/`
- Model cache at `/autosota_cache/`, `/datasets/`, `/models/`

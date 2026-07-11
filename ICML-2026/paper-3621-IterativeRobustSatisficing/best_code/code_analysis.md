# Code Analysis - Paper 3621: IRS on CIFAR-10-LT IF=50

## Evaluation Path
- **Entry**: `code/run_irs_if50.py` → `CIFAR10LTRunner(CONFIG).run()`
- **Config**: `CIFAR10LTConfig` with IF=50, seed=123, WRN-28-10, IRS only
- **Metric output**: Stdout prints final metrics; CSV files written to `cifar10lt_results/<timestamp>/`
  - `irs_seed123.csv`: per-epoch metrics (test_acc, test_balanced_acc, test_worst_acc, kappa)
  - `irs_classwise_seed123.csv`: per-class accuracy (Tail = mean of horse, ship, truck classes 7,8,9)

## Key Files
- `code/cifar10lt_experiment.py` (1310 lines): Main orchestrator, configs, data loading, WRN model, evaluation
- `code/methods/irs.py` (343 lines): IRSObjective, IRSTrainer
- `code/methods/sam.py`: SAM implementation (reference for AMP pattern)
- `code/methods/erm.py`: ERM baseline with AMP

## Metric Parsing
- **Avg**: `test_acc` from CSV or stdout → `{test_acc:.3f}`
- **Tail**: Mean of per-class accuracies for classes horse(7), ship(8), truck(9) from classwise CSV
- **Worst**: `test_worst_acc` from CSV or stdout → min across all 10 classes
- Baseline: Avg=0.6332, Tail=0.434, Worst=0.387

## Key Code Locations
- `TRAIN_TRANSFORM` (line ~496): ToTensor + Normalize ONLY (no augmentations)
- `make_irs_config()` (line ~236): IRSConfig with weight_decay from cfg.weight_decay (0.0)
- `IRSTrainer.train()` (line ~260-343 in irs.py): training loop, no AMP, no grad clipping
- `IRSObjective.training_loss()` (line ~138): core IRS loss computation
- `build_loaders()` (line ~788): standard shuffle=True DataLoader (instance-balanced)

## Safe Modification Targets
1. `TRAIN_TRANSFORM` in cifar10lt_experiment.py: add data augmentations (training only)
2. `IRSTrainer.train()` in irs.py: add gradient clipping, AMP, EMA
3. `IRSConfig` in irs.py: hyperparameter tuning
4. `run_irs_if50.py`: config modifications (weight_decay, epochs, learning rates)
5. `IRSObjective` in irs.py: algorithmic modifications

## Red Lines (DO NOT MODIFY)
- Metric computation functions (evaluate_classification, evaluate)
- Test dataset, splits, labels
- Scoring script (/tools/record_score.sh)
- Tail class definitions (horse, ship, truck = classes 7, 8, 9)
- WideResNet architecture

## Manifest Command Correction
- Manifest eval_command: `cd /repo/code && python3 run_irs_if50.py` → runs correctly inside container
- No correction needed

## Baseline Confirmation
- iter-0 baseline recorded in scores.jsonl: Avg=0.6332, Tail=0.434, Worst=0.387
- Commit: 552ffc1, tag: _baseline

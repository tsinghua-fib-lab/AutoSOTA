# Code Analysis — Paper 1164 (SMART)

## Evaluation Path
- `evaluate.py` — main entry point. Loads ResNet-50, computes test logits, runs SMART calibration across 5 random seeds, reports ECE/AdaECE/Accuracy/NLL.
- `python3 evaluate.py` — the container eval command.

## Key Files
| File | Purpose | Safe to Modify |
|------|---------|----------------|
| `evaluate.py` | Evaluation harness + model definition | Calibration split logic, SMART hyperparameters |
| `smart.py` | SMART calibration core (MarginTemperatureNet + SMART.fit/calibrate) | Model architecture, training loop, optimizer, early stopping |
| `losses.py` | CharbonnierSoftECE loss | Loss function implementation |
| `metrics.py` | ECE, NLL, Accuracy evaluation metrics | DO NOT MODIFY (this would violate red-line) |
| `results_cifar100_resnet50.json` | Stored results from paper | Read-only reference |

## Config Path
- Parameters are hardcoded in `evaluate.py` (line ~145-160) and passed as constructor arguments to `SMART()`.
- No separate config file (config.yaml not present).

## Metric Parser
- Parse from FINAL RESULTS section of stdout: `ECE: X.XX +/- Y.YY%`, `AdaECE: X.XX +/- Y.YY%`, `Acc: X.XX +/- Y.YY%`, `NLL: X.XXXX +/- Y.YYYY`

## Reusable Resources
- `/models/focal_calibration/resnet50_cross_entropy_350.model` — pre-trained ResNet-50 (91MB)
- `/datasets/cifar-100-python` — CIFAR-100 dataset already extracted

## Risky Files (DO NOT MODIFY)
- `metrics.py` — contains metric definitions (ECE, accuracy, NLL)
- `/tools/record_score.sh` — score recording script
- Model definition in `evaluate.py` (CIFARResNet, Bottleneck classes)
- Data loading/transform logic in `evaluate.py`
- Dataset splits / test data

## Safe Modification Targets
- `smart.py` `SMART.__init__()` — add new hyperparameters
- `smart.py` `SMART.fit()` — modify training loop, add validation split, regularization, consistency loss
- `smart.py` `MarginTemperatureNet` — modify architecture (input dim, layers)
- `smart.py` `_normalized_margins()` — numerical stability changes
- `losses.py` `CharbonnierSoftECE` — sigma schedule, new loss variants
- `evaluate.py` SMART constructor call — adjust hyperparameters, calibration split strategy

## Baseline Metrics
- ECE: 1.83% (SMART, 5 seeds mean)
- AdaECE: 2.03%
- Accuracy: 77.30%
- Vanilla ECE: 17.27%, Vanilla Acc: 77.23%

## Known Bottlenecks
1. Training loss for early stopping → overfitting risk (CODE-01)
2. No weight decay → overfitting on small calibration set (CODE-02)
3. Single-margin input → limited expressivity (ALGO-04)
4. No consistency regularization → temperature instability (ALGO-01, ALGO-05)
5. CharbonnierSoftECE sigma fixed → suboptimal binning (ALGO-07, PARAM-01)

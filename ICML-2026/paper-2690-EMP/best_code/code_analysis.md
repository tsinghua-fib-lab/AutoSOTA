# Code Analysis — Paper 2690: Effective Model Pruning

## Evaluation path
- eval.py (standalone, 200 lines) — single entry point for train + prune + eval
- Training: train() uses Adam, lr=1e-4, 5 epochs, ConstantLR (no decay)
- Pruning: prune_neurons_fc() uses L2 norm of weight rows as importance
- Evaluation: evaluate() computes CrossEntropy loss + accuracy on test set

## Key files
| File | Role | Safe to modify? |
|------|------|-----------------|
| eval.py | Main script | Yes — structural/algorithmic changes |
| checkpoints/FC5_FashionMNIST_best.pth | Pre-trained checkpoint | Read-only; delete to force retrain |
| data/ | Auto-downloaded FashionMNIST | Do not touch |

## Metric parser
- JSON output at results/reproduction.json
- Key field: emp_accuracy (float, percentage)
- Also: dense_accuracy, structural_sparsity, accuracy_delta

## Bottleneck analysis
- Root cause: Neuron weight L2 norms are nearly uniform in well-trained FC5
  => N_eff ~= N per layer => beta=1.0 retains almost all neurons => 0.2% sparsity
- Fix vectors:
  1. Change scoring function (BN gamma, activation mean, gradient product)
  2. Change training to produce differentiated weights (dropout, L1 reg, more epochs)
  3. Iterative pruning (IMP-style) to build up variance across rounds

## Safe modification targets
1. FCNet.__init__ — add regularization layers (Dropout, BatchNorm)
2. train() — change optimizer, epochs, LR schedule, add L1 penalty
3. get_dataloaders() — add data augmentation transforms
4. prune_neurons_fc() — change scoring function
5. main() — add iterative pruning loop, new CLI args

## Red-line boundaries
- Test data MUST NOT be used for training, pruning decisions, or scoring
- Metric computation (evaluate()) is fixed and unchanged
- Output format (JSON with emp_accuracy) stays the same
- Seed 0 for reproducibility

# Code Analysis — Paper 5170 PVI SOTA Optimization

## Evaluation Path
- Main eval script: run/run_pdb_eval.py — does full training + evaluation in one run
- Training script: run/run_pdb.py — simpler training script (not used for SOTA evals)

## Key Files
| File | Role | Safe to modify? |
|---|---|---|
| run/run_pdb_eval.py | Main eval/training pipeline | Yes — training hyperparams, optimizer, gradient handling |
| posterior/basic.py | Diagonal Gaussian posterior (10 params) | No — paper-defined |
| posterior/basic_fullrank.py | Full-rank Gaussian posterior (15 params) | No — paper-defined |
| objective/pacmvi_basic.py | PVI-Log objective (log score) | No — paper-defined |
| objective/kl_prior.py | KL Prior regularizer | No — paper-defined |
| objective/pvi_crps.py | PVI-CRPS objective | No — paper-defined |
| jax_posteriordb/model/kidscore_interaction.py | KidScore model (d=5, N=434) | Yes — initialization, new methods |

## Metric Parser
Metrics are parsed from stdout of run_pdb_eval.py:
- Test Log Score: VALUE
- Test CRPS: VALUE
- Test IS (alpha=0.1): VALUE

## Baseline Metrics
- test_log_score: -380.64 (paper: -369.91, VI baseline: -590.67)
- test_crps: 1283.24 (paper: 873)
- test_is: 6927.6

## Gradient Handling
- grads/n normalization with n=434
- Regularized objective compensates: g1 + lamb * g2 * n
- Patched in commit a0ef803

## Dataset
- kidiq.json included at jax_posteriordb/data/kidiq.json (N=434)
- Fixed seed 0 for 60/20/20 split

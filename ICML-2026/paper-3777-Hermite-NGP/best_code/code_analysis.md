# Code Analysis for Hermite-NGP (Paper 3777)

## Key Paths
- **Training**: /repo/examples/helmholtz2d.py (851 lines)
- **Evaluation**: /repo/eval_helmholtz2d.py
- **Model**: /repo/hermite_ngp/models/hermite_pinn.py
- **Encoding**: /repo/hermite_ngp/encoding/hermite_encoding_cuda.py
- **Baseline checkpoint**: /models/helmholtz2d_compact_seed456.npz

## Baseline Configuration
| Parameter | Value |
|-----------|-------|
| hash_size (all 3 tables) | 12 (4096 entries each) |
| hidden_dim | 128 |
| n_layers | 2 |
| omega | 0.5 |
| n_levels | 8 |
| per_level_scale | 2.0 |
| lr | 1e-3 |
| n_epochs | 100000 |
| collocation | 10000 |
| bc_per_edge | 5000 |
| seed | 456 |

## Metric Parser
- Regex from eval stdout: Relative L2 Error line

## Safe Modification Targets
1. examples/helmholtz2d.py - collocation point sampling, GradNorm block, EMA class
2. hermite_ngp/encoding/hermite_encoding_cuda.py - hash table sizes, per-level config
3. CLI args - hash-size, hidden, layers, omega, warm-restart, etc.

## Red-line Boundaries
- Do NOT change: eval_helmholtz2d.py, exact solution, eval grid, metric computation
- Do NOT change: PDE definition, source term, domain

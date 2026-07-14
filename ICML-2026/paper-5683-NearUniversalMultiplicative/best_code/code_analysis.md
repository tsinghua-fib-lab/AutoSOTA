# Code Analysis: Paper 5683 - NNEinFact

## Evaluation Path
- `eval.py` — 10-fold random split evaluation
- Each split: creates NNEinFact, fits with early_stopping, records heldout loss + runtime
- Output: stdout with HELDOUT_LOSS=X and RUNTIME=X, plus eval_results.json

## Core Implementation
- `einfact.py` (213 lines) — NNEinFact class with Multiplicative Updates
- Model: `wr,dr,hr,irk,jkr->wdhij` (5 factors)
- Uber data: 27x24x7x100x100, 99.4% sparse, 284K nonzero entries

## Key Parameters
- k=6 (spatial latent), r=10 (temporal latent), alpha=0.7, beta=0.0
- max_iter=5000, early_stopping=True, 10 splits at 90/10 train/holdout

## Metric Computation
- Heldout loss: (alpha,beta)-divergence between Y and Y_hat on heldout mask
- Parsed from stdout: HELDOUT_LOSS=<float>, RUNTIME=<float>

## Safe Modification Targets
- `einfact.py:fit()` — inner optimization loop (lines 145-204)
- `einfact.py:_initialize_params()` — uniform init (lines 66-70)
- `einfact.py:prepare_masks()` — validation split (lines 94-110)
- `eval.py` — config section (k, r, alpha, max_iter, seed)

## Risky Files (DO NOT MODIFY)
- `data/Y.npz` — test data
- Metric computation in `_calculate_ab_divergence()`
- Evaluation protocol in eval.py (split scheme, metric output format)

## Key Bottlenecks
1. Y_hat recomputed per-factor (line 156, inside factor loop) — 5x per iteration
2. Uniform random init ignores data structure (99.4% sparse pickup data)
3. No momentum/acceleration — standard MU descent
4. Factor scale drift possible due to scale indeterminacy

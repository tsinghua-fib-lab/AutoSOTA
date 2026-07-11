# Code Analysis — Paper 3407 SOTA

## Evaluation Path
- `eval_final.py` — main evaluation script. Runs 20 trials × 10 reps per trial.
- Gap computed as average Euclidean distance between estimated frontier and true frontier.
- True frontier computed from 5000 MC samples (seed 12345).
- B estimated from 100 MC samples (seed 9999).
- CREME time measured per trial for one full CREME call.

## Key Source Files
| File | Role | Safe to modify? |
|------|------|-----------------|
| `eval_final.py` | Evaluation pipeline | Yes — config, B estimation, estimation strategy |
| `model/icrc.py` | ICRC estimator | Yes — estimate() formula, offset |
| `model/creme.py` | CREME frontier constructor | Read-only |
| `optimization/lp.py` | LP solver | Yes — warm start, caching |
| `optimization/base.py` | Base class | Read-only |

## Metric Parser
- Parsed from stdout lines matching `Gap:` and `CREME Time:`
- JSON metrics available after `__METRICS_JSON__` marker.
- Record script expects: `Gap` (lower) and `Time` (lower).

## Reusable Resources
- None — fully synthetic data. No datasets, models, or checkpoints.

## Risky Files (Do Not Modify)
- `model/creme.py` — core CREME algorithm
- `optimization/base.py` — base class
- `/tools/record_score.sh` — scoring infrastructure
- `/autosota_artifacts/paper-3407/sota/scores.jsonl` — scores file

## Safe Modification Targets
1. `eval_final.py`:
   - B estimation: per-lambda B, multi-seed, percentile
   - N_SAMPLES, N_LAMBDA, N_TRIALS configuration
   - offset parameter pass-through
   - MC diagnostic pass
2. `model/icrc.py`:
   - estimate() formula (offset handling, per-lambda B)
3. `optimization/lp.py`:
   - warm_start, solver selection, problem caching

## Conformal Correction Formula
```
regret_hat = regret.mean() * n/(n+1) + (B + offset)/(n+1)
miscoverage_hat = miscoverage.mean() * n/(n+1) + (1 + offset)/(n+1)
```
- B = 1.427 (global max over 100 MC samples)
- n = 10 → n/(n+1) ≈ 0.909, B/(n+1) ≈ 0.130
- The B/(n+1) term dominates small-regret estimates
- Per-lambda B would vary from ~0.08 to ~1.43 over lambda range

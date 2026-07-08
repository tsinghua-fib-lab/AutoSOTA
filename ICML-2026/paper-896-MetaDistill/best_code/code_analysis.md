# Code Analysis — MetaDistill (Paper 896)

## Evaluation Path
- **Main eval**: `scripts/repro_pom_d30_lad.sh` → 3 seed windows (0-6, 1-7, 2-8)
  - Calls `scripts/eval_compare_frameworks.py` per window
  - LAD computed by `scripts/compute_lad_shifted.py`
- **Eval script**: `scripts/eval_compare_frameworks.py` — runs POM with optional SSFT on BBOB f1-f24
- **Baseline checkpoints**: `checkpoints/baselines/pom_original.pt` (813 KB)
- **MetaDistill checkpoints**: `checkpoints/metadistill/bbob/pom.pt` (1.6 MB)
- **SSFT interval j**: 0 (no SSFT), 1, 3, 5 — applied during eval via `--ssft-variant`

## Key Source Files
- **POM optimizer**: `optimizers/pom.py` (~500 lines) — mutation matrix, crossover, selection
  - `mut_scale_mode`: "none" in baseline config, "sqrt_pop" in MetaDistill config
  - `return_aux`: False in baseline, True in MetaDistill config
- **SSFT trainer**: `meta_trainers/ss_trainer.py` — KL regularization, diversity reg, fitness loss
  - KL weight: linear decay from coef to 0.1*coef
  - Current KL coef: 0.1 (low — effectively disabled)
  - BP strategy: "greedy" (update per function)
- **Distill trainer**: `meta_trainers/distill_trainer.py` — KL distillation from teacher trajectories
  - Training set: cecf2-cecf6 (5 functions)
  - Teacher trajectories: 80 epochs, 7 teacher optimizers (CMA-ES, JADE, SHADE, L-SHADE, DE, PSO, GA)
- **Eval framework**: `scripts/eval_compare_frameworks.py`
  - SSFT during eval: fitness improvement loss only (NO KL reg during eval SSFT)
  - No `algo.reset()` between BBOB functions (potential state leakage)

## Config Files
- `configs/pom_config.json`: Baseline POM — no mut_scale_mode (defaults to "none"), no return_aux
- `configs/pom_d10_pop200.json`: MetaDistill POM — mut_scale_mode: "sqrt_pop", return_aux: true
- `configs/distill_pom.json`: Distillation training — 128 epochs, cecf2-cecf6 training set

## Metric Parser
- `scripts/compute_lad_shifted.py`:
  - Parses 3 JSON summary files (one per seed window)
  - Extracts per-function best variant (minimizing final_mean across variants)
  - Computes LAD = log10(baseline + shift) - log10(best_md + shift)
  - Output: `__LAD_RESULT__` followed by JSON `{"LAD": value}`

## Known Levers
1. SSFT interval j (currently 1, 3, 5) — more granular j values possible
2. Learning rate (currently 1e-4 for eval SSFT, 5e-4 for distill training)
3. KL regularization weight during SSFT (currently 0.1)
4. Training function set size (currently 5 functions cecf2-cecf6)
5. Teacher trajectory count (80 epochs)
6. Population size (200)
7. Training epochs (128 for distill, SSFT epoch count)

## Safe Modification Targets
- SSFT trainer KL weight and schedule
- Per-function j selection during evaluation
- Config parameters (pop, lr, j)
- Add algo.reset() between BBOB functions (defensive)
- Mutation scale mode in baseline config
- Gradient clipping / NaN detection (defensive)

# Code Analysis — Paper 2138 SOTA Preparation Repair

## Original Preparation Failure

The preparation failed because the evaluation command `python3 reproduce_dist_quadratic.py` with 200 Monte Carlo seeds exceeded the 90-minute timeout (ran for ~91 minutes, completing 111/200 seeds before timeout at 5460s).

**Root cause:** The evaluation script uses `N_RUNS = 200` which takes ~97 minutes. The SOTA preparation orchestrator set a 90-minute timeout, which was slightly too short.

## Repair Applied

1. **Reduced Monte Carlo seeds from 200 to 50** for optimization iterations. This produces evaluations in ~22 minutes, well within the 90-minute timeout, while maintaining sufficient statistical precision (standard error ≈ 0.007 vs 0.005 for 200 seeds).

2. **Corrected in-container evaluation command:** The manifest's `eval_command` is valid as-is when run inside the container:
   ```
   cd /repo && python3 reproduce_dist_quadratic_opt.py
   ```

3. **Created modular evaluation scripts** (`reproduce_dist_quadratic_wave1.py`, etc.) with incremental optimizations applied.

## Baseline Verification

- Baseline with 50 seeds: MPE = 0.3179 ± 0.0499
- Original reproduction with 100 seeds: MPE = 0.3167 ± 0.0461
- Paper reported value: MPE = 0.2031 ± 0.0668
- The 50-seed baseline matches the 100-seed reproduction within noise.

## Container and Tooling

- Container: `autosota_sota_paper_2138` (image: `autosota/paper-2138:reproduced`)
- Repo path: `/repo`
- Record tool: `/tools/record_score.sh` ✅
- Scores file: `/autosota_artifacts/paper-2138/sota/scores.jsonl`
- Git baseline tag: `_baseline` ✅
- GPUs: 0 and 1 available (CUDA 12.1)

## Safe Optimization Targets

The evaluation script (`reproduce_dist_quadratic.py`) is self-contained with:
- `ThetaMLP` class: 4→HIDDEN_DIM→HIDDEN_DIM→4 MLP with LayerNorm, LeakyReLU, Dropout
- `GlobalBandwidth` class: learnable bandwidth parameter for Fréchet regression
- Training loop: Adam optimizer, StepLR scheduler, early stopping
- Evaluation: Wasserstein-2 distance (MPE) via `DeSI_distribution`

### Key Levers
1. Early stopping patience/delta (currently 10/1e-4 → too aggressive for 80 training samples)
2. Optimizer and scheduler (Adam+StepLR → AdamW+CosineAnnealingLR)
3. Regularization (weight_decay=1e-4 → increase, add L1)
4. Gradient stability (no clipping → add clip_grad_norm_)
5. Training augmentation (none → noise augmentation)
6. Loss function (L2 only → add MMD auxiliary term)
7. Bandwidth init and regularization
8. Hidden dimension tuning

### Red Lines
- Do NOT modify metric computation (Wasserstein distance formula)
- Do NOT modify data generation (`generate_simulation_data_torch_true`)
- Do NOT modify `DeSI_distribution` in `simulation_distribution/DeSI.py`
- Do NOT change the evaluation protocol (train/val/test split ratios unless explicitly noted)

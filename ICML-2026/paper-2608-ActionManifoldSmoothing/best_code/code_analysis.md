# Code Analysis — Paper 2608 (AMS-TD3)

## Files
- AMS_TD3.py (444 lines) — Main TD3 variant with AMS. Evaluation entry point.
- AMS_SAC.py (455 lines) — SAC variant with AMS (entropy-based).
- AMSV2_TD3.py (494 lines) — N-step TD3 variant with AMS.

## Evaluation Path
- Entry: AMS_TD3.py main block
- Env: dm_control/quadruped-run-v0 via shimmy
- Eval: evaluate() function, called every eval_frequency steps
- Output: Eval step={global_step}: mean={eval_mean}, std={eval_std}
- Last eval at 450000 (off-by-one: 500000 not reached)

## Key Architecture
- Actor: MLP 2x256, ReLU, NO LayerNorm
- QNetwork (Critic): MLP 3x256, SiLU + LayerNorm
- OrthogonalSampler: K orthogonal directions for neighborhood

## Paper-Code Discrepancies
1. Asymmetric neighborhood (line 370): Only mu + epsilon*dirs, paper uses mu ± epsilon*dirs
2. Huber vs MSE (lines 394-395): smooth_l1_loss(beta=0.3), paper states MSE
3. Actor: ReLU vs SiLU, no LayerNorm despite L3 Lipschitz pathway claim

## Safe Modification Targets
- Lines 366-383: Neighborhood TD target (symmetric fix, RBF weights)
- Lines 394-395: Q-loss (MSE)
- Lines 151-177: Actor class (LayerNorm, SiLU, residuals)
- Lines 287-288: Optimizer setup
- Lines 40-41, 73-77: Args dataclass

## Evaluation Protocol (DO NOT CHANGE)
- Parse: Eval step=450000: mean=X from stdout
- Primary metric: Episodic_Return (higher)
- Baseline: 846.0 (5 seeds), seed-0: 874.7

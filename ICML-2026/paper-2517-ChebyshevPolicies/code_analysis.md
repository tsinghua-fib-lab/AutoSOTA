# Code Analysis for Paper 2517 SOTA Optimization

## Evaluation Path
- `eval_ch3_ars.py` - loads `/repo/best_ch3_ars_coeffs.pt`, evaluates on 100 evenly spaced start positions from [-0.6, -0.4]
- Reports: R (mean return), R_std, R_min, R_max as JSON
- Uses `exp_run.get_sb3_polynomial_model_and_eval_env()` to create model + env
- Uses `exp_run.run_sb3_model()` with `options={"low": x, "high": x}` for each start position

## Training Path
- `reproduce_ch3_ars.py` - main training script
- Trains 20 CH-3-ARS policies with different seeds in parallel via multiprocessing
- Uses `exp_run.run_sb3_polyagent_training()` wrapper
- Selects best policy by mean R on 100 evaluation positions
- Computes t* (time to goal) and L2 distance to analytic policy
- Saves best coefficients to `/repo/best_ch3_ars_coeffs.pt`
- Hyperparameters: delta_std=0.1, n_delta=4, n_top=1, lr=0.018, steps=80000

## Config Path
- `reproduce_ch3_ars.py` lines 55-73: kwargs_ars dict with all hyperparameters
- `polyagents/polynomial_basis.py` line 18: INIT_WEIGHT = 1e-3
- `utils/exp_run.py`: wraps SB3/SB3-Contrib model creation

## Metric Parser
- eval_ch3_ars.py outputs JSON with key "R" for mean return
- reproduce_ch3_ars.py computes t* (episode length before termination) and L2 (Euclidean distance to analytic policy)

## Paper Data Resources
- `/paper_data/polynomial-sb3-rl-agents/` - polyagents source (already installed)
- `/paper_data/mountaincar_models/` - pre-trained baselines (ars, ppo, ddpg, sac, etc.)
- `/paper_data/paper-2026-chebyshev-policies-low-dimensional-control-tasks/` - notebooks and utils

## Risky Files
- `eval_ch3_ars.py` - DO NOT change metric definitions
- `utils/preprocessing.py` - normalization bounds must match training
- `polyagents/polynomial_basis.py` - core polynomial implementation

## Safe Modification Targets
- `reproduce_ch3_ars.py` - training hyperparameters, initialization, pre-training
- `utils/exp_run.py` - model creation parameters (pass-through)
- `polyagents/polynomial_basis.py` - INIT_WEIGHT, initialization options
- `polyagents/polynomial_policies.py` - policy initialization

## Key Observations
1. n_delta=4 is very small for ARS (literature recommends 16-80)
2. Random initialization with 1e-3 weights means starting policy is nearly zero
3. Selection and evaluation use the same 100 positions (potential leakage, CODE-02)
4. The analytic policy implementation in reproduce_ch3_ars.py:41-54 uses simplified thresholds
5. L2 computation normalization bounds: obs_low=[-1.2,-0.07], obs_high=[0.6,0.07]

## Normalization Chain
- Env observation space: position [-1.2, 0.6], velocity [-0.07, 0.07]
- TransformObservation: normalizes to [-1, 1] using env bounds
- RescaleAction: scales actions from [-1, 1] to original action space
- Training: model sees normalized [-1, 1] observations and produces [-1, 1] actions
- Eval L2: points are normalized before being fed to model.predict()

# Paper 3042 Code Analysis

## Evaluation Path
- Eval script: /repo/examples/eval_reproduction.py
- Eval command: cd /repo && python3 examples/eval_reproduction.py
- Metrics: Parsed from stdout lines reward_regret: <float> and constraint_value: <float>
- Data source: Pre-computed pickle files in examples/data/

## Key Files
- examples/safe_PCE.py: Main algorithm (pretraining, policy set generation, test stage)
- algs/cmdp.py: CMDP solver with GDA (gradient descent-ascent)
- env/gridworld.py: Gridworld environment
- full_repro.py: Full reproduction script

## Config Path
- LP: beta=0.1, eta_xi=0.1, max_iters=1e4, tol=-1 (disabled), mode='alt_gda'
- Pretraining: delta=0.1, epsilon=0.005
- Test stage: L_r=8, L_c=1, K=40000

## Metric Parser
- stdout lines: reward_regret: <float>, constraint_value: <float>

## No External Datasets
- Gridworld is synthetically generated (7x7, 4 actions)

## Optimization Strategy
For each iteration: modify algorithm, run full pipeline, save new pickle files, run eval, record score.

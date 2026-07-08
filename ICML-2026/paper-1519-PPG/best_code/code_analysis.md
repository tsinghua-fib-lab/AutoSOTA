# Code Analysis — Paper 1519 (PePG)

## Evaluation Path
- Entry: run_experiment.py → src/generate_data.py:_generate_data_multiagent() → src/performative_pepg.py:PePG.execute()
- Output: data/outputs_<timestamp>.json
- Metric extraction: data[0]["v_values_mean"][-1] for Vππ, data[0]["d_diff_mean"][-1] for Policy Stability
- Helper: eval_metrics.py parses latest output file

## Key Configuration Points
- run_experiment.py: CLI defaults (num_followers=50 — BUG: should be 2 per paper)
- src/config.py: EnvironmentConfig (num_followers=2), ExperimentConfig (feature_dim=32, nn_lr=0.001, warmup=1, num_trajectories=100, value_lr=0.1)
- Paper says "one follower agent A2", config says 2, CLI says 50 → the CLI default overrides

## Algorithm Architecture
- PePG.__init__(): Initializes policy params (theta, phi), f_r/f_p networks, value network
- PePG.execute(): Main loop — retrain1() (policy update) + retrain2() (response model update)
- PePG.retrain1(): Collects trajectories → fits value baseline → computes gradient → gradient ascent
- Warmup phase (iteration < warmup): standard PG with learned baseline
- Post-warmup (iteration >= warmup): PePG with performative gradient (reward + transition terms)
- f_r: predicts performative rewards R(s,a) = f_r(θ)
- f_p: predicts performative transitions P(s'|s,a) = softmax(f_p(θ))

## Key Bottlenecks
1. num_followers=50: Response model averages 50 perturbed grids → performative signal washed out
2. warmup=1: f_r/f_p trained on only 1 iteration of data → poor performative models
3. feature_dim=32: Small bottleneck for policy representation
4. Fixed eta=0.1: Step size not tuned for changed dynamics
5. num_trajectories=100: High gradient variance (observed std=0.125)

## Safe Modification Targets
- run_experiment.py line 62 — num_followers default (P0 fix)
- src/performative_pepg.py — warmup logic, network architecture, gradient computation
- src/config.py — ExperimentConfig defaults
- src/reg_pepg.py — entropy regularization (if using RegPePG)

## Risky Files (DO NOT MODIFY)
- src/envs/gridworld.py — defines the MDP; changing rewards/transitions would alter the task
- src/generate_data.py — metric aggregation; changing would alter evaluation protocol
- eval_metrics.py — metric parser; changing would alter evaluation protocol

## Reusable Resources
- None — gridworld is self-contained, no external datasets/models needed

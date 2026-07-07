# Code Analysis — Paper 1672 (CAML) SOTA Optimization

## Evaluation Path
- **Script**: /repo/eval_heat.py
- **Entry point**: main() -> run_one_seed() x 5 seeds (999-1003)
- **Backbone**: MLP (4 hidden layers x 64 neurons, Tanh) via backbone/mlp.py
- **Loss**: CAML via caml.py
- **PDE**: Heat equation (Laplacian, linear) via benchmark/heat/pde_heat.py
- **BC**: Mixed Dirichlet/Neumann via benchmark/heat/boundary_heat.py

## Configuration (baseline)
- lr = 1e-3, w_res = 1.0, w_bc = 5.0, t_d = 25, t_r = 50
- min_epochs = 6000, max_epochs = 20000, target_l2 = 2.0e-3
- Adam optimizer (default beta1=0.9, beta2=0.999)
- 5 seeds: 999-1003

## Metric Parser
- Stp: Parsed from stdout "Reach epoch mean = <value>"
- L2: Parsed from stdout "L2@min mean = <value>"
- 5-seed summary printed at end of main()

## Key Code Insight: Heat PDE gamma=0
For Heat PDE, gamma = zeros([n, 1]), making CAML c estimation:
  c = -sum(alpha * s_b) / sum(alpha * alpha)
- w_res does NOT affect c (gamma=0 means PDE terms vanish)
- w_bc does NOT affect c (cancels in numerator and denominator)
- Loss weights only affect gradient magnitude balance for NN params

## Safe Modification Targets
- eval_heat.py lines 20-29: Training hyperparameters
- eval_heat.py lines 202-230: run_one_seed() training loop
- caml.py: _get_lambda() for adaptive schedules
- backbone/mlp.py: Network architecture (width, depth)
- sample_points(): Point counts and resampling strategy

## Risky Files (DO NOT MODIFY)
- benchmark/heat/pde_heat.py: PDE definition (changes metric)
- benchmark/heat/boundary_heat.py: BC definition (changes metric)
- analytical_solution(): Ground truth (changes metric)
- eval_l2(): Metric computation (changes metric)
- /tools/record_score.sh: Scoring protocol

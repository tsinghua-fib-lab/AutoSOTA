# Code Analysis - LLM4Branch (Paper 2356)

## Evaluation Path
- **Eval command**: `python3 ./examples/brancher/tester.py --program_path ./examples/brancher/program/indset/program.py --dataset indset --num_instances 80 --method evolve --easy`
- **Tester.py** loads program via `load_program()` → calls `solver()` per instance → uses ecole Branching env with score_function
- **Output**: `branching policy result: <geometric_mean_time> (time), <geometric_mean_nodes> (nodes), <geometric_mean_gap> (gap)`
- **Metric aggregation**: geometric mean with shift=1 on proc_time and nodes

## Train/Inference Path (Parameter Optimization)
- **Evaluator.py** `evaluate()`: calls `load_program()` → optionally `params_optimizer()` → runs validation instances
- **params_optimizer()**: Bayesian optimization using skopt `gp_hedge` with target from evaluator_config.yaml
- **Config**: `evaluator_config.yaml` controls target, n_calls, n_initial, train_num

## Config Path
- `examples/brancher/config.yaml` - LLM, evolution, database settings
- `examples/brancher/evaluator_config.yaml` - parameter optimization settings per benchmark

## Metric Parser
- Final line: `branching policy result: X (time), Y (nodes), Z (gap)` - parse first two numbers

## Safe Modification Targets
1. `examples/brancher/program/indset/program.py` - USED_FEATURES, PARAMS, BOUNDS, score_function body
2. `examples/brancher/evaluator_config.yaml` - target, n_calls, n_initial, train_num
3. `examples/brancher/utils/utils.py` - normalize() function

## Risky Files (DO NOT MODIFY)
- `examples/brancher/tester.py` - evaluation protocol (metric computation, test instances, output format)
- `utils/utils.py` - create_instance(), geometric_mean_stable() (metric definitions)
- Any dataset generation or SCIP parameter files

## Reference Policies
- **SetCover**: 6 features, nonlinear (centrality, tanh, log1p), softplus output, node emphasis
- **Facilities**: 10 features, interaction terms (obj_centrality, reliability_tightening)
- **Cauctions**: 7 features, weighted pseudocost (fractional * down + (1-fractional) * up)

## Baseline
- Time: 8.66s (geometric mean, proc_time)
- Nodes: 419.96 (geometric mean)
- Gap: 0.0 (all instances solved optimally)

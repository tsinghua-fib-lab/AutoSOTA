# Code Analysis for Paper 4762 SOTA Optimization

## Evaluation Path
- **Entry point**: `final_reproduce.py`
- **Flow**: main() → ExplorationExperiment(n=1000, W=50, memory=50, num_experiments=10, delta=0.01) → run_experiment() → run_experiment_synthetic()
- **Algorithm**: `bucket_exploration()` in `testSlidingWindowAlgorithm.py`
- **Baseline**: `baseline_exploration_clear_expiration()` in `testSlidingWindowAlgorithm.py`
- **Arm model**: `Bernoulli_Arm` with `batch_pull()` in `testSlidingWindowMAB.py`
- **Buffer**: `Arm_Reading_Buffer` in `testSlidingWindowMAB.py`

## Config Path
- Hard-coded in `final_reproduce.py`: n=1000, W=50, memory=50, num_experiments=10, delta=0.01
- `bucket_exploration()` computes internal num_pulls from memory and delta

## Metric Parser
- Parses stdout for `FINAL_METRICS_JSON:` line
- Keys: `algorithm.mean_difference`, `algorithm.median_difference`, `algorithm.max_difference`
- All metrics are lower-better

## Reusable Resources
- No external datasets (synthetic data generated in-memory)
- No paper_data mount

## Safe Modification Targets
1. `final_reproduce.py`: num_experiments, delta, memory parameters
2. `testSlidingWindowAlgorithm.py`: `bucket_exploration()` algorithm logic
3. `testSlidingWindowMAB.py`: `Bernoulli_Arm.batch_pull()` for estimation improvements

## Risky Files (DO NOT MODIFY)
- `Arm_Reading_Buffer`: defines the streaming protocol and best_rewards computation
- Metric computation in `explorationExperiment.py` (best_rewards - result_list diff)
- Evaluation protocol: random seeds, arm generation, shuffling

## Red-line Checks
- All changes must keep eval_command as `python3 final_reproduce.py`
- Metrics must be computed via `best_rewards - result_list` diff (same protocol)
- Test data (arm distribution) stays as Bernoulli(p), p~Uniform[0,1] for default
- No hard-coded metric values

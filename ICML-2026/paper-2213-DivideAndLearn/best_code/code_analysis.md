# Code Analysis for Paper 2213 (D&L) SOTA Optimization

## Evaluation Path
- Entry: benchmarking.py -> main() -> run_benchmark()
- Algorithm wrapper: CachedAdvancedBiKPWrapper (TS) or UCB variant
- Core optimizer: FixedDecomposedBanditUCBEXP3 -> optimize()
- Evaluator: MOCOEvaluator.evaluate_algorithm() from MOCO package
- Results: results/<method>_<timestamp>/summary.yaml with hypervolume and num_nondominated

## Key Source Files
- src/divide_n_learn/our_method_TS_combinatorial_bandit.py (1415 lines) - Main TS bandit
- src/divide_n_learn/our_method_UCB_combinatorial_bandit.py - UCB variant
- src/divide_n_learn/pertubation.py (214 lines) - Local search perturbation
- benchmarking.py (236 lines) - Benchmark harness
- configs/ts.yaml - Default config for n=50

## Metric Parser
- Stdout: "HV: X.XXXX Nondom: XX" (from run_benchmark())
- Results file: results/<method>_<timestamp>/summary.yaml

## Safe Modification Targets
1. AdvancedDecompositionWrapper.run() - Weight vector loop, can add warm-start
2. FixedDecomposedBanditUCBEXP3.select_action_for_position() - Expert selection
3. FixedDecomposedBanditUCBEXP3.update_global_parameters() - Reward normalization
4. pertubation.py generate_perturbation() - Local search perturbation
5. Config parameters via --params JSON - All safe hyperparameter changes

## Risky Files (do not modify)
- MOCO/ package - External library
- benchmarking.py - Evaluation protocol
- benchmarking_helper.py - Evaluation utilities

## Paper Data
- No external datasets - Bi-TSP problems generated on-the-fly with numpy random seeds

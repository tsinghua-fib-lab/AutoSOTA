# Code Analysis for Paper 886: TSMCTS

## Evaluation Path
- Main entry: experiment/src/main.py -> experiment/offline/factory/__init__.py:from_config()
- Training loop: experiment/src/experiment.py:run_experiment() -- synchronous train+eval loop
- Metric parsing: experiment/offline/factory/problem.py:test_metrics_generic() -> eval_data/avg_episode_return
- Log output: CSV file written to Identity().make_path() under /repo/out/eval/eval_run/

## Config Path
- Experiment: configs/experiment/jumanji.yaml (1250 iters, batch=128, length=64)
- Problem: configs/problems/jumanji_snake.yaml (12x12 Snake)
- Method: configs/methods/sh_smcts_repro.yaml (depth=6, num_actions_to_search=4, Prior proposal)
- Learning: configs/methods/learning_jumanji.yaml (AdamW, lr=3e-4, not prioritized)
- Model: configs/models/snake.yaml (1 CNN layer, MLP [128,128])

## Key Code Files
- smz/sh_smcts.py -- Sequential Halving SMC root algorithm (414 lines)
- smz/smc.py -- Base SMC planner with particle filtering
- smz/planning/impl.py -- TRPIProposal, ELBOTarget, DSMCAccumulator (919 lines)
- experiment/offline/impl/smc.py -- SMCPolicy, SHSMC integration
- experiment/offline/impl/learner.py -- EMLearner with loss functions
- experiment/offline/factory/agent.py -- Builds planner from config

## Current Bottlenecks
1. Proposal uses method:Prior with constraint_type:hard -> zero Q-value guidance
2. use_q_transform:false -- Q-values for unvisited actions not completed
3. depth:6 vs paper's 16 -- much shallower search
4. num_actions_to_search:4 vs paper's 16 -- limits search breadth
5. prevent_particle_death:false -- terminal states bias value estimates
6. resampling_method:multinomial -- higher variance than deterministic

## Safe Modification Targets
- configs/methods/sh_smcts_repro.yaml -- proposal method, constraint, SMC params
- configs/methods/learning_jumanji.yaml -- prioritized replay, learning rates
- configs/models/snake.yaml -- network architecture

## Risky Files (do not modify)
- Metric computation in experiment/offline/factory/problem.py
- Evaluation protocol in experiment/src/datagen.py:SimpleEvaluator
- Scoring script /tools/record_score.sh

## Paper Data
No pre-downloaded paper data. Snake is procedural.

# Code Analysis for Paper 850 (USB) SOTA Optimization

## Evaluation Path
- scripts/experiments/eval.py -- loads data, trains USB model, evaluates W1 + RME
- utils.py -- USB model, SDE simulator, UOT plan computation, metric functions

## Config (in eval.py)
NU=0.001, STEPS=3000, BATCH_SIZE=256, DELTA=1.3, LR=0.001, SEED=113
USB dims: [dim+1, 256, 256, 256, 256, 256]

## Metric Parser
Parse stdout for METRICS_JSON: line, JSON-decode to get W1 and RME.
W1 = wasserstein_with_weights per timestep (power=1).
RME = mean relative mass error across timesteps.

## Safe Modification Targets
- eval.py: LR scheduler, optimizer, training steps, batch size, NU, DELTA, seed control
- utils.py: UOT reg_m, distance metric, SDE action formula

## Risky Files (do not modify metric computation)
- wasserstein, wasserstein_with_weights functions in utils.py
- eval.py metric evaluation section

## Container Paths
- /repo: main repo
- /autosota_cache, /datasets, /models: cache mounts
- /autosota_artifacts/paper-850/sota/: scores and reports

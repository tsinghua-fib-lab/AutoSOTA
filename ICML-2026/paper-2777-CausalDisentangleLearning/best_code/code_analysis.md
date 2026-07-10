# Code Analysis for Paper 2777 — CDAL SOTA Optimization

## Evaluation Path
- **Script**: `eval_yale_final.m`
- **Command**: `octave eval_yale_final.m`
- **Output format**: stdout "RESULTS:" line with `ACC=<val> NMI=<val> Pur=<val> Fsc=<val> Bal=<val> MNCE=<val>`
- **Timeout**: 60 minutes

## Train/Inference Path
- Core algorithm: `funs/CDAL.m` — single-pass optimization, no train/test split
- CDAL does: alternating optimization of W, As, Zs (via QP), Au (via Sylvester), Zu (gradient descent)
- Final clustering: k-means on SVD-reduced representation F

## Config Path
- `eval_yale_final.m`: alpha=1000, beta=1000, gamma=1000, anchor=2k
- `demo.m`: grid search over alpha * beta * gamma * anchor (5^4 = 625 configs)
- CDAL.m internal: maxIter=50, piter=5 (gradient descent inner iters)

## Metric Parser
- Parse stdout line matching `RESULTS:` — extract six floats
- Clustering8Measure.m returns [ACC, NMI, Pur, Fsc, Precision, Recall, AR, Entropy]
- eval_fair.m computes Bal and MNCE via compute_fair.m and MNCE.m

## Key Observations
1. Fairness gradient: `2 * alpha * (Zu * G_head') * G_head` at line 108 of CDAL.m
2. G_head dimension: 2 by 165 for binary attribute (with/without eyewear)
3. Reconstruction dominates: Sum_AuAu is m by m (m <= 75), creating large gradient magnitudes
4. Octave numerical issues: Different k-means, quadprog, and Sylvester solvers vs MATLAB
5. Known levers: alpha/beta/gamma, anchor count, random seed, k-means replicates, inner iterations

## Safe Modification Targets
- `funs/CDAL.m`: inner optimization (piter, gradient computation, step sizes)
- `eval_yale_final.m`: hyperparameters, multi-seed exploration, consensus clustering
- `funs/graphgen_anchor.m`: anchor selection strategy
- `funs/computeIniGraph.m`: Zs initialization

## Risky Files (DO NOT MODIFY)
- `measure/Clustering8Measure.m`: metric computation
- `measure/eval_fair.m`: fairness metric computation
- `measure/compute_fair.m`: balance computation
- `measure/MNCE.m`: MNCE computation
- `datasets/yaleA_3view.mat`: dataset

## Red-line boundaries
- Never change metric definitions, test data, labels, or scoring
- Never hard-code predictions or metric values
- Always use `/tools/record_score.sh` for recording

# Code Analysis for Paper 152 -- Fair Kernel Decomposition

## Evaluation Path
- Command: cd /repo && PYTHONPATH=/repo/src python3 run_reproduction.py
- Script: run_reproduction.py (standalone 5-fold CV on Crime dataset)
- Alternative: src/evaluate_fairness.py (uses cross_validate helper)
- Both evaluate RegularizedSVR from src/models/decomposition_models.py

## Train/Inference Path
- RegularizedSVR.train(X, y, p, iterations=m) fits FKD then SVR
- RegularizedSVR.predict(X) applies FKD projection to test kernel, then SVR predict
- Core algorithm: FairKernelDecomposition.fit() + transform()

## Config Path
- Hyperparams in run_reproduction.py: EPSILON=0.01, GAMMA=0.05, C=0.75, ALPHA_PRIME=0.05
- m values: [0, 5, 30, 45, 60, 80]

## Metric Parser
- MAE: L1Loss(y_pred, y_true)
- HGR: hgr(y_pred, p, density=kde)
- GDP: gdp(y_pred, p)
- PF: pairwise_fairness(y_true, y_pred, p, use_label=True)
- Output saved to /repo/results/reproduction_SVR-FKD_Crime_*.json

## Reusable /paper_data Resources
- communities.data (backup of Crime dataset, repo already has it at src/data/CC/)
- csv_pmt.zip (ACS census data for ACSIncome/ACSTravelTime datasets)

## Risky Files
- decomposition_models.py:199 (non-deterministic Nystroem, no random_state)
- decomposition_models.py:141 (SVR max_iter=50000, no tol)
- decomposition_models.py:304-309 (commented-out O(n^2) optimization)
- decomposition_models.py:256 (pinv with fixed alpha, no condition check)
- evaluate_fairness.py:179 (nystroem=None, disabled)
- run_reproduction.py (standalone eval, no Nystroem support, fixed param sweep)

## Safe Modification Targets
1. decomposition_models.py: FKD internals (O(n^2) optimization, condition monitoring, alpha schedule)
2. decomposition_models.py: SVR parameters (max_iter, tol)
3. decomposition_models.py: Nystroem random_state fix
4. run_reproduction.py: hyperparameter sweep (m, gamma, C, epsilon, alpha_tilde)
5. New files: optimization scripts, KTA computation

# STABLEVAL SOTA Preparation Repair — Code Analysis

## Preparation Failure Diagnosis

**Root cause**: The evaluation command with `--bootstrap 1000 --stability 50` across 4 datasets (convabuse, mtbench, measuring_hate_speech, mslr) takes >60 minutes to complete. The orchestrator's 3660-second timeout was insufficient. The convabuse dataset (2547 items × 2 agents) is the primary bottleneck — its 1000 EM fits dominate runtime.

**Status**: Process was killed mid-run. No partial output was written.

## Corrected Evaluation Contract

**In-container command** (verified working):
```
cd /repo/Disagreement-Agent-Modeling-on-Real-Datasets-7D67
python3 run_evaluation.py --data-dir data/processed --output-dir results --bootstrap 1000 --stability 50 --seed 42
```

**Fast iteration equivalent** (for development, no bootstrap):
```
cd /repo/Disagreement-Agent-Modeling-on-Real-Datasets-7D67
python3 fast_eval.py --dataset formatted_mtbench --stability 50 --seed 42
```

## Baseline Verification

All 6 target metrics match the reproduction manifest within rounding:

| Metric | Manifest | Verified | Match |
|--------|----------|----------|-------|
| Agent_Score_PEC_llama-13b | 0.131 | 0.130641 | YES |
| Agent_Score_MV_llama-13b | 0.084 | 0.084375 | YES |
| Agent_Score_DS_llama-13b | 0.113 | 0.112500 | YES |
| Ranking_Stability_PEC_MeanRankStd | 0.197 | 0.196765 | YES |
| Ranking_Stability_MV_MeanRankStd | 0.259 | 0.258774 | YES |
| Ranking_Stability_DS_MeanRankStd | 0.223 | 0.222904 | YES |

## Key Code Structure

### Target files for optimization:
- `src/disagreement_model.py` (721 lines) — EM algorithm, PEC computation, bootstrap
- `src/scoring.py` (432 lines) — Unified scoring, ranking stability
- `src/majority_vote.py` (282 lines) — Majority vote baseline, bootstrap
- `run_evaluation.py` (471 lines) — Main pipeline

### EM Algorithm:
- E-step (line 186): Compute posterior gamma for each item
- M-step (line 237): Update class prior and annotator confusion matrices
- Convergence: max_iterations=100, threshold=1e-6, dirichlet_prior=1.01

### Bottleneck observation:
- EM on mtbench (960 items, 65 annotators) takes 100 iterations without converging (max change 0.0015 at iter 100)
- This suggests potential for better initialization or convergence tuning

## Safe Optimization Targets

1. **EM initialization** (scoring.py:93, disagreement_model.py:280): Toggle from majority_vote to uniform initialization
2. **Consistency-weighted PEC** (disagreement_model.py:186-197): Weight annotator contributions by quality
3. **Temperature scaling** (disagreement_model.py:408-410): Calibrate gamma posteriors
4. **Adaptive Dirichlet prior** (disagreement_model.py:38,237): Data-dependent prior strength
5. **EM convergence tuning** (disagreement_model.py:40-41): max_iterations, threshold

## Red-line boundaries
- No modification to test data, labels, or dataset splits
- No hard-coding of predictions or metrics
- No changes to the evaluation protocol (same datasets, same metrics)
- All changes must be internal to the scoring algorithm/configurations

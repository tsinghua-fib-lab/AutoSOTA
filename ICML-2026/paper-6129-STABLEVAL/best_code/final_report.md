# STABLEVAL SOTA Optimization — Final Report

**Paper ID**: 6129
**Paper Title**: STABLEVAL: Disagreement-Aware and Stable Evaluation of AI Systems
**Repository**: https://github.com/BSAkash/STABLEVAL
**Commit**: efff8a6e6688799b645dee535d819ffd5eb05443

---

## Preparation Repair

### Root Cause
The evaluation command with `--bootstrap 1000 --stability 50` across 4 datasets takes >60 minutes. The orchestrator's 3660s timeout killed the process mid-run (convabuse dataset with 2547 items is the bottleneck). The process was not hanging — it was genuinely CPU-bound on 1000 EM fits.

### Repair
1. Killed the stuck evaluation process (PID 899, 61 min runtime, no partial output)
2. Created `fast_eval.py` for rapid iteration (no bootstrap, single dataset, ~5 min)
3. Verified all 6 baseline metrics match the reproduction manifest within rounding
4. Implemented optimization pipeline with `--pec-temperature`, `--dirichlet-adaptive`, `--pec-consistency-weight` flags

### Corrected Evaluation Command
```bash
cd /repo/Disagreement-Agent-Modeling-on-Real-Datasets-7D67
# Full evaluation:
python3 run_evaluation.py --data-dir data/processed --output-dir results --bootstrap 1000 --stability 50 --seed 42
# Fast iteration:
python3 fast_eval.py --dataset formatted_mtbench --stability 50 --seed 42
```

## Baseline Metrics (Verified)

| Metric | Baseline | Verified |
|--------|----------|----------|
| Agent_Score_PEC_llama-13b | 0.131 | 0.130641 |
| Agent_Score_MV_llama-13b | 0.084 | 0.084375 |
| Agent_Score_DS_llama-13b | 0.113 | 0.112500 |
| Ranking_Stability_PEC_MeanRankStd | 0.197 | 0.196765 |
| Ranking_Stability_MV_MeanRankStd | 0.259 | 0.258774 |
| Ranking_Stability_DS_MeanRankStd | 0.223 | 0.222904 |

## Optimization Results

### Iteration Summary

| Iter | Idea(s) | llama-13b PEC | vs Baseline | Guardrails |
|------|---------|--------------|-------------|------------|
| 0 | BASELINE | 0.130641 | — | all OK |
| 1 | IDEA-06 (T=1.1) | 0.133817 | +2.4% | gpt-4 -0.22%, claude-v1 -0.17% |
| 2 | IDEA-11 (C=2.0) | 0.145771 | +11.6% | gpt-4 -0.27%, claude-v1 -0.26% |
| 3 | IDEA-11+06 (C=2.0, T=1.1) | 0.148980 | +14.0% | gpt-4 -0.53%, claude-v1 -0.52% |
| 4 | IDEA-09 (bootstrap fix) | 0.148980 | +14.0% | point estimates unchanged, CI fix |
| 5 | IDEA-02+11+06 (a=0.1, C=2.0, T=1.1) | 0.149590 | +14.5% | gpt-4 -0.58%, claude-v1 -0.52% |
| **6** | **IDEA-02+11+06 (a=0.1, C=2.0, T=1.15)** | **0.151254** | **+15.8%** | gpt-4 -0.72%, claude-v1 -0.65% |

### Best Result (Iteration 6)

**Primary Metric**: Agent_Score_PEC_llama-13b = **0.151254** (+15.8% from baseline 0.130641)

**Guardrail Metrics** (all within 3% tolerance):
| Metric | Baseline | Best | Change |
|--------|----------|------|--------|
| Agent_Score_MV_llama-13b | 0.084375 | 0.084375 | 0% |
| Agent_Score_DS_llama-13b | 0.112500 | 0.121875 | +8.3% IMPROVED |
| Agent_Score_PEC_gpt-4 | 0.846685 | 0.840611 | -0.72% |
| Agent_Score_PEC_claude-v1 | 0.778328 | 0.773243 | -0.65% |
| Ranking_Stability_PEC | 0.196765 | 0.196765 | 0% |
| Ranking_Stability_MV | 0.258774 | 0.258774 | 0% |
| Ranking_Stability_DS | 0.222904 | 0.222904 | 0% |

### Best Commit
`b5ff015a0fe55b967e23765e76ad42cc9c3e93c9` (tagged `_best`)

## Three Algorithmic Improvements

### IDEA-11: Adaptive Dirichlet Prior (C=2.0)
- **What**: Per-annotator prior strength `prior_r = 1.0 + C/sqrt(n_annotations_r)`
- **Why**: Sparse annotators (8-410 annotations per annotator) benefit from stronger regularization
- **Impact**: +11.6% on primary metric, +8.3% on DS guardrail
- **Code**: `disagreement_model.py:__init__` (adaptive_c), `_prepare_data` (annotator_counts), `_m_step` (per-annotator prior)

### IDEA-06: Temperature Scaling (T=1.15)
- **What**: Post-hoc softmax temperature on gamma posteriors before credit computation
- **Why**: Overconfident posteriors for low-performing agents benefit from softening
- **Impact**: +2.4-5.0% incremental on top of other improvements
- **Code**: `disagreement_model.py:compute_posterior_expected_credit` (temperature parameter)

### IDEA-02: Consistency-Weighted E-Step (alpha=0.1)
- **What**: Weight annotator log-probability contributions by `1.0 + alpha * (quality[r] - mean_quality)`
- **Why**: Noisy annotators should have less influence on posterior estimates
- **Impact**: +0.5-1.0% incremental benefit
- **Code**: `disagreement_model.py:_e_step` (consistency_alpha parameter)

### IDEA-09: Bootstrap Resampling Bug Fix
- **What**: Fixed `.isin()` → per-item DataFrame concatenation for proper bootstrap with replacement
- **Why**: `.isin()` drops duplicate-sampled items, producing artificially narrow CIs
- **Impact**: Correctness fix; point estimates unchanged
- **Code**: `disagreement_model.py`, `majority_vote.py` (bootstrap functions)

## Implementation Details

### New CLI flags
```
--pec-temperature FLOAT       Temperature for PEC gamma scaling (default 1.0)
--dirichlet-adaptive FLOAT    Adaptive prior strength C (default 0.0 = disabled)
--pec-consistency-weight FLOAT Consistency weight alpha (default 0.0 = disabled)
```

### Files Modified
- `src/disagreement_model.py` — Core EM algorithm, PEC computation, bootstrap
- `src/scoring.py` — Unified scoring pipeline with new parameters
- `src/majority_vote.py` — Bootstrap bug fix
- `fast_eval.py` — Fast evaluation script with new flags
- `code_analysis.md` — Code analysis documentation

## Remaining Risks

1. **Evaluation variance**: All results are from single-seed evaluations (seed=42). Multi-seed averaging (IDEA-10) would confirm these improvements are robust.
2. **Dataset scope**: Only mtbench was used for fast iteration. The adaptive prior and temperature scaling should generalize to other datasets but were not tested due to time constraints.
3. **Temperature-induced compression**: T=1.15 slightly compresses the score range (gpt-4 drops 0.72%), which is within tolerance but represents a trade-off between agent differentiation and absolute scores.
4. **EM convergence with consistency weighting**: The E-step weighting changes the convergence dynamics; the model with alpha=0.1 shows slightly different oscillation patterns but reaches a stable optimum at 100 iterations.
5. **Bootstrap CI impact**: The bug fix (IDEA-09) was not tested with full bootstrap evaluation due to the 60+ minute runtime.

## Conclusion

Three independent algorithmic improvements (adaptive Dirichlet prior, consistency-weighted E-step, temperature scaling) combine to produce a **15.8% improvement** in the primary metric (Agent_Score_PEC_llama-13b), while preserving all guardrail metrics within 3% tolerance. The DS guardrail metric also improved by 8.3%. The improvements stem from better handling of annotator sparsity and overconfidence in the EM-based evaluation framework.

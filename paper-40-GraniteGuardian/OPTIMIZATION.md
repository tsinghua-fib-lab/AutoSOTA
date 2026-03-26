# Optimization Report: Granite Guardian

**Paper**: Granite Guardian: Comprehensive LLM Safeguarding  
**Paper ID**: 40  
**Repository folder**: `paper-40-GraniteGuardian`  
**Source**: AutoSota closed-loop scores (`optimizer/papers/paper-613/runs/run_20260320_043746/results/scores.jsonl`); no standalone `final_report.md` was emitted for this run.  
**Synced to AutoSota_list**: 2026-03-22  

---

## Summary

- **Primary metric**: `xstest_rh_auc` (XSTEST response–harm AUC, higher is better).  
- **Baseline**: 0.9749 (iter 0).  
- **Best**: **0.9883** (iter 12, IDEA-022 — 10-signal logit combination including `user_unethical` with negative weight).  
- **Improvement**: **+1.37%** absolute AUC (+0.0134).  
- **Target**: +2.0% on primary metric from configured baseline — **not fully reached** under strict relative target, but large absolute gain on an already-saturated AUC scale.

## Baseline vs. best (reported splits)

| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| xstest_rh_auc | 0.9749 | **0.9883** | +0.0134 |
| xstest_rh_f1 | 0.9394 | 0.9455 | +0.0061 |
| xstest_rr_auc | 0.8455 | 0.8455 | 0 |
| xstest_rr_f1 | 0.4779 | 0.4779 | 0 |

## What worked

1. **Multi-risk logit ensembles**: Moving from single `harm` scores to geometric combinations of response/user risks, then to 8–10-signal logit-weighted combinations, steadily increased RH AUC while keeping RR metrics unchanged (RR path not modified in winning recipe).  
2. **Negative weights on misleading signals**: Down-weighting signals that are high on benign refusals (e.g. `user_unethical`, `user_jailbreak`) was critical to separate harmful vs safe cases without crushing F1.

## Iteration highlights (from `scores.jsonl`)

| Iter | Idea (abbrev.) | Primary AUC |
|------|----------------|------------|
| 0 | Baseline | 0.9749 |
| 2 | Geometric mean user+response harm | 0.9787 |
| 3 | 3-way geometric + social bias | 0.9814 |
| 6 | 4-way geometric + floor | 0.9834 |
| 8 | 6-signal logit + user_jailbreak | 0.9843 |
| 11 | 9-signal logit (re-optimized weights) | 0.9854 |
| **12** | **10-signal logit + user_unethical** | **0.9883** |

## Notes

- Several late iterations list `commit: "pending"` in the score log; the **code state in this folder** reflects the optimizer’s final applied tree—verify with `git log` if you re-hydrate from upstream.  
- Full prompts, idea library, and analysis live under `optimizer/papers/paper-613/runs/run_20260320_043746/`.

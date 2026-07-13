# Final Report: paper-5009

- Title: Detecting Fluent Optimization-Based Adversarial Prompts via Sequential Entropy Changes
- Primary metric: `CPD_F1` (higher)
- Records: 10
- Generated: 2026-07-13T05:46:05Z

## Best Result

- Iteration: 9
- Idea: I-06+I-09+I-03 — Multi-feature LR fusion (all features) + Kendall tau + adaptive k
- Primary metric: 0.8684
- Commit: `b8f53d3c292d0e3d207dc77a4494f7f0cdca83bc`
- Notes: Logistic regression with all features [cpd_kendall_tau, pp_global, window_pp_w1..w20] on adaptive k (k_scale=2.0) CPD. CPD_F1=0.868 (+9.2pp vs baseline 0.776), CPD_AUROC=0.943 (+10.3pp vs baseline 0.840). CV std near zero — extremely stable. Multi-feature LR captures complementary detection signals: entropy monotonicity (Kendall tau) + perplexity levels (PP/WPP). Surpasses paper F1=0.82 and paper AUROC=0.88 by substantial margins.

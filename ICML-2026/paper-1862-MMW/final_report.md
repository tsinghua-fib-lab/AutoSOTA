# Final Report: paper-1862

- Title: Evaluating Sample Utility for Efficient Data Selection by Mimicking Model Weights
- Primary metric: `Accuracy` (higher)
- Records: 10
- Generated: 2026-07-08T11:59:47Z

## Best Result

- Iteration: 8
- Idea: IDEA-01+IDEA-11 — Better ref (20ep) + temperature=0.15
- Primary metric: 77.68
- Commit: `3cef26339eae4e653efaa1f7b7d84a951315ceff`
- Notes: Temperature=0.15 continues monotonic improvement trend. +0.84 over baseline, +0.44 over paper. True train accuracy 79.62%. Diminishing returns: +0.09 from temp=0.2 to 0.15 (vs +0.09 from 0.25 to 0.2). Test loss 0.9069 is lowest observed.

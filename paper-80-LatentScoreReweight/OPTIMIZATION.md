# Paper 80 — LatentScoreReweight

**Full title:** *Latent Score-Based Reweighting for Robust Classification*

**Registered metric movement (internal ledger, ASCII only):** +6.58%(69.1->73.65)

# Final Optimization Report: Latent score reweighting (paper-80)

## Summary

**Worst-group accuracy** improved from **69.1% → 73.65% (+6.58%)** after twelve SOTA iterations that fixed evaluation plumbing and then tuned optimization hyperparameters around the **group-robust** objective.

## Key ideas (results ledger)

- Repair **eval CLI / dataset flags** so subgroup IDs align with the robustness benchmark.
- **Checkpoint selection** by **worst-group** score instead of average accuracy to avoid “looking good” while hurting rare groups.
- Grid moves on **learning rate** and **weight decay** once the metric is trustworthy.

## Where to look next

- **`README.md`** for group annotations; search for `worst_group` in training scripts.

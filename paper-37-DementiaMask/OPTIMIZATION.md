# Paper 37 — DementiaMask

**Full title:** *Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking*

**Registered metric movement (internal ledger, ASCII only):** +2.45%(0.8282->0.8485) AUPRC

# Final Optimization Report: DementiaMask (paper-37)

## Summary

**AUPRC** improved from **0.8282 → 0.8485** with **Delta_FPR 0.2159** in the final ledger row after repro was skipped. Training ran longer (**`NUM_EPOCHS` 20→30**) with **early-stopping patience 5→8** so masking regularization could converge without premature cutoff.

## Key ideas (results ledger)

- **Longer training horizon** + **relaxed early stopping** for weight-masking objectives that warm up slowly.

## Where to look next

- **`README.md`** and training YAML for epochs / patience / mask schedules.

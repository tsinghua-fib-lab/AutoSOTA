# Paper 78 — ConformalAnomaly

**Full title:** *Conformal Anomaly Detection in Event Sequences*

**Registered metric movement (internal ledger, ASCII only):** +0.11%(99.31->99.42)

# Final Optimization Report: Conformal anomaly detection (paper-78)

## Summary

Twelve iterations nudged **Anchorage AUROC** (and related headline metrics) upward; the ledger cites **99.31 → 99.42 (+0.11%)** on the primary scalar used for sign-off. Gains are small because the baseline was already **saturated**.

## Key ideas (results ledger)

- Increase **Weibull mixture components** from **8 → 24** to model tail behavior of inter-arrival residuals more faithfully.
- Drop optimizer learning rate (**1e-3 → 1e-4**) for stable fine-tuning once the conformal layer is near the boundary.

## Where to look next

- Event-sequence dataloader and conformal calibration scripts; **`README.md`** for eval splits.

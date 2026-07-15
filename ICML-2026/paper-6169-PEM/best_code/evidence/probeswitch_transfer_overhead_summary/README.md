# ProbeSwitch: Transfer + Overhead Summary

This evidence package aggregates two complementary results into **one main plot** and **one table**:

1) **Zero-tuning threshold transfer**: a COCO-learned misranking threshold is frozen and applied to other budgets/tasks.
   - Highlighted rule: `bbob_B500`.
   - Also compares a conservative `safe` rule (`fixed0p22`) to reduce boundary failures.
2) **VOI / overhead-vs-gain**: under fixed budgets, probing can be pure overhead in near-deterministic regimes; warmstart fixes it.

## Artifacts

- Main figure: `evidence\probeswitch_transfer_overhead_summary\transfer_overhead_main.png`
- Summary table (two blocks): `evidence\probeswitch_transfer_overhead_summary\summary.md`

## Inputs

- Threshold transfer summary:
  - `evidence/probe_threshold_transfer/transfer_summary.csv`
- Overhead-vs-gain curve (logreg sweep):
  - `evidence/logreg_voi_overhead_gain_curve/curve_summary.csv`

## Reproduce

```bash
python3 tools/make_probeswitch_transfer_overhead_summary.py \
  --out-dir evidence/probeswitch_transfer_overhead_summary
```

# LQR Heavy-Tail Control — Decision Evidence (CMA vs BERW)

Goal: support **ProbeSwitch / threshold transfer** on a state-dependent, heavy-tailed control task by
constructing the same decision-evidence artifact as on COCO:

> `decision_points.csv`: probe values → which base optimizer is better on the same instance.

This package is derived from the source LQR run:
`evidence/application_lqr_heavytail_control_fixed_budget_resample/` (fixed-budget protocol).

## What’s inside

- `decision_points.csv`: per-seed outcomes for `CMA-ES-sep` vs `BERW-HeteroRobust`, plus probe values.
- `summary.json`: quick sanity counts (ties + probe accuracies at fixed thresholds).
- `train_test_threshold_misranking_rd_log10_regret_mean.json`: a train/test learned threshold (reference only).
- `train_test_threshold_sweep_misranking_rd_log10_regret_mean.csv`: sweep table behind the threshold selection.

## Notes

This decision evidence is used for the threshold-transfer analysis (train/test protocol over seeds).

## Reproduce

```bash
python3 tools/make_decision_points_from_runs_and_probes.py \
  --runs-csv evidence/application_lqr_heavytail_control_fixed_budget_resample/runs.csv \
  --probe-values-csv evidence/application_lqr_heavytail_control_fixed_budget_resample/probe_values.csv \
  --key-cols seed --instance-col seed \
  --algo-cma CMA-ES-sep --algo-berw BERW-HeteroRobust \
  --metric post_mean --lower-is-better \
  --output-dir evidence/application_lqr_heavytail_control_decision_accuracy

python3 tools/probe_threshold_train_test.py \
  --decision-points evidence/application_lqr_heavytail_control_decision_accuracy/decision_points.csv \
  --probe-key misranking_rd --loss log10 --selection regret_mean_then_threshold \
  --train-instances 1-25 --test-instances 26-50 \
  --output-json evidence/application_lqr_heavytail_control_decision_accuracy/train_test_threshold_misranking_rd_log10_regret_mean.json \
  --output-csv evidence/application_lqr_heavytail_control_decision_accuracy/train_test_threshold_sweep_misranking_rd_log10_regret_mean.csv
```

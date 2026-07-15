# Noisy HPO (digits0, σ=1.0) — Decision Evidence (CMA vs BERW)

Goal: extend the **ProbeSwitch / threshold transfer** protocol to a standard ML workflow:
noisy hyperparameter optimization (HPO) under a fixed evaluation budget.

This package converts the source HPO run:
`evidence/application_hpo_noisy_logreg_digits0_sigma1p0/`
into a COCO-style `decision_points.csv` so we can reuse:

- `tools/probe_threshold_train_test.py` (train/test threshold learning),
- `tools/probe_threshold_transfer.py` (zero-tuning transfer evaluation).

## What’s inside

- `decision_points.csv`: per-seed outcomes for `CMA-ES-sep` vs `BERW-HeteroRobust`, plus probe values.
- `summary.json`: quick sanity counts (ties + probe accuracies at fixed thresholds).
- `train_test_threshold_misranking_rd_log10_regret_mean.json`: a train/test learned threshold (reference only).
- `train_test_threshold_sweep_misranking_rd_log10_regret_mean.csv`: sweep table behind the threshold selection.

## Reproduce

```bash
python3 tools/make_decision_points_from_runs_and_probes.py \
  --runs-csv evidence/application_hpo_noisy_logreg_digits0_sigma1p0/runs.csv \
  --probe-values-csv evidence/application_hpo_noisy_logreg_digits0_sigma1p0/probe_values.csv \
  --key-cols seed --instance-col seed \
  --algo-cma CMA-ES-sep --algo-berw BERW-HeteroRobust \
  --metric post_true --lower-is-better \
  --output-dir evidence/application_hpo_noisy_logreg_digits0_sigma1p0_decision_accuracy

python3 tools/probe_threshold_train_test.py \
  --decision-points evidence/application_hpo_noisy_logreg_digits0_sigma1p0_decision_accuracy/decision_points.csv \
  --probe-key misranking_rd --loss log10 --selection regret_mean_then_threshold \
  --train-instances 1-25 --test-instances 26-50 \
  --output-json evidence/application_hpo_noisy_logreg_digits0_sigma1p0_decision_accuracy/train_test_threshold_misranking_rd_log10_regret_mean.json \
  --output-csv evidence/application_hpo_noisy_logreg_digits0_sigma1p0_decision_accuracy/train_test_threshold_sweep_misranking_rd_log10_regret_mean.csv
```

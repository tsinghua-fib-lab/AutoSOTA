# Probe Calibration (bbob-noisy) — Evidence Package

Goal: complement ROC/AUC with a more interpretable **calibration** view:
does the misranking probe score monotonically track the probability that BERW/BERW is better than CMA?

We plot an empirical curve:

> probe bin → `Pr(label = BERW)`

using quantile bins + Wilson confidence intervals.

## Artifacts

- `bbob_B200_d40_calibration.(png|pdf)`: bbob-noisy `D=40`, budget `B=200×D` (test instances 6–15).
- `bbob_B500_d40_calibration.(png|pdf)`: bbob-noisy `D=40`, budget `B=500×D` (test instances 6–15).

## Inputs

- `evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/decision_points.csv`
- `evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/decision_points.csv`

## Reproduce

```bash
python3 tools/plot_probe_calibration.py \
  --decision-points evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/decision_points.csv \
  --probe-key misranking_rd \
  --threshold-json evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/train_test_threshold_misranking_rd_log10_regret_mean.json \
  --use-test-split \
  --out evidence/probe_calibration_bbob_noisy/bbob_B200_d40_calibration.png \
  --out-pdf evidence/probe_calibration_bbob_noisy/bbob_B200_d40_calibration.pdf \
  --title "bbob-noisy D=40, B=200D: calibration (test split)"

python3 tools/plot_probe_calibration.py \
  --decision-points evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/decision_points.csv \
  --probe-key misranking_rd \
  --threshold-json evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/train_test_threshold_misranking_rd_log10_regret_mean.json \
  --use-test-split \
  --out evidence/probe_calibration_bbob_noisy/bbob_B500_d40_calibration.png \
  --out-pdf evidence/probe_calibration_bbob_noisy/bbob_B500_d40_calibration.pdf \
  --title "bbob-noisy D=40, B=500D: calibration (test split)"
```


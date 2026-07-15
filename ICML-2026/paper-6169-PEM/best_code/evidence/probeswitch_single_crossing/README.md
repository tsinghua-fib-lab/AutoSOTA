# ProbeSwitch Single-Crossing Check — Evidence Package

Goal: empirically validate (or falsify) the **single-crossing** assumption behind a
threshold decision rule for ProbeSwitch.

We plot the empirical conditional advantage curve:

> `Δ(p) = E[ log10(best_f_CMA) - log10(best_f_BERW) | probe = p ]`

If `Δ(p)` is approximately increasing and crosses zero once, a threshold policy is
well motivated. If it is multi-crossing / non-monotone, we must either (i) downgrade
the claim or (ii) move to a richer policy class.

## What’s inside

- `bbob_B200_d40_single_crossing.png`: bbob-noisy `D=40`, budget `B=200×D`.
- `bbob_B500_d40_single_crossing.png`: bbob-noisy `D=40`, budget `B=500×D`.

## Data source

- `evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/decision_points.csv`
- `evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/decision_points.csv`

## Reproduce

```bash
python3 tools/plot_probeswitch_single_crossing.py \
  --decision-points evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/decision_points.csv \
  --probe-key misranking_rd \
  --threshold-json evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/train_test_threshold_misranking_rd_log10_regret_mean.json \
  --out evidence/probeswitch_single_crossing/bbob_B200_d40_single_crossing.png \
  --title "bbob-noisy D=40, B=200D: empirical advantage curve"

python3 tools/plot_probeswitch_single_crossing.py \
  --decision-points evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/decision_points.csv \
  --probe-key misranking_rd \
  --threshold-json evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/train_test_threshold_misranking_rd_log10_regret_mean.json \
  --out evidence/probeswitch_single_crossing/bbob_B500_d40_single_crossing.png \
  --title "bbob-noisy D=40, B=500D: empirical advantage curve"
```

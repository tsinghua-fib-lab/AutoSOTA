# Probe Threshold Transfer (Zero Tuning) — Evidence Package

Goal: evaluate whether ProbeSwitch can be used **without per-task threshold tuning**.

We evaluate *zero-tuning* transfer: learn a single misranking-probe threshold on one setting
(COCO bbob-noisy, train split), then **freeze it** and apply it to other budgets/tasks without re-fitting.

## What’s inside

- `transfer_summary.csv`: per-(target, method) metrics (accuracy + regret).
- `transfer_summary.md`: a compact table (auto-generated).

## How to read the table

Each target row evaluates multiple decision rules on the same `decision_points.csv`:

- `bbob_B200` / `bbob_B500`: thresholds learned on COCO (D=40) under a train/test split.
- `fixed0p12` / `fixed0p18` / `fixed0p22`: explicit fixed thresholds (for clarity).
  - `fixed0p22` is a **conservative / safer default**: it collapses toward “do not switch” on targets
    where `always_cma` dominates (e.g., HPO), while still retaining gains on several stochastic targets.
- `target_tuned`: target’s own train/test learned threshold (reference upper bound; *not* transfer).
- `always_cma` / `always_berw`: probe-free baselines.

Regret is reported in `log10` scale (same definition as `tools/probe_threshold_train_test.py`):
for each decision point, regret is measured relative to the better of the two base optimizers
(CMA vs BERW) on that instance.

## Key takeaway

Across **D=40** COCO budgets and multiple external decision-evidence datasets
(mini-batch logreg, real-data heavy-tail MLP, RL-style policy search), a COCO-learned threshold
often transfers with small degradation vs target-tuned, and improves over `always_cma` in regimes
where switching is genuinely needed.

The same transfer can *fail* in low dimensions (notably D=10), consistent with our documented boundary:
in low-misranking / low-dim regimes, switching is unnecessary and can be harmful.

Honest boundary:
- Some targets (e.g. a noisy-HPO snapshot) are dominated by `always_cma`, so any transferred threshold
  should correctly collapse toward “do not switch” (low regret even with low accuracy).
- Some small-seed tasks (e.g. LQR decision evidence with ties) are noisy for threshold learning and are
  included mainly as a transfer sanity check, not as a primary claim.

## Reproduce

```bash
python3 tools/probe_threshold_transfer.py \
  --out-dir evidence/probe_threshold_transfer \
  --probe-key misranking_rd --loss log10 \
  --source bbob_B200:evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/train_test_threshold_misranking_rd_log10_regret_mean.json \
  --source bbob_B500:evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/train_test_threshold_misranking_rd_log10_regret_mean.json \
  --fixed-threshold fixed0p12:0.12 --fixed-threshold fixed0p18:0.18 --fixed-threshold fixed0p22:0.22 \
  --target bbob_B200_d40:evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/decision_points.csv \
  --target bbob_B500_d40:evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/decision_points.csv \
  --target bbob_B200_d10:evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200_d10/decision_points.csv \
  --target bbob_B200_d20:evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200_d20/decision_points.csv \
  --target logreg_synth:evidence/application_logreg_minibatch_decision_accuracy/decision_points.csv \
  --target logreg_breast_cancer:evidence/application_logreg_minibatch_breast_cancer_decision_accuracy/decision_points.csv \
  --target logreg_digits0:evidence/application_logreg_minibatch_digits0_decision_accuracy/decision_points.csv \
  --target mlp_digits0_heavytail_vs_noise_switch:evidence/application_mlp_minibatch_digits0_heavytail_sigma1p0_decision_accuracy_vs_noise_switch/decision_points.csv \
  --target rl_cartpole_cma_vs_berw:evidence/application_rl_cartpole_heavytail_quadratic_cost_decision_accuracy/decision_points.csv \
  --target hpo_noisy_logreg_digits0_sigma1p0:evidence/application_hpo_noisy_logreg_digits0_sigma1p0_decision_accuracy/decision_points.csv \
  --target lqr_heavytail_control:evidence/application_lqr_heavytail_control_decision_accuracy/decision_points.csv
```

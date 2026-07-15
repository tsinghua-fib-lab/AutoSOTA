# CartPole (heavy-tail) — Decision Evidence (CMA vs BERW)

Goal: support **ProbeSwitch / threshold transfer** on an RL-style evaluation setting by constructing the same
decision-evidence artifact as on COCO:

> `decision_points.csv`: probe values → which base optimizer is better on the same instance.

This package is derived from the source RL run:
`evidence/application_rl_cartpole_heavytail_quadratic_cost/`.

## What’s inside

- `decision_points.csv`: per-seed outcomes for `CMA-ES-sep` vs `BERW-HeteroRobust`, plus probe values.
- `summary.json`: quick sanity counts (ties + probe accuracies at fixed thresholds).
- `train_test_threshold_misranking_rd_log10_regret_mean.json`: a train/test learned threshold (used by transfer tables).
- `train_test_threshold_sweep_misranking_rd_log10_regret_mean.csv`: sweep table behind the threshold selection.

## Reproduce

Full reproduction: `python3 tools/reproduce_all.py --workers 4` (regenerates this folder from the source run).

```bash
python3 tools/make_decision_points_from_runs_and_probes.py \
  --runs-csv evidence/application_rl_cartpole_heavytail_quadratic_cost/runs.csv \
  --probe-values-csv evidence/application_rl_cartpole_heavytail_quadratic_cost/probe_values.csv \
  --key-cols seed --instance-col seed \
  --algo-cma CMA-ES-sep --algo-berw BERW-HeteroRobust \
  --metric post_true --lower-is-better \
  --output-dir evidence/application_rl_cartpole_heavytail_quadratic_cost_decision_accuracy

python3 tools/probe_threshold_train_test.py \
  --decision-points evidence/application_rl_cartpole_heavytail_quadratic_cost_decision_accuracy/decision_points.csv \
  --probe-key misranking_rd --loss log10 --selection regret_mean_then_threshold \
  --train-instances 1-25 --test-instances 26-50 \
  --output-json evidence/application_rl_cartpole_heavytail_quadratic_cost_decision_accuracy/train_test_threshold_misranking_rd_log10_regret_mean.json \
  --output-csv evidence/application_rl_cartpole_heavytail_quadratic_cost_decision_accuracy/train_test_threshold_sweep_misranking_rd_log10_regret_mean.csv
```

# Probe decision accuracy (bbob-noisy, COCO noise-free labels; i=1–15, B=500×D)

Goal: quantify whether a tiny *probe* can predict which base optimizer should win on a given problem instance, and how regret-aware threshold selection behaves at a higher budget.

Setup:
- Suite: `bbob-noisy`, `D=40`, functions `1–30` (COCO ids `101–130`), instances `1–15` → `n_total=450` (non-ties: `428`).
- Outcome labels: compare `best_f` between:
  - `CMA-ES-sep`
  - `BERW-Hetero`
  using the **COCO noise-free** `bbob_summary.csv` extracted from `exdata/` under budget `B=500×D`.
- Probes:
  - misranking probe: `rank_disagreement` on a small CMA-style candidate set (2 draws)
  - variance probe: `rel_std(f(x0))` from repeated evaluations at `x0`

Key result (decision accuracy at the default thresholds used in this repository, ties dropped):
- misranking-probe (`t=0.12`): **0.755**
- variance-probe (`t=0.05`): **0.720**
- always choose CMA baseline: **0.507**
- always choose BERW baseline: **0.493**

Train/test threshold selection (train instances `1–5`, test instances `6–15`):
- Accuracy-opt thresholds (maximize train accuracy):
  - misranking: `t=0.165`, test accuracy **0.761** (`train_test_threshold_misranking_rd.json`)
  - variance: `t=0.01`, test accuracy **0.702** (`train_test_threshold_variance_rel_sd.json`)
- Regret-aware thresholds (scale-robust **log10 regret**, minimize mean regret on train):
  - misranking: accuracy-opt `t=0.165` gives test mean-log10-regret **0.0259**; regret-opt picks `t=0.12` and improves to **0.0226**
  - variance: accuracy-opt `t=0.01` gives test mean-log10-regret **0.0832**; regret-opt `t=0.02` improves to **0.0649**

Files:
- `summary.json`: aggregate metrics + confusion matrices (decision accuracy at fixed thresholds `t=0.12` / `t=0.05`)
- `decision_points.csv`: per-instance probe values and labels
- `threshold_sweep.csv`: accuracy vs threshold sweep (misranking + variance)
- `train_test_threshold_*.json`, `train_test_threshold_sweep_*.csv`: train/test threshold selection (accuracy-opt, raw regret scale)
- `train_test_threshold_*_log10_*.json`, `train_test_threshold_sweep_*_log10_*.csv`: regret-aware selection outputs (log10 regret)
- `threshold_kfold_k5_*.json`: instance-level k-fold robustness checks for threshold selection (cross-split stability).

Reproduce:
```bash
python3 tools/probe_decision_accuracy.py \
  --results-dir Results/bbob_noisy_d40_i1-15_probe_labels_B500/noisefree \
  --dimension 40 --functions 1-30 --instances 1-15 --budget 500 \
  --algo-cma CMA-ES-sep --algo-berw BERW-Hetero \
  --misranking-threshold 0.12 --variance-threshold 0.05 \
  --output-dir evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500

python3 tools/probe_threshold_train_test.py \
  --decision-points evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/decision_points.csv \
  --probe-key misranking_rd --train-instances 1-5 --test-instances 6-15 --tmax 0.3 --tstep 0.005 \
  --loss log10 --selection regret_mean_then_threshold \
  --output-json evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/train_test_threshold_misranking_rd_log10_regret_mean.json \
  --output-csv evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/train_test_threshold_sweep_misranking_rd_log10_regret_mean.csv
```

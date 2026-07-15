# Experiments (what, why, where)

This document maps each key claim to:

- what experiment(s) support it,
- how to reproduce them,
- where the main artifacts are written.

For a one-command end-to-end run, see `README.md`.

> **Note on naming:** This document uses paper names. Code/CSV files use internal names.
> See `docs/ALGORITHMS.md` for the mapping (e.g., "Residual Bootstrapping" = `BERW-Hetero`).

## Claim A: Fixed-budget sample efficiency beats evaluation-stage resampling

**Question addressed:** Under a *fixed evaluation budget*, can classic resampling / UH-CMA-ES
keep up, or does Residual Bootstrapping (RB-PEM) achieve better convergence by preserving depth?

**Evidence:**

- `evidence/hansen_test_fixed_budget/`
  - Money plot (noise-free best vs evals): `evidence/hansen_test_fixed_budget/money_plot_noisefree_d40_B100_f10-13-16-25_with_resample.png`
  - Fixed-budget significance tables: `evidence/hansen_test_fixed_budget/noisefree/pairwise_sign_test_with_resample.csv`
  - Residual-pool diagnostics: `evidence/hansen_test_fixed_budget/diagnostics/`

- Budget scaling / robustness:
  - `evidence/hansen_test_fixed_budget_grid/` (D=40)
  - `evidence/hansen_test_fixed_budget_grid_d20/` (D=20)

**Reproduce:**

```bash
python3 tools/reproduce_all.py --workers 4 --suite coco
```

## Claim B: Probe-and-Switch is a deployable default (calibration, overhead, and transfer)

**Question addressed:** Is the Probe-and-Switch rule "hand-tuned", fragile, or too costly?

**Evidence:**

- COCO decision evidence + learned thresholds:
  - `evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/`
  - `evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/`
- Calibration plots (test split):
  - `evidence/probe_calibration_bbob_noisy/`
- Threshold transfer and overhead summary:
  - `evidence/probe_threshold_transfer/`
  - `evidence/probeswitch_transfer_overhead_summary/`
- End-to-end transfer win-rates on external tasks:
  - `evidence/probeswitch_external_transfer/`

**Reproduce:**

```bash
python3 tools/reproduce_all.py --workers 4 --suite probes
```

## Claim C: Cross-domain fixed-budget generalization (external tasks)

**Question addressed:** Are gains specific to COCO, or do they appear in standard ML-style tasks?

**Evidence:**

- RL policy search (CartPole, heavy-tail): `evidence/application_rl_cartpole_heavytail_quadratic_cost/`
- RL policy search (Pendulum): `evidence/application_rl_pendulum_heavytail/`
- Noisy HPO (digits0): `evidence/application_hpo_noisy_logreg_digits0_sigma1p0/`
- State-dependent heavy-tail control (LQR): `evidence/application_lqr_heavytail_control_fixed_budget_resample/`
- Nonconvex mini-batch MLP (digits0): `evidence/application_mlp_minibatch_digits0_heavytail_sigma1p0/`

**Reproduce:**

```bash
python3 tools/reproduce_all.py --workers 4 --suite external
```

## Claim D: Mechanistic / diagnostic evidence (when it fails, and why)

**Question addressed:** Do the probe metrics behave sensibly, and do we have measurable diagnostics?

**Evidence:**

- Misranking metric sanity check (RD vs Kendall/top-μ): `evidence/misranking_metric_sandwich/`
- Variance-proxy counterexample (radial/state-dependent noise): `evidence/probe_decoupling_radial/`
- Quadratic mechanism check (misranking → update dispersion): `evidence/theory_update_dispersion_quadratic/`
- Single-crossing check (threshold policy justification): `evidence/probeswitch_single_crossing/`
- RB-PEM estimator ablations (fixed-budget slice):
  - `evidence/berw_reeval_ablation_fixed_budget/`
  - `evidence/berw_bootstrap_samples_ablation_fixed_budget/`
  - `evidence/berw_hetero_model_ablation_fixed_budget/`

**Reproduce:**

```bash
python3 tools/reproduce_all.py --workers 4 --suite diagnostics
```

## Additional baselines & ablations (regenerated on demand)

These baselines and ablations are first-class in the paper; their per-problem CSVs are
**not pre-shipped** under `evidence/` (run the commands below to regenerate). All are wired
into the `external` suite of `reproduce_all.py`.

- **Log-weight ablation** (RB-PEM / Probe-and-Switch with standard CMA-ES log weights):
  ```bash
  bash tools/run_log_weight_ablation.sh && bash tools/run_log_weight_ablation_extra.sh
  python3 tools/merge_log_weight_results.py
  python3 tools/analyze_log_weight_ablation.py Results/log_weight_ablation/bbob_summary_all_budgets.csv
  ```
- **LRA-CMA-ES baseline** (Nomura et al., 2025): `bash tools/run_lra_cmaes_baseline.sh`
- **UH-CMA-ES rank grid** (full 30-function grid): `bash tools/run_uh_cmaes_baseline.sh`
- **Rank-by-probe figure (Fig. 4):** after the three runs above,
  `python3 tools/plot_rank_by_probe.py` writes `evidence/paper_figures/figure_rank_by_probe.pdf`.

Or run all four at once:

```bash
python3 tools/reproduce_all.py --workers 4 --suite external
```

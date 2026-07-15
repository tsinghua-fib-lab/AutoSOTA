# Algorithm Names (Glossary)

This document maps **paper names** (used in the paper) to **code names** (used in scripts, CSVs, and CLI flags).

## Naming Convention

| Paper Name | Code Name | Description |
|------------|-----------|-------------|
| **Residual Bootstrapping (RB-PEM)** | `BERW-Hetero` | Selection-stage uncertainty integration via residual bootstrapping |
| **Probe-and-Switch** | `ProbeSwitch-MR(t=0.12)` | Adaptive switching between CMA-ES and RB-PEM |
| CMA-ES | `CMA-ES-sep` | Separable CMA-ES baseline |
| Resample (k=5) | `CMA-ES-Resample(k=5)` | Fixed-k evaluation-stage resampling |
| Resample (k=10) | `CMA-ES-Resample(k=10)` | Fixed-k evaluation-stage resampling |
| UH-CMA-ES | `UH-CMA-ES(maxevals=30)` | Uncertainty-handling CMA-ES baseline |
| LRA-CMA-ES | `LRA-CMA-ES` | Learning-rate-adaptation CMA-ES baseline (Nomura et al., 2025) |
| RB-PEM (log-weight) | `BERW-Hetero-LogW` | RB-PEM ablation with standard CMA-ES log weights |
| Probe-and-Switch (log-weight) | `ProbeSwitch-MR-LogW(t=0.12)` | Probe-and-Switch ablation with standard CMA-ES log weights |

## Code Names in Detail

This repository uses short **string names** for algorithms in:

- CLI flags such as `--algorithms "..."`,
- CSV columns/legends (e.g., the `algorithm` column in `bbob_summary.csv`),
- evidence tables under `evidence/`.

These names are stable and human-readable. You should never need internal class names for reproduction.

### COCO / BBOB (canonical registry)

For COCO experiments, the canonical mapping is in `tools/run_coco_bbob_noisy.py` (`ALGORITHMS` dict).

#### Baselines

- `CMA-ES`: full-covariance CMA-ES baseline.
- `CMA-ES-sep`: separable CMA-ES baseline (**"CMA-ES" in paper**).
- `CMA-ES-Resample(k=2|3|5|10)`: fixed-k evaluation-stage resampling (**"Resample (k=N)" in paper**).
- `UH-CMA-ES(maxevals=10|30)`: uncertainty-handling baseline (**"UH-CMA-ES" in paper**).
- `LRA-CMA-ES`: learning-rate-adaptation CMA-ES baseline (**"LRA-CMA-ES" in paper**).

#### Residual Bootstrapping Family (paper: "RB-PEM" or "Residual Bootstrapping")

- `BERW-Hetero`: Bootstrap-Estimated Rank Weights with heteroscedastic noise model (**"Residual Bootstrapping" in paper**).
- `BERW-HeteroRobust`: more conservative variant for heavy-tail/state-dependent-noise tasks.
- Ablation variants:
  - `BERW-Hetero(bs=16)`, `BERW-Hetero(bs=64)` (bootstrap sample count)
  - `BERW-Hetero(reeval=0)`, `BERW-Hetero(reeval=3)` (reevaluation cap)
  - `BERW-HeteroTMatch`, `BERW-HeteroVar` (noise model variants)
  - `BERW-Hetero-LogW` (standard CMA-ES log weights instead of the bootstrap-internal map; paper: log-weight ablation)

#### Probe-and-Switch Family (paper: "Probe-and-Switch")

- `ProbeSwitch-MR(t=0.12)`: misranking-probe switch with threshold τ=0.12 (**"Probe-and-Switch (τ=0.12)" in paper**).
- `ProbeSwitch-MR(t=0.22)`: misranking-probe switch with threshold τ=0.22 (**"Probe-and-Switch (τ=0.22)" in paper**).
- `ProbeSwitch-MR-Robust(t=0.12)`, `ProbeSwitch-MR-Robust(t=0.22)`: robust variants for cross-domain transfer.
- `ProbeSwitch-MR-LogW(t=0.12)`: log-weight ablation variant (standard CMA-ES log weights).

## External Tasks

External task runners each define an `algo_map` with the **same code names** as above:

- `tools/run_rl_cartpole_heavytail.py`
- `tools/run_rl_pendulum_heavytail.py`
- `tools/run_hpo_noisy_logreg.py`
- `tools/run_lqr_heavytail_control.py`
- `tools/run_mlp_minibatch_sweep.py`

If a runner rejects an algorithm name, it prints the list of supported keys.

## Display Name Mapping

For programmatic access, the file `src/utils/display_names.py` provides a mapping from code names to paper-facing display names used in figures and tables.

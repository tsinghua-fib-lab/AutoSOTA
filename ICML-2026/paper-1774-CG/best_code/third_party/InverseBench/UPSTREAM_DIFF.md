# InverseBench — provenance and local additions

This directory vendors the git-tracked files of our InverseBench fork
(`dgeyfman/InverseBenchBenchmark`) at commit `b26ffa2`, as the trusted engine for
the black-hole experiment. It is essentially verbatim, with two deliberate
deviations for a public release:

1. **Privacy scrub.** Personal paths were replaced by env-vars in `main.py`, and
   a token-bearing data URL in `README.md` was replaced by the plain data-page
   link. Nothing else in the kept files was edited.
2. **Dropped extras.** Post-hoc analysis / plotting scripts that are not part of
   the reproduction workflow were removed (listed below); they remain in the
   original fork.

Large run artifacts (datasets, checkpoints, W&B logs, `step_samples/`, `exps/`,
output PDFs/zips — all symlinked to `/extra` in the original working tree) were
never tracked and are fetched by
[`experiments/black_hole/download.sh`](../../experiments/black_hole/download.sh).

The black-hole experiment is driven by the **clean entry point**
[`experiments/black_hole/run.py`](../../experiments/black_hole/run.py), which imports
the engine from here. The original `main.py` (a Hydra script) is kept only for
reference.

## Lineage

| | commit | description |
|---|---|---|
| Upstream public repo | `devzhk/InverseBench` | https://github.com/devzhk/InverseBench (ICLR 2025) |
| Fork point | `b6ed607` | last upstream commit our fork branched from |
| Our fork HEAD | `b26ffa2` | this vendored snapshot |

To reproduce the comparison yourself:

```bash
git clone https://github.com/devzhk/InverseBench            # upstream
git -C InverseBench diff b6ed607 <our-fork-HEAD>            # full local diff
```

The diff of the estimator + engine files is saved here as
[`local_additions.patch`](local_additions.patch) for convenience.

## What our fork adds on top of upstream

### Core CBG estimators (used by the black-hole experiment)

| File | Role |
|---|---|
| `algo/reinforce.py` | `CalibratedGuidanceReinforce` — the gradient-free CBG estimator (Eq. 20). Wraps `calibrated_guidance.{guidance,inference}`; supports the DDPM inner-loop posterior (`InverseBenchModel`) and the mean-flow renoise posterior. |
| `algo/meanflow_posterior.py` | `MeanFlowDiffusionPosterior` + `load_meanflow_net` — one-step mean-flow sampler for `p(x0 \| xt)` (the fast "renoise" variant of §5.3 / Table 2). |
| `algo/dps.py` (modified) | adds `batch_reinforce_grad` — the **"slow"** REINFORCE-through-DPS path (the original, pre-framework implementation). |
| `algo/dps_original.py` (new) | the unmodified upstream DPS, kept as a clean baseline. |

### Engine modifications supporting the estimators

| File | Change |
|---|---|
| `utils/scheduler.py` | subset / evenly-spaced scheduler helpers used by the inner loops. |
| `utils/diffusion.py` | `DiffusionSampler` tweaks (filtering, `t_begin`). |
| `training/dataset.py` | black-hole dataset loading. |
| `configs/algorithm/{dps,daps,diffpir,reddiff}.yaml` | config updates. |
| `main.py` | hardcoded black-hole launch branch, paths scrubbed to env-vars (reference only; superseded by `experiments/black_hole/run.py`). |
| `algo/*.py` (many baselines) | minor compatibility edits (e.g. scheduler signature) rippling from the above. |

### Auxiliary analysis scripts (removed from this vendored copy)

The fork also carried post-hoc analysis / plotting scripts that are **not** part
of the reproduction workflow and not imported by `experiments/black_hole/`. They
were dropped from this public copy (and remain in the original fork):

```
analyze_blackhole_results.py      crps_analysis.py
basic_distribution_comparison.py  crps_grouped_analysis.py
distributional_analysis*.py       k_ablation_renoise_analysis.py
qualitative_distribution_*.py     evaluate_existing_results.py
run_baseline.py  sort_results.py  time_methods.py  run_generation.sh
scripts/plot_*.py                 pool_explanation.md  pool_convergence_proof.md
```

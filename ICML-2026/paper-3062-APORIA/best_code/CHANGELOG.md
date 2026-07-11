# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0]

### Added
- Initial repository scaffolding: modular `aporia` library, TOML-based
  configuration system, and project-level metadata files.
- TOML configs for the SOCRATES-300K dataset and the CoQA bridge dataset.
- All eight notebooks migrated from `HALL_lib` to the new `aporia` API:
  - `code/StructuralAnalysis.ipynb`
  - `code/LabelPropagationAnalysis.ipynb`
  - `code/StructuralAnalysis__Sensitivity.ipynb` (Appendix C)
  - `code/Pipeline.ipynb` (Figure 1)
  - `code/LabelPropagation__LambdaSensitivity.ipynb` (Figure 5)
  - `code/LabelPropagation__TrainingSensitivity.ipynb` (Figure 4)
  - `code/LabelPropagation__ProjectionAblation.ipynb` (Appendix F)
  - `code/LabelPropagation__ClassifierAblation.ipynb` (Appendix E)
- New library classes to support the Appendix E classifier ablation:
  - `aporia.IdentityProjection` — no-op projector for the SBERT 384-D
    baseline column.
  - `aporia.CentroidFeaturesProjection` — three-dimensional feature map
    `(d_G, d_H, v^T x)` used by the *Centroid-feat 3D* row of Table 7.
  - `aporia.CentroidPropagator` — nearest-centroid classifier in
    projected space; supports Euclidean and cosine metrics.
  - `aporia.SKLearnPropagator` — adapter that wraps any scikit-learn
    estimator (LR / SVM / kNN in the paper) as a propagator.
- `run_label_propagation_experiment` and `run_full_label_propagation_study`
  now accept `propagator_class` / `propagator_kwargs`, defaulting to
  `WassersteinLabelPropagator` for backward compatibility.  Per-pair
  cache filenames are namespaced by propagator name to avoid collisions
  between ablation runs that share a projector.

### Changed
- Dataset/model metadata is now read from `config/*.toml` rather than
  hard-coded in each notebook.  `model_names`, `model_latextags`,
  `best_reg_lambda`, and `maxResponsesPerPrompt` are still in scope as
  aliases derived from `cfg`, so existing plotting code is unaffected.
- Every notebook now begins with a `chdir`-to-repo-root cell, so it can
  be launched from either the repo root or from `code/` without breaking
  relative paths.  `CONFIG_PATH` is `"config/socrates.toml"` accordingly.
- `run_full_label_propagation_study` and `run_full_lambda_sensitivity_study`
  derive `model_ids` from the data when both `model_ids` and
  `prompt_ids_by_model` are left at their default `None`.  This prevents
  a `KeyError` when the config declares models not present in the data.
- All notebooks now write figures to `figs/` (previously `img/`/`imgs/`),
  matching the directory referenced by the LaTeX `\includegraphics` paths
  in the paper.

### Fixed
- Added `networkx` to the `notebook` optional-dependency group; it is
  required by `Pipeline.ipynb` but was previously missing.
- Removed the unreachable `tomli` fallback in `aporia.config`; the
  project already requires Python ≥ 3.11, which has `tomllib` in stdlib.
- Documented the `n_permutations=0` (or any falsy value) skip-null
  behaviour of `run_structural_analysis` and `analyse_prompt`.  The
  Sensitivity notebook relies on it.
- Notebooks are now self-contained w.r.t. LaTeX rendering.  Each one
  loads `cfg` first, then sets `plt.rcParams['text.latex.preamble']` via
  `ap.matplotlib_latex_preamble(cfg)`, which emits `\newcommand` macros
  matching the paper's model-name conventions.  This removes the
  notebooks' dependency on `camera-ready.tex` for plot rendering.

### Notes
- The library was renamed from `hallgeo` (the working placeholder used
  during development) to **`aporia`** — *Aggregate Prompt-wise
  Observation Retrieving Instability via Asymmetry*.  See the README's
  "About the name" section for the conceptual motivation.
- The default cache root has changed from `cache-icml/` (legacy) to
  `cache/socrates/` (set in `config/socrates.toml`).  Pre-existing caches
  in the legacy layout will not be picked up automatically; either move
  them under `cache/socrates/` or override `[cache] root` in the TOML.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TSL (Tensor Separation Learning) is a glass-box regression model. A fitted model is a
sum of *stages*; each stage is the ordered difference of two non-negative rank-1
products of univariate functions. The core is a Rust crate (`tsl_rust`, library name
`tsl`); `tsl-py/` is a PyO3/maturin wrapper exposing it to Python with a scikit-learn
API; `tsl-split-evolution-dashboard/` (`tslviz`) is a separate FastAPI+D3 app that
visualizes split-event logs written during a fit.

## Commands

This is a Cargo workspace (root crate `tsl_rust` + member `tsl-py`).

**Rust — always pass `--release`** (tests run numerical workloads; debug is far too slow):
```sh
cargo build --release
cargo test -p tsl_rust --release                       # full core test suite (owns tests/)
cargo test -p tsl_rust --release test_fit_result_is_correct   # single test by name
cargo test --release --test forest                     # one integration-test file (tests/forest.rs)
```
The core is pure Rust — no system libraries are required to build. Default cargo features
are `use-rayon` (parallel bag fitting) and `evo-logging` (SQLite split logging).

**Python (`tsl-py/`)** — maturin can't auto-detect the machine's Python 3.14, so use the
project's 3.13 venv with `VIRTUAL_ENV` set:
```sh
# from tsl-py/
VIRTUAL_ENV=/Users/jin/Documents/TSL/.venv /Users/jin/Documents/TSL/.venv/bin/maturin develop
VIRTUAL_ENV=/Users/jin/Documents/TSL/.venv /Users/jin/Documents/TSL/.venv/bin/python -m pytest python/tests/
# single test:
.../bin/python -m pytest python/tests/test_models.py::test_name
```
Note: `tsl-py` is an `extension-module` cdylib, so its Rust test harness can't link
libpython on Linux — exercise that crate through the Python tests, not `cargo test`.

**Benchmarks / profiling:** wall-clock benches are `#[ignore]`d tests in
`tests/histogram_binning.rs` (`cargo test --release --test histogram_binning -- --ignored --nocapture --test-threads=1`); profiling harness is in `scripts/perf/` (samply). See `scripts/perf/README.md`.

**Examples** (reproduce the paper figures): `python tsl-py/examples/california.py` etc.; see `tsl-py/examples/README.md`.

## Commits

Follow the Conventional Commits convention in [`CONTRIBUTING.md`](CONTRIBUTING.md).
Reserve `feat`/`fix`/`perf` for shipped code (`src/`, `tsl-py/src/`, `tsl-py/python/`).
Changes under `tsl-py/examples/` are not library changes: use `docs(examples):` for the
scripts and README, and `chore(examples):` for regenerated `figures/` or pretrained
`models/` binaries.

## Describe the code as it is (VERY IMPORTANT)

Write commit messages, comments, and docstrings **as if the current behavior was
always intended**. Describe what the code *does*, not what it used to do, what
changed, or what was just fixed — that history already lives in git and PRs.
Don't justify a change in a comment or flag that something "now works". Comment
only non-obvious intent or invariants, and leave self-evident code uncommented.

The tell is any phrasing that only makes sense relative to a past version —
"now", "no longer", "previously", "used to", "instead of", "fixed", "still" — or
framing a thing by what it is *not* ("no shadows, no gradients"). Cut it and
state the present behavior directly.

**Docstrings — describe, don't contrast with a rejected alternative:**
- ✗ `"""Flat theme. No translucency, no shadows, no gradient mesh. Solid colours…"""`
- ✓ `"""Flat theme: solid colours, hairline borders, a faint dot-grid…"""`

**Comments — state the present behavior, don't narrate the edit:**
- ✗ `# changed to inches so the gap no longer scales with the figure`
- ✓ `# inches, so the gap stays constant as the figure auto-sizes`
- ✗ `# handle non-contiguous input (this used to panic)`
- ✓ `# rows may be non-contiguous; copy to a contiguous buffer before the BLAS call`

**Comment only what isn't obvious from the code:**
- ✗ `x = x + 1  # increment x`
- ✓ `# λ₊, λ₋ ≥ 0, so the f₊ − f₋ difference carries the sign; clamp negatives here`

**Commit subjects — name the behavior, not the repair:**
- ✗ `fix(plot): bars no longer overflow the card; the spacing was too big before`
- ✓ `fix(plot): keep bar labels inside the card`

## Architecture

The model is a three-level hierarchy, and the `src/` module tree mirrors it exactly.
Read these three files first: `src/grid_tensor.rs`, `src/stage_predictor.rs`, `src/forest.rs`.

1. **`GridTensor`** (`src/grid_tensor/`) — one fitted separable component. Stored in
   **two-tensor form**: per-feature `backbone_values` (b ≥ 0) and `tilt_values` (d ∈ ℝ),
   plus scalars `lambda_plus`/`lambda_minus`. Prediction is
   `f = λ₊·∏ⱼ bⱼe^{dⱼ} − λ₋·∏ⱼ bⱼe^{−dⱼ}`. The backbone is the shared magnitude (a gate:
   near-zero backbone switches the component off); the tilt sum is the signed direction.

2. **`StagePredictor`** (`src/stage_predictor/`) — one boosting stage: a bag of `n_trees`
   `GridTensor`s aggregated (`Mean`/`GeometricMean`/`Combined`) into one
   `primary_grid_tensor`, plus OLS coefficients `scaling_plus`/`scaling_minus`.

3. **`TSL`** (`src/forest/`) — the boosted model, a `Vec<StagePredictor>`. `predict` sums
   stage predictions. `fit_boosted` (`src/forest/fitter.rs`) runs `epochs` rounds: each
   round fits a `StagePredictor` on the current residuals, then refits the coefficients of
   **all** stages so far via incremental OLS over the per-stage `f₊` and `−f₋` columns
   ("orthogonal greedy"). `n_iter` decays by `decay` after the first epoch.

### Two critical invariants

- **Scaling is applied exactly once**, at the `StagePredictor` level via
  `scaling_plus`/`scaling_minus` from the OLS solve. `GridTensor::predict` and
  `extract_two_tensor_predictions_unscaled` deliberately return *unscaled* `f₊`/`f₋`; the
  legacy `grid.scaling` field is ignored in two-tensor mode. Do not multiply by it.
- **Positivity:** `λ₊, λ₋ ≥ 0`, `b ≥ 0`, so `f₊, f₋ ≥ 0`. This removes the sign ambiguity
  of unconstrained tensor decompositions; signed effects come from the `f₊ − f₋`
  difference. Enforced by clamping in the solver (`tests/forest.rs` and
  `grid_tensor/fit.rs` test these invariants).

### The grid-tensor fitting loop (action/reducer pattern)

`grid_tensor::fit` (`src/grid_tensor/fit.rs`) iterates over a `FittingState` (`state.rs`):
a `SplitStrategy` (`splitting.rs`: `Random`/`Best`/`TopK`) proposes a `FittingAction`
(split/resplit/merge), and `fitting_reducer` (`reducer.rs`) applies it, calling the
`RefinementStrategy` (`refinement.rs`: `L2`/`Huber`). The per-node optimization lives in
`two_tensor_solver.rs`; final normalization in `identification.rs`; optional
quantile-binning of split candidates in `histogram_binning.rs` (the `max_bins` param).
Parameters use builder types throughout (`params.rs` in each module; the top-level
`TSLBoostedParamsBuilder` in `src/forest/params.rs` flattens them).

### Python layer (`tsl-py/src/lib.rs`)

PyO3 module `tensorsl._tensorsl` exposes `TSL`, `GridTensor`, `StagePredictor`, `FitResult`.
`TSL.fit(...)` is a classmethod taking flat hyperparameters (mapped onto the Rust
builders). `TSLRegressor` (`python/tensorsl/sklearn.py`) is the sklearn-compatible wrapper —
the main user entry point. Partial dependence is computed in Rust
(`compute_partial_dependence_function`, marginalizing over the empirical joint, not
assuming feature independence). `tensorsl.plot` (lazy-imported, needs matplotlib) holds the
diagnostics. `TSL.load(...)` reads the legacy MPF `.bin` format.

### evo-logging → dashboard

When `visualdb`/`visualdb_path` is set on a fit (and the `evo-logging` feature is on), every
split/resplit/merge and per-stage snapshot is streamed to a SQLite DB (`src/logging/`,
`src/grid_tensor/logging_helpers.rs`). `tsl-split-evolution-dashboard` (`tslviz --db
run.sqlite`) reads that DB read-only to replay how each stage was built.

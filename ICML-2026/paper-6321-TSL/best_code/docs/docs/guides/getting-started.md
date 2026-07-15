# Getting started

## Rust toolchain

TSL's Python and R packages compile the Rust core at install time, so **a working Rust
toolchain is required first**. Install it with [rustup](https://rustup.rs) if Rust is not
already on your machine:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Development

This is a Cargo workspace (root crate `tsl_rust` + member `tsl-py`).

**Rust core** — always pass `--release` (the tests run numerical workloads; debug is far too
slow):

```sh
cargo build --release
cargo test -p tsl_rust --release                 # full core test suite
cargo test -p tsl_rust --release test_name        # a single test by name
```

## Python

### Installation

The Python package is published on PyPI as **`tensorsl`** and imported under the same name:

```sh
pip install tensorsl

# optional extras
pip install "tensorsl[plots]"     # matplotlib for tensorsl.plot
pip install "tensorsl[examples]"  # EBM, XGBoost, SepALS for comparisons
```

Source builds compile the Rust extension at install time, so the Rust toolchain above is
required. The core is pure Rust — no system math libraries are required.

### First fit

`TSLRegressor` is a drop-in scikit-learn regressor:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from tensorsl.sklearn import TSLRegressor

rng = np.random.default_rng(0)
X = rng.uniform(0.0, 1.0, size=(1000, 2))
y = 2.0 * X[:, 1] + X[:, 0] - 0.5 * X[:, 0] * X[:, 1] + 3.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = TSLRegressor(epochs=3, n_trees=4, n_iter=25, seed=1)
model.fit(X_train, y_train)

print(f"Test R^2: {model.score(X_test, y_test):.4f}")
```

The lower-level `TSL.fit(...)` classmethod returns `(model, fit_result)` and takes the same
flat hyperparameters; it expects **float64** arrays. See the
[Python API](../code/python-api.md) and the
[Hyperparameters](hyperparameters.md) reference.

### Development

**Python wrapper** — `maturin develop` builds the Rust extension and installs it into a
Python virtualenv. Point it at the project's venv by setting `VIRTUAL_ENV` (and invoking
that venv's `maturin`):

```sh
# from tsl-py/
VIRTUAL_ENV=/path/to/.venv /path/to/.venv/bin/maturin develop
/path/to/.venv/bin/python -m pytest python/tests/
```

`tsl-py` builds as a Python **extension module** — a `cdylib` compiled with the
`extension-module` feature — so the library leaves CPython's symbols to be resolved by
whatever interpreter imports it. A standalone `cargo test` binary has no interpreter to
resolve them against and fails to link (notably on Linux), so this crate is exercised
through `pytest` (which loads the module into a real interpreter), not `cargo test`. The
Rust core (`tsl_rust`) keeps its own full `cargo test` suite.

## R

### Installation

The R package **`tensorsl`** lives in the `tsl-r/` subdirectory of this repository. Install
it from the repository root with either `pak` or `remotes`:

```r
# pak (owner/repo/subdir):
pak::pak("jyliuu/TSL/tsl-r")

# remotes / devtools:
remotes::install_github("jyliuu/TSL", subdir = "tsl-r")
```

The R build compiles the same Rust core at install time, so it uses the Rust toolchain from
above and does not require system math libraries.

### First fit

`tsl()` is the R entry point for fitting a boosted glass-box model:

```r
library(tensorsl)

set.seed(1)
n <- 500
x <- matrix(runif(n * 3, -2, 2), ncol = 3, dimnames = list(NULL, c("a", "b", "c")))
y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(n, sd = 0.1)

fit <- tsl(x, y, epochs = 20L, seed = 42L, verbosity = 0L)
preds <- predict(fit, x)
```

`tsl()` accepts the same hyperparameters as the Python `TSLRegressor`; see the
[R API](../code/r-api.md) reference for the full interface.

### Development

For local iteration against the working-tree core, add an untracked
`tsl-r/src/rust/.cargo/config.toml` with:

```toml
paths = ["../../.."]
```

Then install from the `tsl-r/` directory:

```r
devtools::install_local("tsl-r")
```

## Next steps

- [The model](../math/model.md) — what TSL actually fits.
- [Hyperparameters](hyperparameters.md) — every knob, what it does, and where it maps.
- [Python API](../code/python-api.md) / [R API](../code/r-api.md) — the per-language reference.
- [Examples](../index.md#examples) — reproduce the paper figures.
- [Visualization dashboard](visualizing.md) — replay how a model was built with `tslviz`.

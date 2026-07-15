# tensorsl

R bindings for **TSL** (Tensor Separation Learning), a glass-box regression
model implemented in Rust. `tensorsl` wraps the high-performance [`tsl`](https://github.com/jyliuu/TSL)
Rust crate through [extendr](https://extendr.github.io/), exposing a small
`fit`/`predict` interface that mirrors the Python `TSLRegressor`.

## Installation

`tensorsl` compiles a Rust static library at build time. The core is pure Rust and
links no system numerical libraries, so the only prerequisite is a Rust
toolchain (`rustc >= 1.80`); install it from [rustup.rs](https://rustup.rs).

`tensorsl` lives in the `tsl-r/` subdirectory of the [TSL repo](https://github.com/jyliuu/TSL),
so installers need to be told the subdirectory:

```r
# pak (owner/repo/subdir):
pak::pak("jyliuu/TSL/tsl-r")

# remotes / devtools:
remotes::install_github("jyliuu/TSL", subdir = "tsl-r")
```

The package depends on the `tsl` core as a pinned **git** dependency, which
cargo fetches during the build — no separate checkout of the core is required.

## Usage

```r
library(tensorsl)

set.seed(1)
n <- 500
x <- matrix(runif(n * 3, -2, 2), ncol = 3, dimnames = list(NULL, c("a", "b", "c")))
y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(n, sd = 0.1)

fit <- tsl(x, y, epochs = 20, seed = 42)
fit
#> <tsl> Tensor Separation Learning model
#>   features:       3
#>   training rows:  500
#>   training error: ...

preds <- predict(fit, x)
```

`tsl()` accepts the same hyperparameters as the Python `TSLRegressor`
(`epochs`, `n_trees`, `n_iter`, `split_strategy`, `refinement_strategy`, …);
see `?tsl`.

> **Note.** A fitted `tsl` object holds an external pointer into Rust and is not
> portable across R sessions via `saveRDS()`.

## Development

The crate is at `src/rust/`. For fast local iteration against the working-tree
core (instead of the pinned git revision), an untracked
`src/rust/.cargo/config.toml` overrides the dependency with a path:

```toml
paths = ["../../.."]
```

This file is gitignored and `.Rbuildignore`d, so it never ships. Regenerate the
R wrappers after changing the Rust API with the bundled `document` binary
(run from `src/`):

```sh
cargo run --bin document --manifest-path=./rust/Cargo.toml --target-dir ./rust/target
Rscript -e 'roxygen2::roxygenise()'   # refresh NAMESPACE/man from the new wrappers
```

When the core is tagged for release, bump the `tag`/`branch` of the `tsl_rust`
git dependency in `src/rust/Cargo.toml`.

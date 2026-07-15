# scripts/perf — profiling infrastructure

Two entry points:

- `examples/profile_housing.rs` — the binary under sample. Loops a TSL
  fit on California Housing under either the exact or binned split path.
- `tests/histogram_binning.rs` — `#[ignore]`d benchmarks for wall-clock
  comparisons across max_bins values and datasets.

## Quick start

```sh
# Default (exact path, n_iter=50, reps=5)
bash scripts/perf/run_profile.sh

# Compare against the binned path
bash scripts/perf/run_profile.sh --max-bins 255

# Tweak workload
bash scripts/perf/run_profile.sh --max-bins 64 --n-iter 100 --reps 10

# Help
bash scripts/perf/run_profile.sh --help
```

The script:

1. Builds `examples/profile_housing` under the `[profile.profiling]`
   profile (release + debuginfo).
2. Records a samply JSON to `docs/perf/profile_<tag>_<timestamp>.json`
   where `tag` is `exact` or `bins<N>`.
3. Prints the saved-to path and a `samply load <file>` line to open in
   the Firefox Profiler.

`samply` is required (`cargo install samply`). The script bails out with
the install command if it's missing.

## Inspecting profiles without a browser

```sh
python3 scripts/perf/analyze_profile.py docs/perf/profile_exact_<TS>.json
python3 scripts/perf/analyze_profile.py docs/perf/profile_bins255_<TS>.json
```

Prints two tables — top by self-time and top by inclusive-time — across
all threads. Pass a substring as the second positional arg to restrict
to one thread (e.g. `profile_housing`).

## Benchmarks (no profiling)

For wall-clock comparisons rather than profiles:

```sh
cargo test --release --test histogram_binning -- --ignored --nocapture \
    --test-threads=1
```

This runs `bench_california_housing`, `bench_cps88_wages`, and
`bench_brazilian_housing`, each fitting at `max_bins=None / 255 / 64 / 32`
and printing per-config wall time and train/test MSE. Adjust within the
test bodies to add datasets.

## Criterion benchmark + CI

`benches/fit.rs` is a criterion benchmark of the core fit on California
Housing, with two functions — `fit/exact` and `fit/binned255` — over a fixed
seeded workload (see the consts at the top of the file).

```sh
cargo bench --bench fit                       # run locally
cargo bench --bench fit -- --save-baseline a  # save a named baseline
critcmp a b                                   # diff two baselines
```

`.github/workflows/bench.yml` runs this on every pull request: it measures the
PR head and its merge base on the *same* runner and writes a `critcmp` delta to
the run's job summary. It is comment-only (never gates the build) because
GitHub-hosted runners are too noisy for reliable absolute regression gating —
treat sub-10% moves as noise.

## Past findings

- `docs/perf/findings.md` — the baseline profile that motivated histogram
  binning and the AoS layout (update_statistics was 68 % self before any
  optimization).
- `docs/perf/binned_prefix_sums.md` — outcome of the binned-prefix-sums
  optimization.
- `docs/perf/aos_and_rayon.md` — outcome of the AoS layout. The rayon
  variant in that branch was rejected and is NOT on `main`.
- `docs/histogram-binning-scoping.md` — pre-implementation scoping.

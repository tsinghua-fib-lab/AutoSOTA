//! Criterion benchmarks for the core grid-tensor fit.
//!
//! Two functions mirror the two production code paths we actively tune:
//!   - `fit/exact`      — max_bins = None      (exact split search)
//!   - `fit/binned255`  — max_bins = Some(255) (histogram binning)
//!
//! Both fit a fixed, seeded subsample of California Housing
//! (`data/housing_full.csv`). The subsample size and n_iter are kept modest so
//! one `cargo bench` stays in the low-minutes range on a 2-core CI runner —
//! bump `ROWS` / `N_ITER` for a heavier local run, but keep them FIXED across
//! commits or the time series stops being comparable.
//!
//! Local:    cargo bench --bench fit
//! Baseline: cargo bench --bench fit -- --save-baseline <name>
//! Compare:  critcmp <a> <b>          (see .github/workflows/bench.yml)

use std::time::Duration;

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion,
};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use tsl::grid_tensor::{
    fit,
    params::{GridTensorParamsBuilder, SplitStrategyParamsBuilder},
};

// Workload knobs. Fixed on purpose: comparability across commits matters more
// than exercising the full dataset. The full file is 20_640 rows.
const DATA_PATH: &str = "data/housing_full.csv";
const ROWS: usize = 20_640;
const N_ITER: usize = 40;
const SUBSAMPLE_SEED: u64 = 0;
const FIT_SEED: u64 = 42;

/// Load a headerless CSV with the response in the last column.
fn load_csv(path: &str) -> (Array2<f64>, Array1<f64>) {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for rec in rdr.records() {
        let r = rec.unwrap();
        let mut row = Vec::with_capacity(r.len() - 1);
        for i in 0..r.len() - 1 {
            row.push(r[i].parse::<f64>().unwrap());
        }
        ys.push(r[r.len() - 1].parse::<f64>().unwrap());
        rows.push(row);
    }
    let n = rows.len();
    let d = rows[0].len();
    let mut x = Array2::<f64>::zeros((n, d));
    for (i, row) in rows.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            x[[i, j]] = *v;
        }
    }
    (x, Array1::from(ys))
}

/// Deterministic subsample of `rows` rows (capped at the dataset size).
fn subsample(x: &Array2<f64>, y: &Array1<f64>, rows: usize) -> (Array2<f64>, Array1<f64>) {
    let n = x.nrows();
    let rows = rows.min(n);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(&mut StdRng::seed_from_u64(SUBSAMPLE_SEED));
    idx.truncate(rows);

    let d = x.ncols();
    let mut xs = Array2::<f64>::zeros((rows, d));
    let mut ys = Array1::<f64>::zeros(rows);
    for (new_i, &old_i) in idx.iter().enumerate() {
        for j in 0..d {
            xs[[new_i, j]] = x[[old_i, j]];
        }
        ys[new_i] = y[old_i];
    }
    (xs, ys)
}

fn bench_fit(c: &mut Criterion) {
    let (x_full, y_full) = load_csv(DATA_PATH);
    let (x, y) = subsample(&x_full, &y_full, ROWS);

    let mut group = c.benchmark_group("fit");
    // Each fit is ~100 ms, so 30 samples over a 15 s window gives critcmp a
    // tight enough interval to call regressions without making CI crawl.
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(15));

    for (label, max_bins) in [("exact", None), ("binned255", Some(255u16))] {
        let params = GridTensorParamsBuilder::new()
            .n_iter(N_ITER)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .best_split()
                    .min_interval_samples(1)
                    .build(),
            )
            .max_bins(max_bins)
            .build();

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            // Fresh seeded RNG per iteration → identical work every time, so
            // the measurement reflects compute, not RNG-driven path variance.
            b.iter_batched(
                || StdRng::seed_from_u64(FIT_SEED),
                |mut rng| {
                    let out = fit(black_box(x.view()), black_box(y.view()), &params, &mut rng);
                    black_box(out)
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_fit);
criterion_main!(benches);

//! Minimal driver for profiling a TSL fit on California Housing.
//!
//! Usage:
//!   cargo build --profile profiling --example profile_housing
//!   samply record --save-only --unstable-presymbolicate \
//!     -o /tmp/profile_housing.json -- \
//!     target/profiling/examples/profile_housing data/housing_full.csv 50 30 255
//!
//! Args (all optional):
//!   1) csv path           (default: data/housing_full.csv)
//!   2) n_iter             (default: 50)
//!   3) reps               (default: 1)
//!   4) max_bins           (default: 0 = exact)
//!
//! Loads CSV with the response in the last column, then loops `reps` fits
//! using `BestSplit` and `min_interval_samples=1`. Prints per-rep wall time
//! and an average.

use csv::ReaderBuilder;
use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, SeedableRng};
use std::time::Instant;
use tsl::grid_tensor::{
    fit,
    params::{GridTensorParamsBuilder, SplitStrategyParamsBuilder},
};

fn load_csv(path: &str) -> (Array2<f64>, Array1<f64>) {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .expect("open csv");
    let mut xs: Vec<Vec<f64>> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for rec in rdr.records() {
        let r = rec.unwrap();
        let mut row = Vec::with_capacity(r.len() - 1);
        for i in 0..r.len() - 1 {
            row.push(r[i].parse::<f64>().unwrap());
        }
        ys.push(r[r.len() - 1].parse::<f64>().unwrap());
        xs.push(row);
    }
    let n = xs.len();
    let d = xs[0].len();
    let mut x = Array2::<f64>::zeros((n, d));
    for (i, row) in xs.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            x[[i, j]] = *v;
        }
    }
    (x, Array1::from(ys))
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/housing_full.csv".to_string());
    let n_iter: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let reps: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let max_bins_arg: u16 = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let max_bins = if max_bins_arg == 0 {
        None
    } else {
        Some(max_bins_arg)
    };

    let (x, y) = load_csv(&path);
    eprintln!(
        "loaded {} rows x {} cols, max_bins={:?}",
        x.nrows(),
        x.ncols(),
        max_bins
    );

    let params = GridTensorParamsBuilder::new()
        .n_iter(n_iter)
        .split_strategy(
            SplitStrategyParamsBuilder::new()
                .best_split()
                .min_interval_samples(1)
                .build(),
        )
        .max_bins(max_bins)
        .build();

    let mut total_ms = 0.0_f64;
    let mut last_err = 0.0;
    for r in 0..reps {
        let mut rng = StdRng::seed_from_u64(42 + r as u64);
        let t0 = Instant::now();
        let (fit_result, _model) = fit(x.view(), y.view(), &params, &mut rng);
        let dt = t0.elapsed();
        last_err = fit_result.err;
        total_ms += dt.as_secs_f64() * 1000.0;
        eprintln!(
            "rep {}: {:.1} ms, err={:.6}",
            r,
            dt.as_secs_f64() * 1000.0,
            fit_result.err
        );
    }
    eprintln!(
        "avg {:.1} ms over {} reps (err={:.6})",
        total_ms / reps as f64,
        reps,
        last_err
    );
}

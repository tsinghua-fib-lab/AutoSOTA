//! One-off binary: fits California Housing with seed=42 and max_bins=255,
//! then writes test-set predictions as raw little-endian f64 bytes to a
//! fixed path. The companion parity test reloads and asserts bit-for-bit
//! equality (`==`, not approx) on the new branch.
//!
//! Usage:
//!   cargo run --release --example capture_parity_baseline
//!
//! Output file:
//!   tests/data/parity_baseline_max_bins_255.bin   (n_test * 8 bytes)
//!
//! Layout: predictions[0..n_test] as native-endian f64. Header is
//! the first 8 bytes interpreted as u64 giving `n_test` so the loader
//! can sanity-check length.

use csv::ReaderBuilder;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::fs::File;
use std::io::Write;
use tsl::grid_tensor::{
    fit,
    params::{
        GridTensorParamsBuilder, RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder,
    },
};

fn setup_data_csv(path: &str) -> (Array2<f64>, Array1<f64>) {
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

fn split_80_20(
    x: &Array2<f64>,
    y: &Array1<f64>,
    seed: u64,
) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = y.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.as_mut_slice().shuffle(&mut rng);
    let split = (n as f64 * 0.8) as usize;
    let train_idx = &indices[..split];
    let test_idx = &indices[split..];
    (
        x.select(ndarray::Axis(0), train_idx),
        y.select(ndarray::Axis(0), train_idx),
        x.select(ndarray::Axis(0), test_idx),
        y.select(ndarray::Axis(0), test_idx),
    )
}

fn main() {
    let (x, y) = setup_data_csv("data/housing_full.csv");
    let (x_train, y_train, x_test, _) = split_80_20(&x, &y, 42);

    let params = GridTensorParamsBuilder::new()
        .n_iter(50)
        .split_strategy(
            SplitStrategyParamsBuilder::new()
                .random_split()
                .split_try(10)
                .colsample_bytree(1.0)
                .min_interval_samples(3)
                .min_split_loss(0.0)
                .build(),
        )
        .refinement_strategy(
            RefinementStrategyParamsBuilder::new()
                .l2()
                .alpha(0.01)
                .prior_sample_size(0.0)
                .build(),
        )
        .max_bins(Some(255))
        .build();

    let mut rng = StdRng::seed_from_u64(43);
    let (fit_result, model) = fit(x_train.view(), y_train.view(), &params, &mut rng);
    let preds = model.predict(x_test.view());

    eprintln!(
        "fit_err = {:.10}, n_test = {}, preds[0..3] = {:?}",
        fit_result.err,
        preds.len(),
        &preds.as_slice().unwrap()[0..3]
    );

    let path = "tests/data/parity_baseline_max_bins_255.bin";
    let mut f = File::create(path).expect("create file");
    let n_test = preds.len() as u64;
    f.write_all(&n_test.to_le_bytes()).unwrap();
    for v in preds.iter() {
        f.write_all(&v.to_bits().to_le_bytes()).unwrap();
    }
    eprintln!("wrote {} ({} bytes)", path, 8 + 8 * n_test);
}

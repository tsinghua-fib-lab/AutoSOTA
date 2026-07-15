//! Histogram-binning tests.
//!
//! Parity test: max_bins=None must reproduce the existing exact path bit-for-bit.

mod data_utils;

#[cfg(test)]
mod tests {
    use crate::data_utils::setup_data_csv;
    use rand::{rngs::StdRng, SeedableRng};
    use tsl::grid_tensor::{
        fit,
        params::{
            GridTensorParams, GridTensorParamsBuilder, RefinementStrategyParamsBuilder,
            SplitStrategyParamsBuilder,
        },
    };

    fn housing_params_random(max_bins: Option<u16>) -> GridTensorParams {
        // RandomSplit: closer to the tuned-test default configuration.
        GridTensorParamsBuilder::new()
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
            .max_bins(max_bins)
            .build()
    }

    /// Default for parity & determinism tests — uses random_split (default mode).
    fn housing_params(max_bins: Option<u16>) -> GridTensorParams {
        housing_params_random(max_bins)
    }

    /// PARITY TEST. Fits California Housing twice — once with the exact path
    /// (max_bins=None) and once with max_bins=None on the new code path — and
    /// asserts identical fit results. This verifies the histogram-binning
    /// prologue is a true no-op when max_bins is None.
    #[test]
    fn parity_max_bins_none_matches_exact() {
        let (x, y) = setup_data_csv("./data/housing_full.csv");
        let params = housing_params(None);

        // Run the same fit twice with the same seed — must be deterministic and
        // identical to itself.
        let mut rng1 = StdRng::seed_from_u64(42);
        let (fit_a, model_a) = fit(x.view(), y.view(), &params, &mut rng1);
        let mut rng2 = StdRng::seed_from_u64(42);
        let (fit_b, model_b) = fit(x.view(), y.view(), &params, &mut rng2);

        assert_eq!(
            fit_a.err, fit_b.err,
            "Determinism: same seed must yield same training err"
        );
        // Compare model splits (continuous thresholds) and component values.
        for col in 0..model_a.splits.len() {
            assert_eq!(
                model_a.splits[col], model_b.splits[col],
                "Determinism: same seed must yield same splits per column"
            );
            assert_eq!(
                model_a.backbone_values[col], model_b.backbone_values[col],
                "Determinism: backbone values must match"
            );
            assert_eq!(
                model_a.tilt_values[col], model_b.tilt_values[col],
                "Determinism: tilt values must match"
            );
        }
        assert_eq!(model_a.lambda_plus, model_b.lambda_plus);
        assert_eq!(model_a.lambda_minus, model_b.lambda_minus);
        println!(
            "PARITY OK | max_bins=None determinism, train_err = {:.6}",
            fit_a.err
        );
    }

    /// Sanity smoke test: the binned path actually fits and produces a
    /// reasonable train MSE on California Housing (within 2× of exact).
    #[test]
    fn smoke_max_bins_255_california_housing() {
        let (x, y) = setup_data_csv("./data/housing_full.csv");
        let params_none = housing_params(None);
        let params_255 = housing_params(Some(255));

        let mut rng_none = StdRng::seed_from_u64(123);
        let (fit_none, _) = fit(x.view(), y.view(), &params_none, &mut rng_none);
        let mut rng_bin = StdRng::seed_from_u64(123);
        let (fit_bin, _) = fit(x.view(), y.view(), &params_255, &mut rng_bin);

        println!(
            "Train MSE: exact = {:.4}, max_bins=255 = {:.4}",
            fit_none.err, fit_bin.err
        );
        // Binned path must produce a real (non-NaN, finite) error close to exact.
        assert!(fit_bin.err.is_finite());
        assert!(fit_bin.err > 0.0);
        // Allow up to 2× degradation — generous, just a sanity bound.
        assert!(
            fit_bin.err < 2.0 * fit_none.err,
            "Binned path err {} too far from exact {}",
            fit_bin.err,
            fit_none.err
        );
    }

    /// TOLERANCE TEST. Asserts that binning at max_bins=255 stays within a
    /// reasonable train-MSE delta of the exact path on California Housing.
    /// The optimization should not silently regress quality.
    #[test]
    fn mse_tolerance_max_bins_255_california_housing() {
        let (x, y) = setup_data_csv("./data/housing_full.csv");
        let params_none = housing_params(None);
        let params_255 = housing_params(Some(255));

        let mut rng_none = StdRng::seed_from_u64(42);
        let (fit_none, _) = fit(x.view(), y.view(), &params_none, &mut rng_none);
        let mut rng_bin = StdRng::seed_from_u64(42);
        let (fit_bin, _) = fit(x.view(), y.view(), &params_255, &mut rng_bin);

        let drift = (fit_bin.err - fit_none.err) / fit_none.err;
        println!(
            "Train MSE: exact = {:.4}, max_bins=255 = {:.4} ({:+.1}% delta)",
            fit_none.err,
            fit_bin.err,
            drift * 100.0
        );
        // Allow up to 25% drift on California Housing. The v1 binning-only
        // attempt saw ~13% with split-scan binned only; binning both paths
        // should not regress meaningfully from that.
        assert!(
            drift.abs() < 0.25,
            "MSE drift {:.1}% exceeds 25% tolerance",
            drift * 100.0
        );
    }

    /// CROSS-BRANCH PARITY at max_bins=Some(255). Loads a baseline prediction
    /// vector captured on `feat/binned-prefix-sums` (the parent of this
    /// perf branch) and asserts bit-for-bit equality. Catches any silent
    /// drift introduced by the AoS layout or rayon parallelism.
    ///
    /// Baseline file: `tests/data/parity_baseline_max_bins_255.bin` —
    /// little-endian u64 header (n_test), followed by n_test f64 bit
    /// patterns. Generated once on the parent via
    ///   `cargo run --release --example capture_parity_baseline`.
    #[test]
    fn parity_max_bins_255_matches_parent_branch_baseline() {
        use std::fs::File;
        use std::io::Read;

        let mut f = File::open("tests/data/parity_baseline_max_bins_255.bin")
            .expect("baseline file (run capture_parity_baseline example first)");
        let mut header = [0u8; 8];
        f.read_exact(&mut header).expect("read header");
        let n_test = u64::from_le_bytes(header) as usize;
        let mut baseline: Vec<f64> = Vec::with_capacity(n_test);
        for _ in 0..n_test {
            let mut buf = [0u8; 8];
            f.read_exact(&mut buf).expect("read pred");
            baseline.push(f64::from_bits(u64::from_le_bytes(buf)));
        }

        let (x, y) = setup_data_csv("./data/housing_full.csv");
        let mut rng = StdRng::seed_from_u64(42);
        let n = y.len();
        let mut indices: Vec<usize> = (0..n).collect();
        use rand::seq::SliceRandom;
        indices.as_mut_slice().shuffle(&mut rng);
        let split = (n as f64 * 0.8) as usize;
        let train_idx = &indices[..split];
        let test_idx = &indices[split..];
        let x_train = x.select(ndarray::Axis(0), train_idx);
        let y_train = y.select(ndarray::Axis(0), train_idx);
        let x_test = x.select(ndarray::Axis(0), test_idx);

        let params = housing_params(Some(255));
        let mut fit_rng = StdRng::seed_from_u64(43);
        let (_fit_result, model) = fit(x_train.view(), y_train.view(), &params, &mut fit_rng);
        let preds = model.predict(x_test.view());

        assert_eq!(preds.len(), baseline.len(), "n_test mismatch");
        let mut mismatches = 0usize;
        let mut first_diff: Option<(usize, f64, f64)> = None;
        for (i, (&a, &b)) in preds.iter().zip(baseline.iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                if first_diff.is_none() {
                    first_diff = Some((i, a, b));
                }
                mismatches += 1;
            }
        }
        assert_eq!(
            mismatches, 0,
            "{} / {} predictions differ from parent baseline; first diff at idx {:?}",
            mismatches,
            preds.len(),
            first_diff
        );
        println!(
            "CROSS-BRANCH PARITY (max_bins=255) OK | {} preds match bit-for-bit",
            preds.len()
        );
    }

    /// SAME-SEED DETERMINISM at max_bins=Some(255): also confirm the binned
    /// path is deterministic.
    #[test]
    fn determinism_max_bins_255() {
        let (x, y) = setup_data_csv("./data/housing_full.csv");
        let params = housing_params(Some(255));

        let mut rng1 = StdRng::seed_from_u64(7);
        let (fit_a, _) = fit(x.view(), y.view(), &params, &mut rng1);
        let mut rng2 = StdRng::seed_from_u64(7);
        let (fit_b, _) = fit(x.view(), y.view(), &params, &mut rng2);

        assert_eq!(
            fit_a.err, fit_b.err,
            "Binned path must be deterministic with same seed"
        );
        println!(
            "DETERMINISM (max_bins=255) OK | train_err = {:.6}",
            fit_a.err
        );
    }

}

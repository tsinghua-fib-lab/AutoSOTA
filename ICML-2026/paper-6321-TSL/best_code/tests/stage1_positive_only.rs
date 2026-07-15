mod data_utils;

#[cfg(test)]
mod tests {
    // Note: data_utils not needed for these synthetic tests
    use tsl::{
        grid_tensor::{
            fit,
            params::{
                RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder, GridTensorParamsBuilder,
            },
        },
    };
    use ndarray::{Array1, Array2};
    use rand::{rngs::StdRng, SeedableRng};

    /// Test Stage 1 positive-only mode with all nonnegative data
    #[test]
    fn test_stage1_positive_only_detection() {
        // Create synthetic data with all nonnegative responses
        let n = 50;
        let p = 2;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut y = Array1::<f64>::zeros(n);

        // Simple pattern: y = x[0] + x[1] + noise (all positive)
        for i in 0..n {
            x[[i, 0]] = (i as f64) / (n as f64);
            x[[i, 1]] = ((i * 2) as f64) / (n as f64);
            y[i] = x[[i, 0]] + x[[i, 1]] + 0.1; // Add 0.1 to ensure all positive
        }

        let params = GridTensorParamsBuilder::new()
            .n_iter(5)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .best_split()
                    .min_interval_samples(2)
                    .build(),
            )
            .refinement_strategy(
                RefinementStrategyParamsBuilder::new()
                    .l2()
                    .alpha(0.01)
                    .build(),
            )
            .build();

        let mut rng = StdRng::seed_from_u64(42);
        let (fit_result, model) = fit(x.view(), y.view(), &params, &mut rng);

        // Verify predictions are nonnegative (Stage 1 invariant I23)
        let preds = model.predict(x.view());
        for (i, &pred) in preds.iter().enumerate() {
            assert!(
                pred >= -1e-10,
                "I23 violation: prediction[{}] = {} should be >= 0 in Stage 1 mode",
                i,
                pred
            );
        }

        // Verify error is reasonable
        let residuals = &y - &preds;
        let computed_err = residuals.mapv(|r| r * r).mean().unwrap();
        println!(
            "Stage 1 test - Computed error: {}, Fit error: {}",
            computed_err, fit_result.err
        );
        assert!(
            (computed_err - fit_result.err).abs() < 1e-8,
            "Error mismatch: computed={}, fit_result={}",
            computed_err,
            fit_result.err
        );
    }

    /// Test that Stage 1 mode is correctly detected when all residuals are nonnegative
    #[test]
    fn test_stage1_initialization_invariants() {
        // Create data with all nonnegative responses
        let n = 20;
        let p = 2;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut y = Array1::<f64>::zeros(n);

        for i in 0..n {
            x[[i, 0]] = (i as f64) / (n as f64);
            x[[i, 1]] = ((i * 2) as f64) / (n as f64);
            y[i] = 1.0 + x[[i, 0]]; // All positive
        }

        let params = GridTensorParamsBuilder::new()
            .n_iter(1) // Just one iteration to check initialization
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .best_split()
                    .min_interval_samples(2)
                    .build(),
            )
            .refinement_strategy(
                RefinementStrategyParamsBuilder::new()
                    .l2()
                    .alpha(0.01)
                    .build(),
            )
            .build();

        let mut rng = StdRng::seed_from_u64(42);
        let (fit_result, _) = fit(x.view(), y.view(), &params, &mut rng);

        // Check that predictions are nonnegative
        // (This indirectly verifies Stage 1 mode since f = f_+ - f_- and f_- should be 0)
        assert!(
            fit_result.y_hat.iter().all(|&y_hat| y_hat >= -1e-10),
            "All predictions should be nonnegative in Stage 1 mode"
        );
    }

    /// Test that mixed-sign data does NOT use Stage 1 mode
    #[test]
    fn test_full_two_tensor_with_mixed_signs() {
        // Create data with mixed signs
        let n = 30;
        let p = 2;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut y = Array1::<f64>::zeros(n);

        for i in 0..n {
            x[[i, 0]] = (i as f64) / (n as f64);
            x[[i, 1]] = ((i * 2) as f64) / (n as f64);
            // Mix positive and negative
            y[i] = if i % 2 == 0 {
                1.0 + x[[i, 0]]
            } else {
                -1.0 - x[[i, 0]]
            };
        }

        let params = GridTensorParamsBuilder::new()
            .n_iter(3)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .best_split()
                    .min_interval_samples(2)
                    .build(),
            )
            .refinement_strategy(
                RefinementStrategyParamsBuilder::new()
                    .l2()
                    .alpha(0.01)
                    .build(),
            )
            .build();

        let mut rng = StdRng::seed_from_u64(42);
        let (fit_result, model) = fit(x.view(), y.view(), &params, &mut rng);

        // With mixed signs, predictions can be negative (full two-tensor mode)
        let preds = model.predict(x.view());
        println!(
            "Mixed-sign test - Predictions range: [{}, {}]",
            preds.iter().cloned().fold(f64::INFINITY, f64::min),
            preds.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        );

        // Verify error is reasonable
        let residuals = &y - &preds;
        let computed_err = residuals.mapv(|r| r * r).mean().unwrap();
        assert!(
            (computed_err - fit_result.err).abs() < 1e-8,
            "Error mismatch: computed={}, fit_result={}",
            computed_err,
            fit_result.err
        );
    }
}

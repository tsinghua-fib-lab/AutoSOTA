mod data_utils;

#[cfg(test)]
mod reproducibility_tests {
    use crate::data_utils::setup_data_csv;

    use tsl::forest::{fit_boosted, params::TSLBoostedParamsBuilder};

    use std::ops::Div;

    #[test]
    fn test_tsl_boosted_reproducibility() {
        let (x, y) = setup_data_csv("./data/2covars.csv");

        // Use builder pattern for cleaner parameter construction
        let params = TSLBoostedParamsBuilder::new()
            .epochs(10)
            .n_trees(5)
            .n_iter(25) // Using default, but explicitly stated for clarity
            .seed(42)
            .build();

        // Train two models with the same seed
        let (_, model1) = fit_boosted(x.view(), y.view(), &params);
        let (_, model2) = fit_boosted(x.view(), y.view(), &params);

        // Generate predictions
        let pred1 = model1.predict(x.view());
        let pred2 = model2.predict(x.view());

        // Check predictions are identical
        let diff = &pred1 - &pred2;
        assert!(
            diff.iter().all(|&x| x.abs() < 1e-10),
            "Models with same seed produced different predictions"
        );
    }

    #[test]
    fn test_tsl_boosted_different_seeds() {
        let (x, y) = setup_data_csv("./data/2covars.csv");

        // Use builder pattern for cleaner parameter construction
        let params1 = TSLBoostedParamsBuilder::new()
            .epochs(2)
            .n_trees(5)
            .n_iter(25) // Using default, but explicitly stated for clarity
            .seed(42)
            .build();

        // Train models with different seeds
        let (_, model1) = fit_boosted(x.view(), y.view(), &params1);

        let params2 = TSLBoostedParamsBuilder::new()
            .epochs(2)
            .n_trees(5)
            .n_iter(25)
            .seed(43) // Different seed
            .build();

        let (_, model2) = fit_boosted(x.view(), y.view(), &params2);

        // Generate predictions
        let pred1 = model1.predict(x.view());
        let pred2 = model2.predict(x.view());

        // Check predictions are different
        let diff = &pred1 - &pred2;
        assert!(
            diff.iter().any(|&x| x.abs() > 1e-10),
            "Models with different seeds produced identical predictions"
        );
    }

    #[test]
    fn test_fit_result_error_is_y_minus_sum_preds() {
        let (x, y) = setup_data_csv("./data/2covars.csv");
        let params = TSLBoostedParamsBuilder::new()
            .epochs(10)
            .n_trees(10)
            .n_iter(10)
            .seed(42)
            .build();
        let (fit_result, model) = fit_boosted(x.view(), y.view(), &params);
        let preds = model.predict(x.view());
        let err = y
            .view()
            .iter()
            .zip(preds.iter())
            .map(|(y, p)| (y - p).powi(2))
            .sum::<f64>()
            .div(y.len() as f64);

        assert!((fit_result.err - err).abs() < 1e-15);
    }
}

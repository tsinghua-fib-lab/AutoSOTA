//! Tuned-baseline TSL regression tests.
//!
//! Each test below uses the best hyperparameters from an internal
//! hyperparameter sweep and asserts that the test MSE on a 50/50 random
//! split (seed=1) stays within 20% of the recorded baseline test_mse.
//! Drift past that bound is a regression alert.
//!
//! Note: the recorded baseline was measured on the OpenML task's official
//! train/test split; here we use a 50/50 shuffle of the same CSV. The 20%
//! band is loose enough to absorb that variance on small datasets.

mod data_utils;

#[cfg(test)]
mod tests {
    use crate::data_utils::setup_data_csv;
    use ndarray::{Array1, Array2};
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use tsl::{
        forest::{fit_boosted, params::TSLBoostedParamsBuilder},
        grid_tensor::params::{RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder},

    };

    fn split_5050(
        x: &Array2<f64>,
        y: &Array1<f64>,
        seed: u64,
    ) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = y.len();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.as_mut_slice().shuffle(&mut rng);
        let split = n / 2;
        let train_idx = &idx[..split];
        let test_idx = &idx[split..];
        (
            x.select(ndarray::Axis(0), train_idx),
            y.select(ndarray::Axis(0), train_idx),
            x.select(ndarray::Axis(0), test_idx),
            y.select(ndarray::Axis(0), test_idx),
        )
    }

    fn report(label: &str, base: f64, train: f64, test: f64, baseline: f64) {
        println!(
            "{label} | base_err={base:.6} train_err={train:.6} \
             test_err={test:.6} tuned_baseline={baseline:.6} \
             ratio={:.3}",
            test / baseline
        );
    }

    /// Best hyperparameters from an internal sweep; baseline test_mse ≈ 0.3856.
    #[test]
    fn test_tsl_red_wine_tuned() {
        let tuned_baseline = 0.3855837447017473;
        let (x, y) = setup_data_csv("data/44972_red_wine.csv");
        let (x_train, y_train, x_test, y_test) = split_5050(&x, &y, 1);

        let params = TSLBoostedParamsBuilder::new()
            .epochs(1)
            .n_trees(200)
            .n_iter(58)
            .decay(0.33644475374899324)
            .similarity_threshold(0.06998406165377283)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .random_split()
                    .split_try(2)
                    .colsample_bytree(0.786262699456606)
                    .min_interval_samples(20)
                    .min_split_loss(0.0)
                    .complexity_penalty(0.0)
                    .build(),
            )
            .refinement_strategy(
                RefinementStrategyParamsBuilder::new()
                    .huber()
                    .alpha(2.7251225043453166e-06)
                    .prior_sample_size(0.0)
                    .update_clamp(24.834065493203312)
                    .tilt_rho(0.0)
                    .tilt_tau(0.0)
                    .build(),
            )


            .seed(42)
            .build();

        let (fit_result, model) = fit_boosted(x_train.view(), y_train.view(), &params);
        let preds = model.predict(x_test.view());
        let mean = y_test.mean().unwrap();
        let base_err = y_test.mapv(|v| (v - mean).powi(2)).mean().unwrap();
        let test_err = (&y_test - &preds).mapv(|d| d * d).mean().unwrap();

        report("RedWine", base_err, fit_result.err, test_err, tuned_baseline);
        assert!(
            test_err < tuned_baseline * 1.20,
            "test MSE {test_err:.6} exceeds 1.20× tuned baseline {tuned_baseline:.6}"
        );
    }

    /// Best hyperparameters from an internal sweep; baseline test_mse ≈ 1.3600.
    #[test]
    fn test_tsl_fish_toxicity_tuned() {
        let tuned_baseline = 1.3600211553047568;
        let (x, y) = setup_data_csv("data/44970_QSAR_fish_toxicity.csv");
        let (x_train, y_train, x_test, y_test) = split_5050(&x, &y, 1);

        let params = TSLBoostedParamsBuilder::new()
            .epochs(2)
            .n_trees(200)
            .n_iter(40)
            .decay(0.8104989030334956)
            .similarity_threshold(0.22758238420842475)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .random_split()
                    .split_try(16)
                    .colsample_bytree(0.9831750502366127)
                    .min_interval_samples(4)
                    .min_split_loss(0.0)
                    .complexity_penalty(0.0)
                    .build(),
            )
            .refinement_strategy(
                RefinementStrategyParamsBuilder::new()
                    .huber()
                    .alpha(0.00010894353213340127)
                    .prior_sample_size(0.0)
                    .update_clamp(5.938972779183364)
                    .tilt_rho(0.0)
                    .tilt_tau(0.0)
                    .build(),
            )


            .seed(42)
            .build();

        let (fit_result, model) = fit_boosted(x_train.view(), y_train.view(), &params);
        let preds = model.predict(x_test.view());
        let mean = y_test.mean().unwrap();
        let base_err = y_test.mapv(|v| (v - mean).powi(2)).mean().unwrap();
        let test_err = (&y_test - &preds).mapv(|d| d * d).mean().unwrap();

        report("FishToxicity", base_err, fit_result.err, test_err, tuned_baseline);
        assert!(
            test_err < tuned_baseline * 1.20,
            "test MSE {test_err:.6} exceeds 1.20× tuned baseline {tuned_baseline:.6}"
        );
    }

    /// Best hyperparameters from an internal sweep; baseline test_mse ≈ 11864.57.
    /// Smallest dataset (517 rows), low n_iter — runs in a few seconds.
    #[test]
    fn test_tsl_forest_fires_tuned() {
        let tuned_baseline = 11864.567729381284;
        let (x, y) = setup_data_csv("data/44962_forest_fires.csv");
        let (x_train, y_train, x_test, y_test) = split_5050(&x, &y, 1);

        let params = TSLBoostedParamsBuilder::new()
            .epochs(2)
            .n_trees(200)
            .n_iter(12)
            .decay(0.3255417875916115)
            .similarity_threshold(0.5248023808967928)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .random_split()
                    .split_try(5)
                    .colsample_bytree(0.7901043727721213)
                    .min_interval_samples(59)
                    .min_split_loss(0.0)
                    .complexity_penalty(0.0)
                    .build(),
            )
            .refinement_strategy(
                RefinementStrategyParamsBuilder::new()
                    .huber()
                    .alpha(6.85248630061216e-05)
                    .prior_sample_size(0.0)
                    .update_clamp(3.243045328783296)
                    .tilt_rho(0.0)
                    .tilt_tau(0.0)
                    .build(),
            )


            .seed(42)
            .build();

        let (fit_result, model) = fit_boosted(x_train.view(), y_train.view(), &params);
        let preds = model.predict(x_test.view());
        let mean = y_test.mean().unwrap();
        let base_err = y_test.mapv(|v| (v - mean).powi(2)).mean().unwrap();
        let test_err = (&y_test - &preds).mapv(|d| d * d).mean().unwrap();

        report("ForestFires", base_err, fit_result.err, test_err, tuned_baseline);
        assert!(
            test_err < tuned_baseline * 1.20,
            "test MSE {test_err:.6} exceeds 1.20× tuned baseline {tuned_baseline:.6}"
        );
    }

    /// Best hyperparameters from an internal sweep; baseline test_mse ≈ 0.3442.
    /// L2 refinement (other tuned configs use Huber). n_iter=199 → slowest of the four.
    #[test]
    fn test_tsl_energy_efficiency_tuned() {
        let tuned_baseline = 0.3442055325051992;
        let (x, y) = setup_data_csv("data/44960_energy_efficiency.csv");
        let (x_train, y_train, x_test, y_test) = split_5050(&x, &y, 1);

        let params = TSLBoostedParamsBuilder::new()
            .epochs(2)
            .n_trees(200)
            .n_iter(199)
            .decay(0.8728065551269725)
            .similarity_threshold(0.6741675605988402)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .random_split()
                    .split_try(18)
                    .colsample_bytree(0.3000174855114284)
                    .min_interval_samples(3)
                    .min_split_loss(0.0)
                    .complexity_penalty(0.0)
                    .build(),
            )
            .refinement_strategy(
                RefinementStrategyParamsBuilder::new()
                    .l2()
                    .alpha(0.03774411956217899)
                    .prior_sample_size(0.0)
                    .update_clamp(30.54915446269604)
                    .tilt_rho(0.0)
                    .tilt_tau(0.0)
                    .build(),
            )


            .seed(42)
            .build();

        let (fit_result, model) = fit_boosted(x_train.view(), y_train.view(), &params);
        let preds = model.predict(x_test.view());
        let mean = y_test.mean().unwrap();
        let base_err = y_test.mapv(|v| (v - mean).powi(2)).mean().unwrap();
        let test_err = (&y_test - &preds).mapv(|d| d * d).mean().unwrap();

        report("EnergyEff", base_err, fit_result.err, test_err, tuned_baseline);
        assert!(
            test_err < tuned_baseline * 1.20,
            "test MSE {test_err:.6} exceeds 1.20× tuned baseline {tuned_baseline:.6}"
        );
    }
}

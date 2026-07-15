mod data_utils;

#[cfg(feature = "evo-logging")]
mod evo_logging_tests {

    use crate::data_utils::{setup_data_csv, split_data};
    use tsl::{
        forest::{fit_boosted, params::TSLBoostedParamsBuilder},
        grid_tensor::params::{RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder},
        stage_predictor::Aggregation,
    };

    #[test]
    fn test_socmob() {
        let (x, y) = setup_data_csv("./data/44987_socmob.csv");
        let (x_train, y_train, x_test, y_test) = split_data(&x, &y);

        let params = TSLBoostedParamsBuilder::new()
            .epochs(1)
            .n_trees(200)
            .n_iter(92)
            .decay(0.9882439500155061)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .random_split()
                    .split_try(3)
                    .colsample_bytree(0.8828809775150922)
                    .min_interval_samples(1)
                    .min_split_loss(0.0)
                    .build(),
            )
            .refinement_strategy(
                RefinementStrategyParamsBuilder::new()
                    .l2()
                    .alpha(0.0)
                    .update_clamp(3.4919359535659664)
                    .build(),
            )
            .similarity_threshold(0.36343748787856406)


            .seed(42)
            .visualdb_path(Some("target/socmob_splits.sqlite".to_string()))
            .build();

        let (fit_result, model) = fit_boosted(x_train.view(), y_train.view(), &params);
        let preds = model.predict(x_test.view());
        let mean = y_test.mean().unwrap();
        let base_err = y_test.mapv(|v| (v - mean).powi(2)).mean().unwrap();
        let test_err = (y_test - preds).powi(2).mean().unwrap();

        println!(
            "SocMob | Base: {:.4}, Train: {:.4}, Test: {:.4}",
            base_err, fit_result.err, test_err
        );
        assert!(test_err < base_err);
    }

    #[test]
    fn test_tsl_auction() {
        let (x, y) = setup_data_csv("data/44958_auction_verification.csv");
        let (x_train, y_train, x_test, y_test) = split_data(&x, &y);

        let params = TSLBoostedParamsBuilder::new()
            .epochs(2)
            .n_trees(200)
            .n_iter(57)

            .aggregation_method(Aggregation::Combined)
            .refinement_strategy(
                RefinementStrategyParamsBuilder::new()
                    .l2()
                    .alpha(0.0008761191731033688)
                    .prior_sample_size(0.0)
                    .update_clamp(3.1388455603935297)
                    .build(),
            )

            .similarity_threshold(0.3573991614563707)
            .decay(0.8418796974151387)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .random_split()
                    .split_try(20)
                    .colsample_bytree(0.631161799847988)
                    .min_interval_samples(6)
                    .min_split_loss(0.0)
                    .build(),
            )
            .visualdb_path(Some("target/auction_splits.sqlite".to_string()))
            .build();

        let (fit_result, model) = fit_boosted(x_train.view(), y_train.view(), &params);
        let preds = model.predict(x_test.view());
        let mean = y_test.mean().unwrap();
        let base_err = y_test.mapv(|v| (v - mean).powi(2)).mean().unwrap();
        let test_err = (y_test - preds).powi(2).mean().unwrap();

        println!(
            "Auction | Base: {:.4}, Train: {:.4}, Test: {:.4}",
            base_err, fit_result.err, test_err
        );
        assert!(test_err < base_err);
    }

    #[test]
    fn test_tsl_synthetic() {
        let (x, y) = setup_data_csv("./data/data_gen3.csv");
        let (x_train, y_train, x_test, y_test) = split_data(&x, &y);

        let params = TSLBoostedParamsBuilder::new()
            .epochs(3)
            .n_iter(20)
            .n_trees(389)

            .refinement_strategy(
                RefinementStrategyParamsBuilder::new()
                    .l2()
                    .alpha(0.41267364726662886)
                    .tilt_tau(5.558658765668141e-05)
                    .tilt_rho(0.00021010963348998642)
                    .build(),
            )
            .similarity_threshold(0.9)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .random_split()
                    .split_try(2)
                    .colsample_bytree(0.39588592353906327)
                    .min_interval_samples(38)
                    .min_split_loss(0.7540167160231375)
                    .build(),
            )
            .log_level("info")
            .visualdb_path(Some("target/synthetic_data_splits.sqlite".to_string()))
            .build();

        let (fit_result, model) = fit_boosted(x_train.view(), y_train.view(), &params);
        let preds = model.predict(x_test.view());
        let mean = y_test.mean().unwrap();
        let base_err = y_test.mapv(|v| (v - mean).powi(2)).mean().unwrap();
        let test_err = (y_test - preds).powi(2).mean().unwrap();

        println!(
            "Synthetic | Base: {:.4}, Train: {:.4}, Test: {:.4}",
            base_err, fit_result.err, test_err
        );
        assert!(test_err < base_err);
    }

    #[test]
    fn test_tsl_housing() {
        let (x, y) = setup_data_csv("./data/housing_full.csv");
        let (x_train, y_train, x_test, y_test) = split_data(&x, &y);

        let params = TSLBoostedParamsBuilder::new()
            .epochs(7)
            .n_iter(150)
            .n_trees(200)
            .aggregation_method(Aggregation::Combined)
            .refinement_strategy(
                RefinementStrategyParamsBuilder::new()
                    .l2()
                    .alpha(0.01)
                    .build(),
            )

            .similarity_threshold(0.1)
            .decay(0.9)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .random_split()
                    .split_try(10)
                    .colsample_bytree(1.0)
                    .min_interval_samples(20)
                    .min_split_loss(0.9)
                    .build(),
            )
            .visualdb_path(Some("target/housing_splits.sqlite".to_string()))
            .build();

        let (fit_result, model) = fit_boosted(x_train.view(), y_train.view(), &params);
        let preds = model.predict(x_test.view());
        let mean = y_test.mean().unwrap();
        let base_err = y_test.mapv(|v| (v - mean).powi(2)).mean().unwrap();
        let test_err = (y_test - preds).powi(2).mean().unwrap();

        println!(
            "Housing | Base: {:.4}, Train: {:.4}, Test: {:.4}",
            base_err, fit_result.err, test_err
        );
        assert!(test_err < base_err);
    }

    #[test]
    fn test_tsl_red_wine() {
        let (x, y) = setup_data_csv("data/red_wine_processed.csv");
        let (x_train, y_train, x_test, y_test) = split_data(&x, &y);

        let params = TSLBoostedParamsBuilder::new()
            .epochs(10)
            .n_trees(203)
            .n_iter(105)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .random_split()
                    .split_try(3)
                    .colsample_bytree(0.652015893646319)
                    .min_interval_samples(43)
                    .min_split_loss(0.060839473717718556)
                    .build(),
            )
            .refinement_strategy(
                RefinementStrategyParamsBuilder::new()
                    .l2()
                    .alpha(2.196_807_874_090_684_2e-9)
                    .prior_sample_size(0.0)
                    .build(),
            )
            .similarity_threshold(0.0)


            .log_level("info")
            .seed(1)
            .visualdb_path(Some("target/red_wine_splits.sqlite".to_string()))
            .build();

        let (fit_result, model) = fit_boosted(x_train.view(), y_train.view(), &params);
        let preds = model.predict(x_test.view());
        let mean = y_test.mean().unwrap();
        let base_err = y_test.mapv(|v| (v - mean).powi(2)).mean().unwrap();
        let test_err = (y_test - preds).powi(2).mean().unwrap();

        println!(
            "Red Wine | Base: {:.4}, Train: {:.4}, Test: {:.4}",
            base_err, fit_result.err, test_err
        );
        assert!(test_err < base_err);
    }

    #[test]
    fn test_tsl_fish_toxicity() {
        let (x, y) = setup_data_csv("data/fish_toxicity.csv");
        let (x_train, y_train, x_test, y_test) = split_data(&x, &y);

        let params = TSLBoostedParamsBuilder::new()
            .epochs(9)
            .n_trees(121)
            .n_iter(15)
            .decay(0.9301908910375473)
            .similarity_threshold(0.7027072768106223)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .random_split()
                    .split_try(16)
                    .colsample_bytree(0.7030641276642211)
                    .min_interval_samples(5)
                    .min_split_loss(0.3530290549534714)
                    .build(),
            )


            .seed(100)
            .visualdb_path(Some("target/fish_toxicity_splits.sqlite".to_string()))
            .build();

        let (fit_result, model) = fit_boosted(x_train.view(), y_train.view(), &params);
        let preds = model.predict(x_test.view());
        let mean = y_test.mean().unwrap();
        let base_err = y_test.mapv(|v| (v - mean).powi(2)).mean().unwrap();
        let test_err = (y_test - preds).powi(2).mean().unwrap();

        println!(
            "Fish Toxicity | Base: {:.4}, Train: {:.4}, Test: {:.4}",
            base_err, fit_result.err, test_err
        );
        assert!(test_err < base_err);
    }
}

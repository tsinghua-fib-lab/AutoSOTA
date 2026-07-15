mod data_utils;

#[cfg(test)]
mod tests {
    use crate::data_utils::{setup_data_csv, split_data};
    
    use tsl::{
        stage_predictor::Aggregation,
        forest::{fit_boosted, params::TSLBoostedParamsBuilder, TSL},
    };
    use std::fs;

    #[test]
    fn test_tsl_save_and_load() {
        // Setup data
        let (x, y) = setup_data_csv("./data/2covars.csv");
        let (x_train, y_train, x_test, _y_test) = split_data(&x, &y);

        // Fit a model
        let params = TSLBoostedParamsBuilder::new()
            .epochs(3)
            .n_trees(5)
            .n_iter(10)
            .seed(42)
            .aggregation_method(Aggregation::Combined)
            .build();

        let (_fit_result, original_model) = fit_boosted(x_train.view(), y_train.view(), &params);

        // Get predictions from original model
        let original_preds = original_model.predict(x_test.view());

        // Serialize and save to file using bincode
        let bytes =
            bincode::serialize(&original_model).expect("Failed to serialize TSL model to bincode");
        fs::write("target/data.bin", bytes).expect("Failed to write to target/data.bin");

        // Read and deserialize from file
        let bytes = fs::read("target/data.bin").expect("Failed to read from target/data.bin");
        let deserialized_model: TSL =
            bincode::deserialize(&bytes).expect("Failed to deserialize TSL model from bincode");

        // Get predictions from deserialized model
        let deserialized_preds = deserialized_model.predict(x_test.view());

        // Verify predictions are identical
        let pred_diff = &original_preds - &deserialized_preds;
        let max_diff = pred_diff.iter().map(|x| x.abs()).fold(0.0, f64::max);

        assert!(
            max_diff < 1e-10,
            "Predictions from deserialized model differ from original (max diff: {})",
            max_diff
        );
    }
}

use crate::{
    grid_tensor::params::{RefinementStrategyParams, SplitStrategyParams},
    stage_predictor::{
        params::{StagePredictorParams, StagePredictorParamsBuilder},
        Aggregation,
    },
};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct TSLBoostedParams {
    pub epochs: usize,
    pub decay: f64,
    pub sp_params: StagePredictorParams,
    pub seed: u64,
    pub log_level: String,
    pub visualdb_path: Option<String>,
}

#[derive(Debug)]
pub struct TSLBoostedParamsBuilder {
    epochs: usize,
    decay: f64,
    sp_params_builder: StagePredictorParamsBuilder,
    seed: u64,
    log_level: String,
    visualdb_path: Option<String>,
}

impl TSLBoostedParamsBuilder {
    pub fn new() -> Self {
        Self {
            epochs: 5,
            decay: 1.0,
            sp_params_builder: StagePredictorParamsBuilder::new(),
            seed: 42,
            log_level: "info".to_string(),
            visualdb_path: None,
        }
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn decay(mut self, decay: f64) -> Self {
        self.decay = decay;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn log_level(mut self, log_level: &str) -> Self {
        self.log_level = log_level.to_string();
        self
    }

    pub fn visualdb_path(mut self, visualdb_path: Option<String>) -> Self {
        self.visualdb_path = visualdb_path;
        self
    }

    pub fn n_trees(mut self, n_trees: usize) -> Self {
        self.sp_params_builder = self.sp_params_builder.n_trees(n_trees);
        self
    }

    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.sp_params_builder = self.sp_params_builder.n_iter(n_iter);
        self
    }

    pub fn aggregation_method(mut self, aggregation_method: Aggregation) -> Self {
        self.sp_params_builder = self.sp_params_builder.aggregation_method(aggregation_method);
        self
    }

    pub fn similarity_threshold(mut self, similarity_threshold: f64) -> Self {
        self.sp_params_builder = self.sp_params_builder.similarity_threshold(similarity_threshold);
        self
    }

    pub fn split_strategy(mut self, strategy: SplitStrategyParams) -> Self {
        self.sp_params_builder = self.sp_params_builder.split_strategy(strategy);
        self
    }

    pub fn refinement_strategy(mut self, strategy: RefinementStrategyParams) -> Self {
        self.sp_params_builder = self.sp_params_builder.refinement_strategy(strategy);
        self
    }

    pub fn max_bins(mut self, max_bins: Option<u16>) -> Self {
        self.sp_params_builder = self.sp_params_builder.max_bins(max_bins);
        self
    }

    pub fn build(self) -> TSLBoostedParams {
        TSLBoostedParams {
            epochs: self.epochs,
            decay: self.decay,
            sp_params: self.sp_params_builder.build(),
            seed: self.seed,
            log_level: self.log_level,
            visualdb_path: self.visualdb_path,
        }
    }
}

impl Default for TSLBoostedParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TSLBoostedParams {
    fn default() -> Self {
        TSLBoostedParamsBuilder::new().build()
    }
}

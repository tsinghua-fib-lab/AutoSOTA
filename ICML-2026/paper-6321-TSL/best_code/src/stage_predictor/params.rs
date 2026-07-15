use crate::{
    grid_tensor::params::{
        GridTensorParams, GridTensorParamsBuilder, RefinementStrategyParams, SplitStrategyParams,
    },
    stage_predictor::Aggregation,
};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct StagePredictorParams {
    pub n_trees: usize,
    pub tg_params: GridTensorParams,
    pub similarity_threshold: f64,
    pub aggregation_method: Aggregation,
}

#[derive(Debug)]
pub struct StagePredictorParamsBuilder {
    n_trees: usize,
    tg_params_builder: GridTensorParamsBuilder,
    similarity_threshold: f64,
    aggregation_method: Aggregation,
}

impl StagePredictorParamsBuilder {
    pub fn new() -> Self {
        Self {
            n_trees: 100,
            tg_params_builder: GridTensorParamsBuilder::new(),
            similarity_threshold: 0.0,
            aggregation_method: Aggregation::Combined,
        }
    }

    pub fn n_trees(mut self, n_trees: usize) -> Self {
        self.n_trees = n_trees;
        self
    }

    pub fn aggregation_method(mut self, aggregation_method: Aggregation) -> Self {
        self.aggregation_method = aggregation_method;
        self
    }

    pub fn similarity_threshold(mut self, similarity_threshold: f64) -> Self {
        self.similarity_threshold = similarity_threshold;
        self
    }

    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.tg_params_builder = self.tg_params_builder.n_iter(n_iter);
        self
    }

    pub fn split_strategy(mut self, strategy: SplitStrategyParams) -> Self {
        self.tg_params_builder = self.tg_params_builder.split_strategy(strategy);
        self
    }

    pub fn refinement_strategy(mut self, strategy: RefinementStrategyParams) -> Self {
        self.tg_params_builder = self.tg_params_builder.refinement_strategy(strategy);
        self
    }

    pub fn max_bins(mut self, max_bins: Option<u16>) -> Self {
        self.tg_params_builder = self.tg_params_builder.max_bins(max_bins);
        self
    }

    pub fn build(self) -> StagePredictorParams {
        StagePredictorParams {
            n_trees: self.n_trees,
            tg_params: self.tg_params_builder.build(),
            similarity_threshold: self.similarity_threshold,
            aggregation_method: self.aggregation_method,
        }
    }
}

impl Default for StagePredictorParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for StagePredictorParams {
    fn default() -> Self {
        StagePredictorParamsBuilder::new().build()
    }
}

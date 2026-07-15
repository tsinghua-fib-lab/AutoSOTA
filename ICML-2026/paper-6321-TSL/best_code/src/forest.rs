use ndarray::{Array1, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::stage_predictor::StagePredictor;

mod fitter;
pub mod params;
pub use fitter::{fit_boosted, fit_boosted_with_test_error};

#[derive(Debug, Serialize, Deserialize)]
pub struct TSL {
    stage_predictors: Vec<StagePredictor>,
}

impl TSL {
    pub fn get_stage_predictors(&self) -> &Vec<StagePredictor> {
        &self.stage_predictors
    }
}

impl TSL {
    pub const fn new(stage_predictors: Vec<StagePredictor>) -> Self {
        Self { stage_predictors }
    }
}

impl TSL {
    pub fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(x.shape()[0]);
        for stage_predictor in &self.stage_predictors {
            result += &stage_predictor.predict(x);
        }

        result
    }
}

use ndarray::Array1;

pub mod stage_predictor;
pub mod forest;
pub mod grid_tensor;
pub mod logging;

#[derive(Debug)]
pub struct FitResult {
    pub err: f64,
    pub residuals: Array1<f64>,
    pub y_hat: Array1<f64>,
}

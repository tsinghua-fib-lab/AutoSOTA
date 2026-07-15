use super::grid_tensor::GridTensor;
pub use fitter::fit_ensemble;
use ndarray::{Array1, ArrayView2};
use serde::{Deserialize, Serialize};
pub mod params;

mod aggregate_bagged;
mod combine_grids;
mod fitter;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagePredictor {
    pub grid_tensors: Vec<GridTensor>,
    pub primary_grid_tensor: GridTensor,
    pub candidate_indices: Option<Vec<usize>>,
    pub aggregation_method: Aggregation,
    pub scaling_plus: Option<f64>,
    pub scaling_minus: Option<f64>,
    pub energy: Option<f64>,
}

#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub enum Aggregation {
    Mean,
    GeometricMean,
    Combined,
}

impl Aggregation {
    fn predict(&self, x: ArrayView2<f64>, tgf: &StagePredictor) -> Array1<f64> {
        // Extract UNSCALED f+ and f- predictions from primary_grid_tensor (two-tensor fields)
        // IMPORTANT: Extract directly from two-tensor fields, do NOT use grid.predict()
        // which might apply scaling. We want raw f+ and f- values.
        let (f_plus, f_minus) = extract_two_tensor_predictions_unscaled(&tgf.primary_grid_tensor, x);

        // Apply scalings from OLS solution: scaling_plus * f_+ + scaling_minus * (-f_-)
        // Note: scaling_minus is the coefficient for the -f_- column in the design matrix
        // So we multiply by -f_minus (the column value) and scaling_minus (the coefficient)
        let scaling_plus = tgf.scaling_plus.unwrap_or(1.0);
        let scaling_minus = tgf.scaling_minus.unwrap_or(0.0);
        &f_plus * scaling_plus + &(-f_minus) * scaling_minus
    }

    fn predict_unscaled(&self, x: ArrayView2<f64>, tgf: &StagePredictor) -> Array1<f64> {
        let n = x.nrows();
        let n_grids = tgf.grid_tensors.len();
        match self {
            Self::Mean => {
                let mut preds = Array1::zeros(n);
                for grid_tensor in tgf.grid_tensors.iter() {
                    preds += &grid_tensor.predict(x);
                }
                preds / n_grids as f64
            }
            Self::GeometricMean => {
                let preds: Vec<Array1<f64>> =
                    tgf.grid_tensors.iter().map(|tg| tg.predict(x)).collect();
                let mut stacked_preds = Vec::with_capacity(n * n_grids);
                for i in 0..n {
                    for pred in preds.iter() {
                        stacked_preds.push(pred[i]);
                    }
                }

                Array1::from_shape_fn(n, |i| {
                    let slice = &stacked_preds[i * n_grids..(i + 1) * n_grids];
                    combine_grids::geometric_mean_combiner(slice)
                })
            }
            Self::Combined => {
                // For Combined aggregation, extract f+ and f- separately
                let (f_plus, f_minus) =
                    extract_two_tensor_predictions_unscaled(&tgf.primary_grid_tensor, x);
                f_plus - f_minus
            }
        }
    }
}

/// Extract UNSCALED f+ and f- from two-tensor grid.
/// CRITICAL: This extracts raw values from two-tensor fields, NO scaling applied.
/// The grid.scaling field is IGNORED - we only use lambda_plus and lambda_minus.
pub fn extract_two_tensor_predictions_unscaled(
    grid: &GridTensor,
    x: ArrayView2<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let n = x.nrows();
    let mut f_plus = Array1::zeros(n);
    let mut f_minus = Array1::zeros(n);

    // For each point, compute directly from two-tensor fields:
    // f_+ = lambda_+ * prod_j (backbone[j][k] * exp(tilt[j][k]))
    // f_- = lambda_- * prod_j (backbone[j][k] * exp(-tilt[j][k]))
    //
    // CRITICAL: Do NOT multiply by grid.scaling!
    // - grid.scaling is ONLY used in legacy mode (when backbone_values is None)
    // - For two-tensor mode, GridTensor::predict_single_unscaled() does NOT apply scaling
    // - We extract f+ and f- using ONLY: lambda_+, lambda_-, backbone_values, tilt_values
    // - Scaling will be applied LATER via scaling_plus and scaling_minus from OLS solution
    // - This ensures scaling is applied exactly ONCE (in StagePredictor::predict())

    for i in 0..n {
        let mut fp = grid.lambda_plus;
        let mut fm = grid.lambda_minus;
        for j in 0..x.ncols() {
            let val = x[[i, j]];
            let col_idx = grid.splits[j].partition_point(|&split| split <= val);
            let col_idx = col_idx.min(grid.backbone_values[j].len() - 1);
            let b = grid.backbone_values[j][col_idx];
            let d = grid.tilt_values[j][col_idx];
            fp *= b * d.exp();
            fm *= b * (-d).exp();
        }
        f_plus[i] = fp;
        f_minus[i] = fm;
    }

    (f_plus, f_minus)
}

impl StagePredictor {
    pub fn get_grid_tensors(&self) -> &Vec<GridTensor> {
        &self.grid_tensors
    }

    pub fn get_primary_grid_tensor(&self) -> &GridTensor {
        &self.primary_grid_tensor
    }

    pub fn get_candidate_indices(&self) -> Option<&[usize]> {
        self.candidate_indices.as_deref()
    }

    pub fn new_exact(grid_tensors: Vec<GridTensor>, aggregation_method: Aggregation) -> Self {
        let primary_grid_tensor = grid_tensors[0].clone();
        Self {
            grid_tensors,
            primary_grid_tensor,
            candidate_indices: None,
            aggregation_method,
            scaling_plus: None,
            scaling_minus: None,
            energy: None,
        }
    }

    pub fn new_ensemble(
        grid_tensors: Vec<GridTensor>,
        primary_grid_tensor: GridTensor,
        candidate_indices: Vec<usize>,
        aggregation_method: Aggregation,
    ) -> Self {
        let scaling_plus = primary_grid_tensor.lambda_plus;
        let scaling_minus = primary_grid_tensor.lambda_minus;
        Self {
            grid_tensors,
            primary_grid_tensor,
            candidate_indices: Some(candidate_indices),
            aggregation_method,
            scaling_plus: Some(scaling_plus),
            scaling_minus: Some(scaling_minus),
            energy: None,
        }
    }

    pub fn predict_unscaled(&self, x: ArrayView2<f64>) -> Array1<f64> {
        self.aggregation_method.predict_unscaled(x, self)
    }

}

impl StagePredictor {
    pub fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        self.aggregation_method.predict(x, self)
    }
}

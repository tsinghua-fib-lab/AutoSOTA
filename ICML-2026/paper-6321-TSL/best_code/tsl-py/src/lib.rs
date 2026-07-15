use numpy::ndarray::{Array1, Array2, Array3, ArrayView2};
use pyo3::{
    prelude::*,
    types::{PyList, PyTuple},
};
use rand::{rngs::StdRng, SeedableRng};

use core::f64;
use std::fs;

use numpy::{PyArray1, PyArray2, PyArray3, ToPyArray};

use tsl::{
    forest::{fit_boosted, params::TSLBoostedParamsBuilder, TSL},
    grid_tensor::{
        self,
        params::{RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder},
        GridTensor, GridTensorParamsBuilder,
    },
    stage_predictor::StagePredictor,
    FitResult,
};

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::types::PyType;

#[derive(Debug)]
#[pyclass(name = "GridTensor")]
pub struct GridTensorPy(GridTensor);

impl From<GridTensor> for GridTensorPy {
    fn from(tg: GridTensor) -> Self {
        GridTensorPy(tg)
    }
}

#[derive(Debug)]
#[pyclass(name = "FitResult")]
pub struct FitResultPy {
    #[pyo3(get)]
    err: f64,
    #[pyo3(get)]
    residuals: Py<PyArray1<f64>>,
    #[pyo3(get)]
    y_hat: Py<PyArray1<f64>>,
}

impl From<FitResult> for FitResultPy {
    fn from(fit_result: FitResult) -> Self {
        Python::with_gil(|py| FitResultPy {
            err: fit_result.err,
            residuals: fit_result.residuals.to_pyarray(py).unbind(),
            y_hat: fit_result.y_hat.to_pyarray(py).unbind(),
        })
    }
}

#[derive(Debug)]
#[pyclass(name = "StagePredictor")]
pub struct StagePredictorPy(StagePredictor);

#[pymethods]
impl StagePredictorPy {
    #[getter]
    pub fn get_grid_tensors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let grid_tensors_py: Vec<GridTensorPy> = self
            .0
            .get_grid_tensors()
            .iter()
            .map(|tg| GridTensorPy::from(tg.clone()))
            .collect();

        PyList::new(py, grid_tensors_py)
    }

    #[getter]
    pub fn get_combined_grid_tensor(&self) -> PyResult<GridTensorPy> {
        Ok(GridTensorPy::from(self.0.get_primary_grid_tensor().clone()))
    }

    #[getter]
    pub fn get_candidate_indices(&self) -> PyResult<Vec<usize>> {
        Ok(self.0.get_candidate_indices().unwrap_or_default().to_vec())
    }

    #[getter]
    pub fn get_scaling_plus(&self) -> Option<f64> {
        self.0.scaling_plus
    }

    #[getter]
    pub fn get_scaling_minus(&self) -> Option<f64> {
        self.0.scaling_minus
    }

    #[pyo3(name = "predict")]
    pub fn _predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = x.as_array();
        let y_hat = self.0.predict(x);
        Ok(y_hat.to_pyarray(py))
    }
}

#[derive(Debug)]
#[pyclass(name = "TSL")]
pub struct TSLPy(TSL);

impl From<TSL> for TSLPy {
    fn from(tsl: TSL) -> Self {
        TSLPy(tsl)
    }
}

#[pymethods]
impl TSLPy {
    #[getter]
    pub fn get_stage_predictors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let stage_predictors_py: Vec<StagePredictorPy> = self
            .0
            .get_stage_predictors()
            .iter()
            .map(|tgf: &StagePredictor| StagePredictorPy(tgf.clone()))
            .collect();

        PyList::new(py, stage_predictors_py)
    }

    #[pyo3(name = "predict")]
    pub fn _predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = x.as_array();
        let y_hat = self.0.predict(x);
        Ok(y_hat.to_pyarray(py))
    }

    /// Compute partial dependence function: E_{X_S^c}[f(X_S, X_S^c)]
    ///
    /// For each tree grid family (epoch), computes the partial dependence where
    /// some features are fixed and others are marginalized over their empirical joint distribution.
    ///
    /// Computed as: (1/n) ∑_{i=1}^n f(x_S, x_{i,S^c}) where we average over training samples,
    /// preserving the joint distribution of marginalized features (not assuming independence).
    ///
    /// Parameters:
    /// - fixed_indices: Vector of feature indices that are fixed (in order)
    /// - fixed_values: Array2 of shape (n_observations, len(fixed_indices))
    ///                 Each row is one observation, columns correspond to fixed_indices order
    /// - data_x: Training data for estimating marginal distributions (n_samples, n_features)
    ///
    /// Returns:
    /// - Tuple of (constants_per_epoch, pd_values):
    ///   - constants_per_epoch: list of (C_plus, C_minus) per epoch
    ///     Constants are E[∏_{j ∉ S} f_j(X_j)] (expectation over marginalized features)
    ///     Constants include OLS scaling (computed with effective_lambda = scaling * lambda)
    ///   - pd_values: Array2 of shape (n_observations, 2 * n_epochs) with columns
    ///     [f+_epoch0, f-_epoch0, f+_epoch1, f-_epoch1, ...]
    ///     Values have scaling absorbed into lambda, so final prediction per epoch = f+ + f- (add columns)
    #[pyo3(name = "compute_partial_dependence_function")]
    pub fn _compute_partial_dependence_function<'py>(
        &self,
        py: Python<'py>,
        fixed_indices: Vec<usize>,
        fixed_values: PyReadonlyArray2<'py, f64>,
        data_x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let fixed_values = fixed_values.as_array();
        let fixed_values_contiguous = fixed_values.as_standard_layout();
        let data_x = data_x.as_array();
        let data_x_contiguous = data_x.as_standard_layout();

        let n_observations = fixed_values_contiguous.nrows();
        let n_fixed = fixed_indices.len();
        let n_epochs = self.0.get_stage_predictors().len();

        // Validate dimensions
        if fixed_values_contiguous.ncols() != n_fixed {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "fixed_values must have {} columns (one per fixed_indices), got {}",
                n_fixed,
                fixed_values_contiguous.ncols()
            )));
        }

        // Convert fixed_indices to HashSet for fast lookup
        let fixed_set: std::collections::HashSet<usize> = fixed_indices.iter().copied().collect();

        // Validate all indices are unique
        if fixed_set.len() != n_fixed {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "fixed_indices must contain unique values",
            ));
        }

        // Initialize result array: (n_observations, 2 * n_epochs)
        // Columns alternate: [f+_epoch0, f-_epoch0, f+_epoch1, f-_epoch1, ...]
        let mut results = Array2::<f64>::zeros((n_observations, 2 * n_epochs));
        let mut constants_per_epoch: Vec<(f64, f64)> = Vec::with_capacity(n_epochs);

        for (epoch_idx, tgf) in self.0.get_stage_predictors().iter().enumerate() {
            let grid = &tgf.primary_grid_tensor;
            let n_features = grid.splits.len();

            // Validate fixed indices are in bounds
            for &idx in &fixed_indices {
                if idx >= n_features {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Feature index {} out of bounds (n_features={})",
                        idx, n_features
                    )));
                }
            }

            let n_samples = data_x_contiguous.nrows();

            // Get OLS scaling coefficients and absorb into lambda
            // Prediction formula: scaling_plus * f_plus + scaling_minus * (-f_minus)
            // So we use: effective_lambda_plus = scaling_plus * lambda_plus
            //            effective_lambda_minus = -scaling_minus * lambda_minus
            let scaling_plus = tgf.scaling_plus.unwrap_or(1.0);
            let scaling_minus = tgf.scaling_minus.unwrap_or(0.0);
            let effective_lambda_plus = scaling_plus * grid.lambda_plus;
            let effective_lambda_minus = -scaling_minus * grid.lambda_minus;

            // Step 1: Compute E[∏_{j ∉ S} f_j(X_j)] once per epoch
            // Since f = λ_± * ∏_j a_±,j, we compute the expectation of the product over non-fixed features
            // Note: lambda values already include scaling
            let mut expected_marginalized_plus = 0.0;
            let mut expected_marginalized_minus = 0.0;

            for train_idx in 0..n_samples {
                let mut prod_plus = effective_lambda_plus;
                let mut prod_minus = effective_lambda_minus;

                // Only iterate over non-fixed features
                for j in 0..n_features {
                    if fixed_set.contains(&j) {
                        continue; // Skip fixed features
                    }

                    let x_j = data_x_contiguous[[train_idx, j]];
                    let col_idx = grid.splits[j].partition_point(|&split| split <= x_j);
                    let col_idx = col_idx.min(grid.backbone_values[j].len() - 1);

                    let b = grid.backbone_values[j][col_idx];
                    let d = grid.tilt_values[j][col_idx];

                    prod_plus *= b * d.exp();
                    prod_minus *= b * (-d).exp();
                }

                expected_marginalized_plus += prod_plus;
                expected_marginalized_minus += prod_minus;
            }

            expected_marginalized_plus /= n_samples as f64;
            expected_marginalized_minus /= n_samples as f64;

            // Store constants for this epoch (expectation over marginalized features)
            constants_per_epoch.push((expected_marginalized_plus, expected_marginalized_minus));

            // Step 2: For each observation, compute ∏_{j ∈ S} f_j(x_j) and multiply
            for obs_idx in 0..n_observations {
                let mut prod_fixed_plus = 1.0;
                let mut prod_fixed_minus = 1.0;

                // Only iterate over fixed features
                for (col_idx, &feature_idx) in fixed_indices.iter().enumerate() {
                    let x_j = fixed_values_contiguous[[obs_idx, col_idx]];

                    let interval_idx =
                        grid.splits[feature_idx].partition_point(|&split| split <= x_j);
                    let interval_idx =
                        interval_idx.min(grid.backbone_values[feature_idx].len() - 1);

                    let b = grid.backbone_values[feature_idx][interval_idx];
                    let d = grid.tilt_values[feature_idx][interval_idx];

                    prod_fixed_plus *= b * d.exp();
                    prod_fixed_minus *= b * (-d).exp();
                }

                // Combine: ∏_{j ∈ S} f_j(x_j) · E[∏_{j ∉ S} f_j(X_j)]
                // Scaling is already absorbed into effective_lambda values
                let f_plus = prod_fixed_plus * expected_marginalized_plus;
                let f_minus = prod_fixed_minus * expected_marginalized_minus;

                // Store f+ and f- separately (scaling already absorbed into lambda)
                // Columns alternate: [f+_epoch0, f-_epoch0, f+_epoch1, f-_epoch1, ...]
                // Final prediction per epoch = f+ + f- (add the two columns)
                results[[obs_idx, 2 * epoch_idx]] = f_plus;
                results[[obs_idx, 2 * epoch_idx + 1]] = f_minus;
            }
        }

        // Convert constants to Python list of tuples
        let constants_list = PyList::empty(py);
        for (c_plus, c_minus) in &constants_per_epoch {
            let constant_tuple = PyTuple::new(py, [*c_plus, *c_minus])?;
            constants_list.append(constant_tuple)?;
        }

        let pd_py = results.to_pyarray(py);
        let tuple = PyTuple::new(py, [constants_list.as_any(), pd_py.as_any()])?;
        Ok(tuple)
    }

    /// Compute first-order partial dependence functions for every feature.
    ///
    /// For each epoch and feature j, computes a constant C_{+,j} and C_{-,j}:
    ///   C_{+,j} = E[lambda_+ * ∏_{k != j} a_{+,k}(X_k)]
    ///   C_{-,j} = E[lambda_- * ∏_{k != j} a_{-,k}(X_k)]
    ///
    /// Then PD_{+,j}(x_j) = C_{+,j} * a_{+,j}(x_j) and
    ///      PD_{-,j}(x_j) = C_{-,j} * a_{-,j}(x_j).
    ///
    /// Returns a list of length p where each entry is:
    ///   (constants_per_epoch, pd_values)
    ///
    /// - constants_per_epoch: Vec of (C_plus, C_minus) for each epoch
    ///   Constants include OLS scaling (computed with effective_lambda = scaling * lambda)
    /// - pd_values: Array2 of shape (n_observations, 2 * n_epochs) with columns
    ///   [f+_epoch0, f-_epoch0, f+_epoch1, f-_epoch1, ...] for feature j.
    ///   Values have scaling absorbed into constants, so final prediction per epoch = f+ + f- (add columns)
    #[pyo3(name = "compute_first_order_partial_dependence_functions")]
    pub fn _compute_first_order_partial_dependence_functions<'py>(
        &self,
        py: Python<'py>,
        values_x: PyReadonlyArray2<'py, f64>,
        data_x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyList>> {
        let values_x = values_x.as_array();
        let values_x_contiguous = values_x.as_standard_layout();
        let data_x = data_x.as_array();
        let data_x_contiguous = data_x.as_standard_layout();

        let n_observations = values_x_contiguous.nrows();
        let n_features = values_x_contiguous.ncols();
        let n_samples = data_x_contiguous.nrows();
        let n_epochs = self.0.get_stage_predictors().len();

        if n_samples == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "data_x must contain at least one sample",
            ));
        }
        if data_x_contiguous.ncols() != n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "values_x and data_x must have the same number of columns, got {} and {}",
                n_features,
                data_x_contiguous.ncols()
            )));
        }

        let mut constants_per_dim: Vec<Vec<(f64, f64)>> =
            vec![Vec::with_capacity(n_epochs); n_features];
        let mut pd_values_per_dim: Vec<Array2<f64>> = (0..n_features)
            .map(|_| Array2::<f64>::zeros((n_observations, 2 * n_epochs)))
            .collect();

        for (epoch_idx, tgf) in self.0.get_stage_predictors().iter().enumerate() {
            let grid = &tgf.primary_grid_tensor;
            let grid_features = grid.splits.len();

            if grid_features != n_features {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Feature count mismatch: values_x has {} columns but grid has {}",
                    n_features, grid_features
                )));
            }

            // Get OLS scaling coefficients and absorb into lambda
            // Prediction formula: scaling_plus * f_plus + scaling_minus * (-f_minus)
            // So we use: effective_lambda_plus = scaling_plus * lambda_plus
            //            effective_lambda_minus = -scaling_minus * lambda_minus
            let scaling_plus = tgf.scaling_plus.unwrap_or(1.0);
            let scaling_minus = tgf.scaling_minus.unwrap_or(0.0);
            let effective_lambda_plus = scaling_plus * grid.lambda_plus;
            let effective_lambda_minus = -scaling_minus * grid.lambda_minus;

            let mut sum_plus = vec![0.0; n_features];
            let mut sum_minus = vec![0.0; n_features];

            let mut factors_plus = vec![0.0; n_features];
            let mut factors_minus = vec![0.0; n_features];
            let mut prefix_plus = vec![1.0; n_features + 1];
            let mut prefix_minus = vec![1.0; n_features + 1];
            let mut suffix_plus = vec![1.0; n_features + 1];
            let mut suffix_minus = vec![1.0; n_features + 1];

            for sample_idx in 0..n_samples {
                for j in 0..n_features {
                    let x_j = data_x_contiguous[[sample_idx, j]];
                    let interval_idx = grid.splits[j].partition_point(|&split| split <= x_j);
                    let interval_idx =
                        interval_idx.min(grid.backbone_values[j].len().saturating_sub(1));

                    let b = grid.backbone_values[j][interval_idx];
                    let d = grid.tilt_values[j][interval_idx];

                    factors_plus[j] = b * d.exp();
                    factors_minus[j] = b * (-d).exp();
                }

                prefix_plus[0] = 1.0;
                prefix_minus[0] = 1.0;
                for j in 0..n_features {
                    prefix_plus[j + 1] = prefix_plus[j] * factors_plus[j];
                    prefix_minus[j + 1] = prefix_minus[j] * factors_minus[j];
                }

                suffix_plus[n_features] = 1.0;
                suffix_minus[n_features] = 1.0;
                for j in (0..n_features).rev() {
                    suffix_plus[j] = suffix_plus[j + 1] * factors_plus[j];
                    suffix_minus[j] = suffix_minus[j + 1] * factors_minus[j];
                }

                for j in 0..n_features {
                    // Use effective lambdas (scaling already absorbed)
                    sum_plus[j] += effective_lambda_plus * prefix_plus[j] * suffix_plus[j + 1];
                    sum_minus[j] += effective_lambda_minus * prefix_minus[j] * suffix_minus[j + 1];
                }
            }

            let mut constants_plus = vec![0.0; n_features];
            let mut constants_minus = vec![0.0; n_features];
            let denom = n_samples as f64;
            for j in 0..n_features {
                let c_plus = sum_plus[j] / denom;
                let c_minus = sum_minus[j] / denom;
                constants_plus[j] = c_plus;
                constants_minus[j] = c_minus;
                constants_per_dim[j].push((c_plus, c_minus));
            }

            for obs_idx in 0..n_observations {
                for j in 0..n_features {
                    let x_j = values_x_contiguous[[obs_idx, j]];
                    let interval_idx = grid.splits[j].partition_point(|&split| split <= x_j);
                    let interval_idx =
                        interval_idx.min(grid.backbone_values[j].len().saturating_sub(1));

                    let b = grid.backbone_values[j][interval_idx];
                    let d = grid.tilt_values[j][interval_idx];

                    let m_plus = b * d.exp();
                    let m_minus = b * (-d).exp();

                    // Compute PD values (scaling already absorbed into constants via effective_lambda)
                    let col_plus = 2 * epoch_idx;
                    let col_minus = 2 * epoch_idx + 1;
                    pd_values_per_dim[j][[obs_idx, col_plus]] = constants_plus[j] * m_plus;
                    pd_values_per_dim[j][[obs_idx, col_minus]] = constants_minus[j] * m_minus;
                }
            }
        }

        let mut output_items: Vec<Bound<'_, PyAny>> = Vec::with_capacity(n_features);
        for j in 0..n_features {
            // Convert Vec<(f64, f64)> to Python list of tuples
            let constants_list = PyList::empty(py);
            for (c_plus, c_minus) in &constants_per_dim[j] {
                let constant_tuple = PyTuple::new(py, [*c_plus, *c_minus])?;
                constants_list.append(constant_tuple)?;
            }

            let pd_py = pd_values_per_dim[j].to_pyarray(py);
            let tuple = PyTuple::new(py, [constants_list.as_any(), pd_py.as_any()])?;
            output_items.push(tuple.as_any().clone());
        }

        PyList::new(py, output_items)
    }

    /// Compute Individual Conditional Expectation (ICE) curves for a single feature.
    ///
    /// For each observation, varies the specified feature over the provided range while
    /// keeping all other features fixed at that observation's values. Computes f+ and f-
    /// separately for each epoch.
    ///
    /// Parameters:
    /// - observations: Array2 of shape (n_obs, n_features) - observations to compute ICE for
    /// - feature_index: Index of the feature to vary
    /// - x_range: Array1 of values to evaluate for the varying feature
    /// - data_x: Training data (n_samples, n_features) - used for validation only
    ///
    /// Returns:
    /// - Array3 of shape (n_obs, n_range_values, 2 * n_epochs)
    ///   Last dimension: [f+_epoch0, f-_epoch0, f+_epoch1, f-_epoch1, ...]
    ///   Values have scaling applied: scaling_plus * f_+ and scaling_minus * (-f_-)
    #[pyo3(name = "compute_ice_curves")]
    pub fn _compute_ice_curves<'py>(
        &self,
        py: Python<'py>,
        observations: PyReadonlyArray2<'py, f64>,
        feature_index: usize,
        x_range: PyReadonlyArray1<'py, f64>,
        data_x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let observations = observations.as_array();
        let observations_contiguous = observations.as_standard_layout();
        let x_range = x_range.as_array();
        let x_range_contiguous = x_range.as_standard_layout();
        let data_x = data_x.as_array();
        let data_x_contiguous = data_x.as_standard_layout();

        let n_obs = observations_contiguous.nrows();
        let n_features = observations_contiguous.ncols();
        let n_range_values = x_range_contiguous.len();
        let n_epochs = self.0.get_stage_predictors().len();

        // Validate dimensions
        if feature_index >= n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "feature_index {} out of bounds (n_features={})",
                feature_index, n_features
            )));
        }

        if data_x_contiguous.ncols() != n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "observations and data_x must have the same number of columns, got {} and {}",
                n_features,
                data_x_contiguous.ncols()
            )));
        }

        if n_range_values == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "x_range must contain at least one value",
            ));
        }

        // Initialize result array: (n_obs, n_range_values, 2 * n_epochs)
        let mut results = Array3::<f64>::zeros((n_obs, n_range_values, 2 * n_epochs));

        for (epoch_idx, tgf) in self.0.get_stage_predictors().iter().enumerate() {
            let grid = &tgf.primary_grid_tensor;
            let grid_features = grid.splits.len();

            if grid_features != n_features {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Feature count mismatch: observations has {} columns but grid has {}",
                    n_features, grid_features
                )));
            }

            // Get OLS scaling coefficients
            let scaling_plus = tgf.scaling_plus.unwrap_or(1.0);
            let scaling_minus = tgf.scaling_minus.unwrap_or(0.0);

            // For each observation
            for obs_idx in 0..n_obs {
                // Pre-compute constant factors for all features except the varying one
                let mut const_plus = grid.lambda_plus;
                let mut const_minus = grid.lambda_minus;

                for j in 0..n_features {
                    if j == feature_index {
                        continue; // Skip the varying feature
                    }

                    let x_j = observations_contiguous[[obs_idx, j]];
                    let interval_idx = grid.splits[j].partition_point(|&split| split <= x_j);
                    let interval_idx =
                        interval_idx.min(grid.backbone_values[j].len().saturating_sub(1));

                    let b = grid.backbone_values[j][interval_idx];
                    let d = grid.tilt_values[j][interval_idx];

                    const_plus *= b * d.exp();
                    const_minus *= b * (-d).exp();
                }

                // For each value in x_range
                for range_idx in 0..n_range_values {
                    let v = x_range_contiguous[range_idx];

                    // Find interval for the varying feature at value v
                    let interval_idx =
                        grid.splits[feature_index].partition_point(|&split| split <= v);
                    let interval_idx = interval_idx
                        .min(grid.backbone_values[feature_index].len().saturating_sub(1));

                    let b = grid.backbone_values[feature_index][interval_idx];
                    let d = grid.tilt_values[feature_index][interval_idx];

                    // Compute f+ and f- with the varying feature
                    let f_plus_unscaled = const_plus * (b * d.exp());
                    let f_minus_unscaled = const_minus * (b * (-d).exp());

                    // Apply scaling (matching prediction formula: scaling_plus * f_+ + scaling_minus * (-f_-))
                    // For ICE curves, return scaled f+ and scaled (-f-) separately
                    let f_plus_scaled = scaling_plus * f_plus_unscaled;
                    let f_minus_scaled = scaling_minus * (-f_minus_unscaled);

                    // Store results directly in 3D array
                    results[[obs_idx, range_idx, 2 * epoch_idx]] = f_plus_scaled;
                    results[[obs_idx, range_idx, 2 * epoch_idx + 1]] = f_minus_scaled;
                }
            }
        }

        // Return as 3D array: (n_obs, n_range_values, 2 * n_epochs)
        Ok(results.to_pyarray(py))
    }

    /// Compute per-stage feature importance metrics.
    ///
    /// For each stage ℓ and feature j, computes:
    /// - Backbone variance: Var_n[log b_j^(ℓ)(X_j)]
    /// - Tilt variance: Var_n[d_j^(ℓ)(X_j)]
    ///
    /// Returns:
    /// - Array2 of shape (n_stages, n_features) for backbone importance
    /// - Array2 of shape (n_stages, n_features) for tilt importance
    #[pyo3(name = "compute_per_stage_feature_importance")]
    pub fn _compute_per_stage_feature_importance<'py>(
        &self,
        py: Python<'py>,
        data_x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let data_x = data_x.as_array();
        let data_x_contiguous = data_x.as_standard_layout();

        let (backbone_importance, tilt_importance) =
            self.compute_per_stage_feature_importance_internal(data_x_contiguous.view())?;

        Ok((
            backbone_importance.to_pyarray(py),
            tilt_importance.to_pyarray(py),
        ))
    }

    /// Compute aggregated global feature importance.
    ///
    /// Computes stage weights based on energy:
    /// ω_ℓ = ||λ_{+,ℓ} m̂_{+}^{(ℓ)} - λ_{-,ℓ} m̂_{-}^{(ℓ)}||_n^2 / sum_k ||λ_{+,k} m̂_{+}^{(k)} - λ_{-,k} m̂_{-}^{(k)}||_n^2
    ///
    /// Then aggregates per-stage importance:
    /// I_j^b = sum_ℓ ω_ℓ I_j^{b,(ℓ)}
    /// I_j^d = sum_ℓ ω_ℓ I_j^{d,(ℓ)}
    ///
    /// Returns:
    /// - Array1 of shape (n_features,) for global backbone importance
    /// - Array1 of shape (n_features,) for global tilt importance
    /// - Array1 of shape (n_stages,) for stage weights
    #[pyo3(name = "compute_aggregated_feature_importance")]
    pub fn _compute_aggregated_feature_importance<'py>(
        &self,
        py: Python<'py>,
        data_x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let data_x = data_x.as_array();
        let data_x_contiguous = data_x.as_standard_layout();

        // Step 1: Compute per-stage feature importance
        let (backbone_per_stage, tilt_per_stage) =
            self.compute_per_stage_feature_importance_internal(data_x_contiguous.view())?;

        // Step 2: Compute stage weights
        let stage_weights = self.compute_stage_weights(data_x_contiguous.view())?;

        // Step 3: Aggregate global importance
        let n_stages = backbone_per_stage.nrows();
        let n_features = backbone_per_stage.ncols();
        let mut global_backbone = Array1::<f64>::zeros(n_features);
        let mut global_tilt = Array1::<f64>::zeros(n_features);

        for stage_idx in 0..n_stages {
            let weight = stage_weights[stage_idx];
            for j in 0..n_features {
                global_backbone[j] += weight * backbone_per_stage[[stage_idx, j]];
                global_tilt[j] += weight * tilt_per_stage[[stage_idx, j]];
            }
        }

        Ok((
            global_backbone.to_pyarray(py),
            global_tilt.to_pyarray(py),
            Array1::from_vec(stage_weights).to_pyarray(py),
        ))
    }

    /// Compute combined feature importance score.
    ///
    /// Combines backbone and tilt importance:
    /// I_j = I_j^b + γ * I_j^d
    ///
    /// Parameters:
    /// - data_x: Training data (n_samples, n_features)
    /// - gamma: Weight for tilt importance (default: 1.0)
    ///
    /// Returns:
    /// - Array1 of shape (n_features,) for combined importance
    /// - Array1 of shape (n_features,) for backbone importance
    /// - Array1 of shape (n_features,) for tilt importance
    #[pyo3(name = "compute_combined_feature_importance", signature = (data_x, gamma=None))]
    pub fn _compute_combined_feature_importance<'py>(
        &self,
        py: Python<'py>,
        data_x: PyReadonlyArray2<'py, f64>,
        gamma: Option<f64>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let gamma = gamma.unwrap_or(1.0);

        // Get aggregated importance by calling internal helpers directly
        let data_x = data_x.as_array();
        let data_x_contiguous = data_x.as_standard_layout();

        // Compute per-stage importance and stage weights
        let (backbone_per_stage, tilt_per_stage) =
            self.compute_per_stage_feature_importance_internal(data_x_contiguous.view())?;
        let stage_weights = self.compute_stage_weights(data_x_contiguous.view())?;

        // Aggregate global importance
        let n_stages = backbone_per_stage.nrows();
        let n_features = backbone_per_stage.ncols();
        let mut global_backbone = Array1::<f64>::zeros(n_features);
        let mut global_tilt = Array1::<f64>::zeros(n_features);

        for stage_idx in 0..n_stages {
            let weight = stage_weights[stage_idx];
            for j in 0..n_features {
                global_backbone[j] += weight * backbone_per_stage[[stage_idx, j]];
                global_tilt[j] += weight * tilt_per_stage[[stage_idx, j]];
            }
        }

        let combined = &global_backbone + &(gamma * &global_tilt);

        Ok((
            combined.to_pyarray(py),
            global_backbone.to_pyarray(py),
            global_tilt.to_pyarray(py),
        ))
    }

    /// Save the TSL model to a binary file (preserves exact floating point values).
    ///
    /// **Note**: Binary format is same-version-only. Models saved with one version of tsl-py
    /// may not load with a different version due to schema changes. For portability across
    /// versions, consider exporting model parameters or predictions instead.
    #[pyo3(name = "save")]
    pub fn save(&self, path: &str) -> PyResult<()> {
        let bytes = bincode::serialize(&self.0).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to serialize TSL model: {}",
                e
            ))
        })?;
        fs::write(path, bytes).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to write to file {}: {}",
                path, e
            ))
        })?;
        Ok(())
    }

    /// Load an TSL model from a binary file.
    ///
    /// **Note**: Binary format is same-version-only. Models saved with one version of tsl-py
    /// may not load with a different version due to schema changes.
    #[classmethod]
    #[pyo3(name = "load")]
    pub fn load(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<TSLPy> {
        let bytes = fs::read(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to read from file {}: {}",
                path, e
            ))
        })?;
        let tsl: TSL = bincode::deserialize(&bytes).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to deserialize TSL model: {}",
                e
            ))
        })?;
        Ok(TSLPy(tsl))
    }

    #[classmethod]
    #[pyo3(name = "fit")]
    #[pyo3(signature = (
        x, y, epochs,
        decay=1.0,
        n_trees=10,
        n_iter=10,
        split_try=10,
        colsample_bytree=0.8,
        alpha=0.0,
        complexity_penalty=0.0,
        min_split_loss=0.0,
        min_interval_samples=1,
        refinement_strategy="l2",
        prior_sample_size=0.0,
        update_clamp=f64::INFINITY,
        tilt_tau=0.01,
        tilt_rho=0.0,
        split_strategy="random",
        top_k=10,
        must_fill_all_k=true,
        similarity_threshold=0.0,
        bagged=false,
        seed=42,
        verbosity=1,
        visualdb=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn _fit<'py>(
        _cls: &Bound<'_, PyType>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        epochs: usize,
        decay: f64,
        n_trees: usize,
        n_iter: usize,
        split_try: usize,
        colsample_bytree: f64,
        alpha: f64,
        complexity_penalty: f64,
        min_split_loss: f64,
        min_interval_samples: usize,
        refinement_strategy: &str,
        prior_sample_size: f64,
        update_clamp: f64,
        tilt_tau: f64,
        tilt_rho: f64,
        split_strategy: &str,
        top_k: usize,
        must_fill_all_k: bool,
        similarity_threshold: f64,
        bagged: bool,
        seed: u64,
        verbosity: u8,
        visualdb: Option<String>,
    ) -> PyResult<(TSLPy, FitResultPy)> {
        let x = x.as_array();
        let y = y.as_array();
        let x_contiguous = x.as_standard_layout();
        let y_contiguous = y.as_standard_layout();

        let split_strategy_params = match split_strategy {
            "random" => SplitStrategyParamsBuilder::new()
                .random_split()
                .split_try(split_try)
                .colsample_bytree(colsample_bytree)
                .min_interval_samples(min_interval_samples)
                .min_split_loss(min_split_loss)
                .complexity_penalty(complexity_penalty)
                .build(),
            "best_split" => SplitStrategyParamsBuilder::new()
                .best_split()
                .min_interval_samples(min_interval_samples)
                .min_split_loss(min_split_loss)
                .complexity_penalty(complexity_penalty)
                .build(),
            "top_k" => SplitStrategyParamsBuilder::new()
                .top_k_splits()
                .top_k(top_k)
                .must_fill_all_k(must_fill_all_k)
                .min_interval_samples(min_interval_samples)
                .min_split_loss(min_split_loss)
                .complexity_penalty(complexity_penalty)
                .build(),
            s => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown split_strategy: '{s}'. Valid: 'random', 'best_split', 'top_k'"
                )))
            }
        };

        let refinement_strategy_params = match refinement_strategy {
            "l2" => RefinementStrategyParamsBuilder::new()
                .l2()
                .alpha(alpha)
                .prior_sample_size(prior_sample_size)
                .update_clamp(update_clamp)
                .tilt_tau(tilt_tau)
                .tilt_rho(tilt_rho)
                .build(),
            "huber" => RefinementStrategyParamsBuilder::new()
                .huber()
                .alpha(alpha)
                .prior_sample_size(prior_sample_size)
                .update_clamp(update_clamp)
                .tilt_tau(tilt_tau)
                .tilt_rho(tilt_rho)
                .build(),
            s => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown refinement_strategy: '{s}'. Valid: 'l2', 'huber'"
                )))
            }
        };

        let params = TSLBoostedParamsBuilder::new()
            .epochs(epochs)
            .decay(decay)
            .n_trees(n_trees)
            .n_iter(n_iter)
            .split_strategy(split_strategy_params)
            .refinement_strategy(refinement_strategy_params)
            .similarity_threshold(similarity_threshold)
            .seed(seed)
            .log_level(match verbosity {
                0 => "off",
                1 => "info",
                2 => "debug",
                3 => "trace",
                _ => "info",
            })
            .visualdb_path(visualdb)
            .build();

        let (fit_result, tsl) = fit_boosted(x_contiguous.view(), y_contiguous.view(), &params);
        Ok((tsl.into(), FitResultPy::from(fit_result)))
    }
}

// Internal helper methods (not exposed to Python)
impl TSLPy {
    /// Internal helper: Compute per-stage feature importance (returns owned arrays).
    /// This is used internally to avoid duplication across the three public methods.
    fn compute_per_stage_feature_importance_internal(
        &self,
        data_x: ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), PyErr> {
        let n_samples = data_x.nrows();
        let n_features = data_x.ncols();

        let stage_predictors = self.0.get_stage_predictors();
        let n_stages = stage_predictors.len();

        // Validate dimensions
        for tgf in stage_predictors.iter() {
            let grid = tgf.get_primary_grid_tensor();
            if grid.splits.len() != n_features {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Feature dimension mismatch: data has {} features, but model has {}",
                    n_features,
                    grid.splits.len()
                )));
            }
        }

        // Initialize result arrays: (n_stages, n_features)
        let mut backbone_importance = Array2::<f64>::zeros((n_stages, n_features));
        let mut tilt_importance = Array2::<f64>::zeros((n_stages, n_features));

        for (stage_idx, tgf) in stage_predictors.iter().enumerate() {
            let grid = tgf.get_primary_grid_tensor();

            // For each feature j, evaluate b_j(x_j) and d_j(x_j) on all samples
            for j in 0..n_features {
                // Evaluate log(b_j(x_j)) and d_j(x_j) for all samples
                let mut log_b_values = Vec::with_capacity(n_samples);
                let mut d_values = Vec::with_capacity(n_samples);

                for i in 0..n_samples {
                    let x_j = data_x[[i, j]];
                    let col_idx = grid.splits[j].partition_point(|&split| split <= x_j);
                    let col_idx = col_idx.min(grid.backbone_values[j].len() - 1);

                    let b = grid.backbone_values[j][col_idx];
                    let d = grid.tilt_values[j][col_idx];

                    log_b_values.push(b.ln());
                    d_values.push(d);
                }

                // Compute mean
                let log_b_mean = log_b_values.iter().sum::<f64>() / n_samples as f64;
                let d_mean = d_values.iter().sum::<f64>() / n_samples as f64;

                // Compute variance
                let backbone_var = log_b_values
                    .iter()
                    .map(|&val| (val - log_b_mean).powi(2))
                    .sum::<f64>()
                    / n_samples as f64;
                let tilt_var = d_values
                    .iter()
                    .map(|&val| (val - d_mean).powi(2))
                    .sum::<f64>()
                    / n_samples as f64;

                backbone_importance[[stage_idx, j]] = backbone_var;
                tilt_importance[[stage_idx, j]] = tilt_var;
            }
        }

        Ok((backbone_importance, tilt_importance))
    }

    /// Internal helper: Compute stage weights based on energy.
    fn compute_stage_weights(&self, data_x: ArrayView2<f64>) -> Result<Vec<f64>, PyErr> {
        use tsl::stage_predictor::extract_two_tensor_predictions_unscaled;

        let stage_predictors = self.0.get_stage_predictors();
        let n_stages = stage_predictors.len();
        let n_samples = data_x.nrows();

        if n_stages == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Model has no stages",
            ));
        }

        // Compute stage weights based on energy
        // Energy = ||scaling_+ * f_+ - scaling_- * f_-||_n^2
        // where f_+ and f_- already include lambda, so we multiply by scaling
        let mut stage_energies = Vec::with_capacity(n_stages);

        for tgf in stage_predictors.iter() {
            let grid = tgf.get_primary_grid_tensor();
            let (f_plus, f_minus) = extract_two_tensor_predictions_unscaled(grid, data_x);

            // Get OLS scaling coefficients (default to 1.0 for scaling_plus, 0.0 for scaling_minus)
            let scaling_plus = tgf.scaling_plus.unwrap_or(1.0);
            let scaling_minus = tgf.scaling_minus.unwrap_or(0.0);

            // Compute ||scaling_+ * f_+ - scaling_- * f_-||_n^2
            // This matches the actual prediction formula: scaling_plus * f_+ + scaling_minus * (-f_-)
            let mut energy = 0.0;
            for i in 0..n_samples {
                let diff = scaling_plus * f_plus[i] - scaling_minus * f_minus[i];
                energy += diff * diff;
            }
            energy /= n_samples as f64;
            stage_energies.push(energy);
        }

        // Normalize stage weights
        let total_energy: f64 = stage_energies.iter().sum();
        let stage_weights: Vec<f64> = if total_energy > 0.0 {
            stage_energies.iter().map(|&e| e / total_energy).collect()
        } else {
            // If total energy is zero, use uniform weights
            vec![1.0 / n_stages as f64; n_stages]
        };

        Ok(stage_weights)
    }
}

#[pymethods]
impl GridTensorPy {
    #[getter]
    pub fn get_scaling(&self) -> f64 {
        self.0.scaling
    }

    #[getter]
    pub fn get_splits<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        // Convert the reference to a PyList
        PyList::new(py, &self.0.splits)
    }

    #[getter]
    pub fn get_intervals<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        // Convert the reference to a PyList
        PyList::new(py, &self.0.intervals)
    }

    #[getter]
    pub fn mean_factor<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mean_factor = self.0.get_mean_factor();
        PyList::new(py, mean_factor)
    }

    #[getter]
    pub fn grid_values<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.mean_factor(py)
    }

    #[getter]
    pub fn backbone_values<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, &self.0.backbone_values)
    }

    #[getter]
    pub fn tilt_values<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, &self.0.tilt_values)
    }

    #[getter]
    pub fn lambda_plus(&self) -> f64 {
        self.0.lambda_plus
    }

    #[getter]
    pub fn lambda_minus(&self) -> f64 {
        self.0.lambda_minus
    }

    #[pyo3(name = "predict")]
    pub fn _predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = x.as_array();
        let y_hat = self.0.predict(x);
        Ok(y_hat.to_pyarray(py))
    }

    #[classmethod]
    #[pyo3(name = "fit")]
    #[pyo3(signature = (x, y, n_iter, split_try, colsample_bytree, complexity_penalty=0.0, seed=42))]
    pub fn _fit<'py>(
        _cls: &Bound<'_, PyType>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        n_iter: usize,
        split_try: usize,
        colsample_bytree: f64,
        complexity_penalty: f64,
        seed: u64,
    ) -> PyResult<(GridTensorPy, FitResultPy)> {
        let x = x.as_array();
        let y = y.as_array();
        let x_contiguous = x.as_standard_layout();
        let y_contiguous = y.as_standard_layout();
        let params = GridTensorParamsBuilder::new()
            .n_iter(n_iter)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .random_split()
                    .split_try(split_try)
                    .colsample_bytree(colsample_bytree)
                    .min_interval_samples(1)
                    .complexity_penalty(complexity_penalty)
                    .build(),
            )
            .build();
        let mut rng = StdRng::seed_from_u64(seed);
        let (fit_result, tg) =
            grid_tensor::fit(x_contiguous.view(), y_contiguous.view(), &params, &mut rng);
        Ok((tg.into(), FitResultPy::from(fit_result)))
    }
}

#[pymethods]
impl FitResultPy {
    fn __repr__(&self) -> String {
        format!(
            "FitResult(error={}, residuals={}, y_hat={})",
            self.err, self.residuals, self.y_hat
        )
    }
}

#[pymodule]
fn _tensorsl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GridTensorPy>()?;
    m.add_class::<FitResultPy>()?;
    m.add_class::<StagePredictorPy>()?;
    m.add_class::<TSLPy>()?;
    Ok(())
}

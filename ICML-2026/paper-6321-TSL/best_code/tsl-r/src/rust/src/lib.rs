use extendr_api::prelude::*;
use ndarray::{ArrayView1, ArrayView2};
use tsl::{
    forest::{fit_boosted, params::TSLBoostedParamsBuilder, TSL},
    grid_tensor::{
        params::{RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder},
        GridTensor,
    },
};

// Fit a boosted TSL model.
//
// Mirrors the flat-hyperparameter `TSL.fit` classmethod in the Python bindings:
// the scalar arguments are mapped onto the `tsl` crate's strategy/param builders.
// Returns a list with the fitted model (an external pointer) and the training
// fit statistics. Wrapped for R users by `tsl()`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn tsl_fit(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    epochs: i32,
    decay: f64,
    n_trees: i32,
    n_iter: i32,
    split_try: i32,
    colsample_bytree: f64,
    alpha: f64,
    complexity_penalty: f64,
    min_split_loss: f64,
    min_interval_samples: i32,
    refinement_strategy: &str,
    prior_sample_size: f64,
    update_clamp: f64,
    tilt_tau: f64,
    tilt_rho: f64,
    split_strategy: &str,
    top_k: i32,
    must_fill_all_k: bool,
    similarity_threshold: f64,
    bagged: bool,
    seed: f64,
    verbosity: i32,
) -> List {
    // `bagged` is accepted for parity with the Python API; the bagged
    // aggregation path is selected by `similarity_threshold`, so it is unused here.
    let _ = bagged;

    let split_strategy_params = match split_strategy {
        "random" => SplitStrategyParamsBuilder::new()
            .random_split()
            .split_try(split_try as usize)
            .colsample_bytree(colsample_bytree)
            .min_interval_samples(min_interval_samples as usize)
            .min_split_loss(min_split_loss)
            .complexity_penalty(complexity_penalty)
            .build(),
        "best_split" => SplitStrategyParamsBuilder::new()
            .best_split()
            .min_interval_samples(min_interval_samples as usize)
            .min_split_loss(min_split_loss)
            .complexity_penalty(complexity_penalty)
            .build(),
        "top_k" => SplitStrategyParamsBuilder::new()
            .top_k_splits()
            .top_k(top_k as usize)
            .must_fill_all_k(must_fill_all_k)
            .min_interval_samples(min_interval_samples as usize)
            .min_split_loss(min_split_loss)
            .complexity_penalty(complexity_penalty)
            .build(),
        s => throw_r_error(format!(
            "Unknown split_strategy: '{s}'. Valid: 'random', 'best_split', 'top_k'"
        )),
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
        s => throw_r_error(format!(
            "Unknown refinement_strategy: '{s}'. Valid: 'l2', 'huber'"
        )),
    };

    let params = TSLBoostedParamsBuilder::new()
        .epochs(epochs as usize)
        .decay(decay)
        .n_trees(n_trees as usize)
        .n_iter(n_iter as usize)
        .split_strategy(split_strategy_params)
        .refinement_strategy(refinement_strategy_params)
        .similarity_threshold(similarity_threshold)
        .seed(seed as u64)
        .log_level(match verbosity {
            0 => "off",
            1 => "info",
            2 => "debug",
            3 => "trace",
            _ => "info",
        })
        .build();

    let (fit_result, model) = fit_boosted(x, y, &params);

    list!(
        model = ExternalPtr::new(model),
        err = fit_result.err,
        residuals = fit_result.residuals.to_vec(),
        y_hat = fit_result.y_hat.to_vec(),
    )
}

// Predict with a fitted TSL model. `model` is the external pointer returned by
// `tsl_fit`; `x` is the new design matrix (rows = samples, columns = features).
// Wrapped for R users by `predict.tsl()`.
#[extendr]
fn tsl_predict(model: ExternalPtr<TSL>, x: ArrayView2<f64>) -> Vec<f64> {
    model.as_ref().predict(x).to_vec()
}

// Pack each axis of a per-interval-per-axis field into an R list of vectors.
fn axis_list_f64(axes: &[Vec<f64>]) -> List {
    List::from_values(axes.iter().map(|axis| axis.clone()))
}

// Convert one fitted separable component into its two-tensor form: per-axis
// `splits`, `backbone_values` (b ≥ 0), `tilt_values` (d ∈ ℝ), interval sample
// `observation_counts`, and the branch scalars `lambda_plus`/`lambda_minus`.
fn grid_tensor_to_list(grid: &GridTensor) -> List {
    let counts = List::from_values(
        grid.observation_counts
            .iter()
            .map(|axis| axis.iter().map(|&c| c as i32).collect::<Vec<i32>>()),
    );
    list!(
        splits = axis_list_f64(&grid.splits),
        backbone_values = axis_list_f64(&grid.backbone_values),
        tilt_values = axis_list_f64(&grid.tilt_values),
        observation_counts = counts,
        lambda_plus = grid.lambda_plus,
        lambda_minus = grid.lambda_minus,
        scaling = grid.scaling
    )
}

// Extract the fitted structure of a boosted model: one entry per stage, each
// with its OLS scalings, the indices of the bagged trees kept after similarity
// filtering, the aggregated `combined_grid_tensor`, and the bag of per-tree
// `grid_tensors`. Wrapped for R users by `tsl_components()`.
#[extendr]
fn tsl_model_structure(model: ExternalPtr<TSL>) -> List {
    let stages: Vec<Robj> = model
        .as_ref()
        .get_stage_predictors()
        .iter()
        .map(|sp| -> Robj {
            let grid_tensors =
                List::from_values(sp.get_grid_tensors().iter().map(grid_tensor_to_list));
            // 1-based indices of the per-tree components in the stage.
            let candidate_indices: Vec<i32> = sp
                .get_candidate_indices()
                .map(|idx| idx.iter().map(|&i| i as i32 + 1).collect())
                .unwrap_or_default();
            list!(
                scaling_plus = sp.scaling_plus.unwrap_or(f64::NAN),
                scaling_minus = sp.scaling_minus.unwrap_or(f64::NAN),
                candidate_indices = candidate_indices,
                combined_grid_tensor = grid_tensor_to_list(sp.get_primary_grid_tensor()),
                grid_tensors = grid_tensors
            )
            .into()
        })
        .collect();
    List::from_values(stages)
}

extendr_module! {
    mod tensorsl;
    fn tsl_fit;
    fn tsl_predict;
    fn tsl_model_structure;
}

use crate::logging::{self, init_logging};

#[cfg(feature = "evo-logging")]
use crate::logging::{create_logger, LoggingConfig};
use crate::FitResult;
use ndarray::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

use super::{params::TSLBoostedParams, TSL};
use crate::stage_predictor::fit_ensemble;
use nalgebra::{DMatrix, DVector};

type Logger = Option<std::rc::Rc<std::cell::RefCell<Box<dyn crate::logging::EvoLogger>>>>;

/// Minimum-norm least-squares solution of `x w ≈ y` via SVD.
///
/// Singular values at or below `σ_max · max(n, k) · ε` are treated as zero, so a
/// rank-deficient design — which arises once stages contribute near-collinear
/// `f₊`/`−f₋` columns — yields the pseudo-inverse solution instead of diverging.
/// Returns `None` only when the SVD lacks the factors needed to back-substitute.
fn solve_least_squares(x: ArrayView2<f64>, y: ArrayView1<f64>) -> Option<Array1<f64>> {
    let (n, k) = (x.nrows(), x.ncols());
    let a = DMatrix::from_row_iterator(n, k, x.iter().copied());
    let b = DVector::from_iterator(n, y.iter().copied());
    let svd = a.svd(true, true);
    let sigma_max = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let cutoff = sigma_max * (n.max(k) as f64) * f64::EPSILON;
    let w = svd.solve(&b, cutoff).ok()?;
    Some(Array1::from_iter(w.iter().copied()))
}

/// Set up logging and event channel if visualdb_path is specified
fn setup_logging(
    db_path: Option<&String>,
    hyperparameters: &TSLBoostedParams,
    n_rows: usize,
    n_cols: usize,
) -> Option<crate::logging::EventChannel> {
    if let Some(db_path) = db_path {
        #[cfg(feature = "evo-logging")]
        {
            log::info!("Will save split events to database: {}", db_path);
            let config = LoggingConfig {
                db_path: db_path.clone(),
                run_label: None,
                record_residual_updates: false,
                pack_updates_as_blob: false,
            };
            let boxed = create_logger(Some(config));
            let logger_rc = std::rc::Rc::new(std::cell::RefCell::new(boxed));

            // Start the run
            let params_json = serde_json::to_string(hyperparameters).unwrap();
            logger_rc
                .borrow_mut()
                .start_run(&params_json, n_rows, n_cols);

            // Set up event channel
            let (tx, rx) = std::sync::mpsc::channel();
            crate::logging::set_event_sender(tx);

            Some((rx, logger_rc))
        }

        #[cfg(not(feature = "evo-logging"))]
        {
            log::warn!(
                "visualdb parameter specified but evo-logging feature is not enabled. Split events will not be saved to database: {}",
                db_path
            );
            None
        }
    } else {
        None
    }
}

/// Log combined grid snapshot for visualization
#[cfg(feature = "evo-logging")]
fn log_combined_grid_snapshot(
    logger: &Logger,
    stage_predictor: &crate::stage_predictor::StagePredictor,
    epoch: usize,
    f_plus: Option<&ndarray::Array1<f64>>,
    f_minus: Option<&ndarray::Array1<f64>>,
) {
    if logger.is_some() {
        use crate::grid_tensor::GridTensor;
        use serde_json::json;
        let primary: &GridTensor = &stage_predictor.primary_grid_tensor;

        // Build JSON with legacy fields and two-tensor fields
        let mean_factor = primary.get_mean_factor();
        let grid_json = json!({
            "grid_values": mean_factor,  // Computed: b * cosh(d)
            "splits": primary.splits,
            "intervals": primary.intervals,
            // Two-tensor fields
            "backbone_values": primary.backbone_values,
            "tilt_values": primary.tilt_values,
            "lambda_plus": primary.lambda_plus,
            "lambda_minus": primary.lambda_minus,
        })
        .to_string();

        // Convert f+ and f- arrays to Vec if provided
        let f_plus_vec = f_plus.map(|arr| arr.to_vec());
        let f_minus_vec = f_minus.map(|arr| arr.to_vec());

        logging::log_combined_grid(logging::CombinedGridSnapshot {
            epoch,
            energy: stage_predictor.energy,
            scaling: stage_predictor.scaling_plus, // Legacy field for backward compatibility
            scaling_plus: stage_predictor.scaling_plus,
            scaling_minus: stage_predictor.scaling_minus,
            grid_json,
            f_plus: f_plus_vec,
            f_minus: f_minus_vec,
        });
    }
}

/// Flush events to database if logging was set up
fn flush_logger(logger: &Logger) {
    #[cfg(feature = "evo-logging")]
    if let Some(ref logger_rc) = logger {
        if let Err(e) = logger_rc.borrow_mut().flush() {
            log::warn!("Failed to flush events to database: {}", e);
        } else {
            log::info!("Successfully saved split events to database");
        }
    }
}

/// Generic training loop for boosted model
fn train_boosted_model<F>(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &TSLBoostedParams,
    mut on_epoch_end: F,
) -> (FitResult, TSL)
where
    F: FnMut(usize, &crate::stage_predictor::StagePredictor, &[crate::stage_predictor::StagePredictor], &Logger),
{
    init_logging(&hyperparameters.log_level);
    log::info!(
        "Fitting boosted model with hyperparameters: {:#?}",
        hyperparameters
    );

    // Set up logging and event channel if visualdb_path is specified
    let event_channel = setup_logging(
        hyperparameters.visualdb_path.as_ref(),
        hyperparameters,
        x.nrows(),
        x.ncols(),
    );

    let mut sp_params = hyperparameters.sp_params.clone();

    let mut rng = StdRng::seed_from_u64(hyperparameters.seed);
    let mut y_new = y.to_owned();
    let mut stage_predictors: Vec<crate::stage_predictor::StagePredictor> = Vec::new();
    // Pre-allocate matrix to store unscaled predictions as columns for Orthogonal Greedy
    // Always use 2 columns per epoch: f+ and -f- (will be populated in Step 4)
    let n_samples = x.nrows();
    let n_components_per_epoch = 2; // f+ and -f- columns per epoch
    let mut predictions_matrix =
        Array2::<f64>::zeros((n_samples, hyperparameters.epochs * n_components_per_epoch));

    for i in 0..hyperparameters.epochs {
        logging::with_epoch(i, || {
            log::info!("Fitting epoch {}", i);
            if i > 0 {
                let current_n_iter = sp_params.tg_params.n_iter as f64;
                sp_params.tg_params.n_iter =
                    ((current_n_iter * hyperparameters.decay).round() as usize).max(1);
            }

            // Single-family fitting (split_residuals removed - replaced by two-tensor f+ and -f- columns)
            {
                // Original single-family fitting logic
                let (_fit_result, mut stage_predictor) =
                    fit_ensemble(x.view(), y_new.view(), &sp_params, &mut rng);

                // Always use OrthogonalGreedy scaling strategy
                // Store f+ and f- for logging
                use crate::stage_predictor::extract_two_tensor_predictions_unscaled;

                // Extract UNSCALED f+ and f- predictions from combined grid
                // CRITICAL: Extract directly from two-tensor fields, do NOT apply any scaling
                // The grid's scaling field is ignored - we only use scaling_plus/scaling_minus from OLS
                let (f_plus, f_minus) =
                    extract_two_tensor_predictions_unscaled(&stage_predictor.primary_grid_tensor, x);

                // Store for logging
                let f_plus_opt: Option<ndarray::Array1<f64>> = Some(f_plus.clone());
                let f_minus_opt: Option<ndarray::Array1<f64>> = Some(f_minus.clone());

                // Compute energy for logging
                let energy_plus = f_plus.pow2().mean().unwrap();
                let energy_minus = f_minus.pow2().mean().unwrap();
                stage_predictor.energy = Some(energy_plus + energy_minus);
                log::info!(
                    "Energy of epoch {}: f+={:.6}, f-={:.6}, total={:.6}",
                    i,
                    energy_plus,
                    energy_minus,
                    energy_plus + energy_minus
                );

                // Store in predictions matrix: columns i*2 (f+) and i*2+1 (-f-)
                // Note: We store -f_minus in the design matrix (not f_minus)
                // This is because the model is: f = scaling_plus * f_+ + scaling_minus * (-f_-)
                predictions_matrix.column_mut(i * 2).assign(&f_plus);
                let f_minus_neg = -&f_minus;
                predictions_matrix
                    .column_mut(i * 2 + 1)
                    .assign(&f_minus_neg);

                // Slice matrix up to current epoch: columns 0..(2*(i+1))
                // At epoch i, we have 2*(i+1) columns total (2 columns per epoch: f+ and -f-)
                let k = 2 * (i + 1);
                let x_mat = predictions_matrix.slice(s![.., 0..k]);

                // Solve weighted OLS on ALL columns up to current epoch: X w ≈ y
                // This is incremental OLS: at each epoch, we add 2 new columns and refit on all columns
                // This is the "Orthogonal Greedy" approach - greedy in the sense that we add columns
                // incrementally (one stage at a time), and orthogonal in that we refit OLS on all columns
                let (err, residuals) = if let Some(w) = solve_least_squares(x_mat, y) {
                    log::info!("New projected weights (two-tensor mode): {:?}", w);

                    // Update scalings for all families based on OLS solution
                    // w has length k = 2*(i+1), with 2 coefficients per epoch
                    // Each family now needs TWO scalings: one for f+ and one for -f-
                    for (j, prev_tgf) in stage_predictors.iter_mut().enumerate() {
                        // j is the family index (0..i)
                        // Family j's coefficients are at w[2*j] (f+) and w[2*j+1] (-f-)
                        if 2 * j + 1 < w.len() {
                            prev_tgf.scaling_plus = Some(w[2 * j]); // f+ coefficient
                            prev_tgf.scaling_minus = Some(w[2 * j + 1]);
                            // -f- coefficient (note: already negative in design matrix)
                        }
                    }

                    // Current epoch's scalings
                    stage_predictor.scaling_plus = Some(w[k - 2]); // f+ coefficient
                    stage_predictor.scaling_minus = Some(w[k - 1]); // -f- coefficient

                    // Update residuals and error
                    let preds_all = x_mat.dot(&w);
                    let residuals = &y - &preds_all;
                    let err = residuals.mapv(|v| v.powi(2)).mean().unwrap();
                    let residuals_owned = residuals.to_owned();
                    (err, residuals_owned)
                } else {
                    log::warn!(
                        "OLS solve failed for OrthogonalGreedy, falling back to unscaled predictions"
                    );
                    // Fallback: use unscaled predictions (f+ - f-), no scaling applied
                    // This is the raw two-tensor prediction without OLS coefficients
                    let combined_preds = &f_plus - &f_minus;
                    let residuals = &y - &combined_preds;
                    let err = residuals.mapv(|v| v.powi(2)).mean().unwrap();
                    (err, residuals.to_owned())
                };

                // Log epoch scalings (always using OrthogonalGreedy)
                if event_channel.is_some() {
                    // Log all scalings that were updated (epochs 0..i)
                    for (j, prev_tgf) in stage_predictors.iter().enumerate() {
                        if let Some(scaling_plus) = prev_tgf.scaling_plus {
                            logging::log_epoch_scaling(logging::EpochScalingSnapshot {
                                epoch: j,
                                scaling: scaling_plus, // TODO: Update logging to show both scalings
                                optimization_epoch: i, // Re-optimized at epoch i
                            });
                        }
                    }
                    // Log current epoch's scaling
                    if let Some(scaling_plus) = stage_predictor.scaling_plus {
                        logging::log_epoch_scaling(logging::EpochScalingSnapshot {
                            epoch: i,
                            scaling: scaling_plus, // TODO: Update logging to show both scalings
                            optimization_epoch: i,
                        });
                    }
                }

                y_new = residuals;

                // Log the combined/primary tree grid snapshot for visualization
                #[cfg(feature = "evo-logging")]
                {
                    if let Some((_, ref logger_rc)) = event_channel {
                        log_combined_grid_snapshot(
                            &Some(logger_rc.clone()),
                            &stage_predictor,
                            i,
                            f_plus_opt.as_ref(),
                            f_minus_opt.as_ref(),
                        );
                    }
                }

                // Callback for specific logic (test error, etc)
                // Pass stage_predictors so callback can access all previous families with updated scalings
                #[cfg(feature = "evo-logging")]
                let logger_for_callback = event_channel
                    .as_ref()
                    .map(|(_, logger_rc)| logger_rc.clone());
                #[cfg(not(feature = "evo-logging"))]
                let logger_for_callback = None;
                on_epoch_end(
                    i,
                    &stage_predictor,
                    &stage_predictors,
                    &logger_for_callback,
                );

                log::info!("Epoch {}, error: {:?}", i, err);
                stage_predictors.push(stage_predictor);
            }
        });
    }

    let err = y_new.iter().map(|&x| x * x).sum::<f64>() / y_new.len() as f64;
    let fit_result = FitResult {
        err,
        residuals: y_new.clone(),
        y_hat: -y_new + y,
    };

    // Drain all events and flush to database if logging was set up
    #[cfg(feature = "evo-logging")]
    if let Some(ref event_channel) = event_channel {
        crate::logging::drain_logging_events(event_channel);
        crate::logging::clear_event_sender();
        flush_logger(&Some(event_channel.1.clone()));
    }

    (fit_result, TSL::new(stage_predictors))
}

pub fn fit_boosted(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &TSLBoostedParams,
) -> (FitResult, TSL) {
    train_boosted_model(
        x,
        y,
        hyperparameters,
        |_epoch, _stage_predictor, _prev_families, _logger| {
            // Logging is now done before the callback, so nothing to do here
        },
    )
}

/// Fit a boosted TSL model while tracking per-epoch test error on a provided
/// evaluation dataset. The stage-wise predictor at epoch `i` is the sum of
/// predictions from the first `i+1` fitted `StagePredictor`s.
pub fn fit_boosted_with_test_error(
    x_train: ArrayView2<f64>,
    y_train: ArrayView1<f64>,
    x_test: ArrayView2<f64>,
    y_test: ArrayView1<f64>,
    hyperparameters: &TSLBoostedParams,
) -> (FitResult, TSL, Array1<f64>) {
    // Maintain cumulative test predictions of the stage-wise model
    let mut test_errors: Vec<f64> = Vec::with_capacity(hyperparameters.epochs);

    let (fit_res, tsl) = train_boosted_model(
        x_train,
        y_train,
        hyperparameters,
        |epoch, stage_predictor, prev_families, logger| {
            // Scalings are updated retroactively, so we need to
            // recompute cumulative predictions from scratch using all families with updated scalings
            // prev_families contains families 0..epoch-1 with updated scalings
            // stage_predictor has the updated scaling for the current epoch

            // Recompute cumulative test predictions from scratch with current scalings
            let mut yhat_test_epoch = Array1::<f64>::zeros(x_test.nrows());
            // Add predictions from all previous families (with updated scalings)
            for family in prev_families {
                yhat_test_epoch += &family.predict(x_test.view());
            }

            // Add prediction from current family (with updated scaling)
            yhat_test_epoch += &stage_predictor.predict(x_test.view());

            let test_err = (y_test.to_owned() - &yhat_test_epoch)
                .pow2()
                .mean()
                .unwrap();

            // Log test error and combined grid snapshot
            #[cfg(feature = "evo-logging")]
            if logger.is_some() {
                use crate::logging::{
                    log_grid_error_combined, log_grid_error_fitted, GridErrorVariant,
                };

                // Log family-level test error
                log_grid_error_combined(test_err, GridErrorVariant::Test);
                // Also log per-tree test error for trees in current family
                // Note: This computes error if we replaced the current family with just this tree
                let mut per_tree_test_errs: Vec<f64> = Vec::new();
                for (tree_id, tg) in stage_predictor.get_grid_tensors().iter().enumerate() {
                    logging::with_tree_id(tree_id, || {
                        // Compute cumulative predictions: all previous families + this tree
                        let mut yhat_test_tree = Array1::<f64>::zeros(x_test.nrows());
                        // Add predictions from all previous families
                        for family in prev_families {
                            yhat_test_tree += &family.predict(x_test.view());
                        }
                        // Replace current family with just this tree
                        yhat_test_tree += &tg.predict(x_test.view());
                        let tree_test_err =
                            (y_test.to_owned() - &yhat_test_tree).pow2().mean().unwrap();
                        per_tree_test_errs.push(tree_test_err);
                        log_grid_error_fitted(tree_test_err, GridErrorVariant::Test);
                    });
                }
                if !per_tree_test_errs.is_empty() {
                    let mut min_err = f64::INFINITY;
                    let mut max_err = f64::NEG_INFINITY;
                    let mut sum_err = 0.0;
                    for e in &per_tree_test_errs {
                        if *e < min_err {
                            min_err = *e;
                        }
                        if *e > max_err {
                            max_err = *e;
                        }
                        sum_err += *e;
                    }
                    let mean_err = sum_err / per_tree_test_errs.len() as f64;
                    log::info!(
                            "Epoch {} per-tree test error range: min={:.6}, max={:.6}, mean={:.6} (n={})",
                            epoch, min_err, max_err, mean_err, per_tree_test_errs.len()
                        );
                }
            }

            log::info!("Epoch {}, test error: {:?}", epoch, test_err);

            // Logging is now done before the callback in train_boosted_model, so nothing to do here

            test_errors.push(test_err);
        },
    );

    (fit_res, tsl, Array1::from(test_errors))
}

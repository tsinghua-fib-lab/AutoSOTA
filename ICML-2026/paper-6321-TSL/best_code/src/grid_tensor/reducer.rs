#[cfg(feature = "evo-logging")]
use crate::grid_tensor::logging_helpers::create_simple_split_event;
use crate::grid_tensor::{
    action::FittingAction,
    refinement::RefinementStrategy,
    splitting::{MergeCandidate, ResplitCandidate, SplitCandidate, SplitStrategy},
    state::AffectedRange,
    state::FittingState,
};
#[cfg(feature = "evo-logging")]
use crate::logging::reducer::log_split_reducer;

/// Central reducer function that consumes state and action
/// Returns the new state (using move semantics, no cloning)
pub fn fitting_reducer<'a>(
    state: FittingState<'a>,
    action: FittingAction,
    refinement_strategy: &RefinementStrategy,
    split_strategy: &SplitStrategy,
) -> FittingState<'a> {
    match action {
        FittingAction::ApplyScaling { factor } => reduce_apply_scaling(state, factor),
        FittingAction::Terminate { reason } => reduce_terminate(state, reason),
        FittingAction::Composite { actions } => {
            reduce_composite(state, actions, refinement_strategy, split_strategy)
        }
        FittingAction::ApplyMerge { .. }
        | FittingAction::ApplyResplit { .. }
        | FittingAction::ApplySplit { .. } => {
            reduce_apply_transformation(state, action, refinement_strategy, split_strategy)
        }
    }
}

/// Apply a split/resplit/merge with all associated side effects
/// This combines: update_tree + forbid_around_split + update_statistics
fn reduce_apply_transformation<'a>(
    mut state: FittingState<'a>,
    transformation: FittingAction,
    refinement_strategy: &RefinementStrategy,
    split_strategy: &SplitStrategy,
) -> FittingState<'a> {
    // Capture state for logging (always capture, reducer handles None)
    let old_error = state.current_error;
    // Clone the action before transformation to avoid borrowing issues
    let action_for_logging = transformation.clone();

    // Step 1: Update tree structure (insert/remove/update boundaries and grid_values)

    state = match transformation {
        FittingAction::ApplySplit {
            split: split_candidate,
        } => {
            let idx = split_candidate.index;
            let col = split_candidate.col;
            let interval_idx = state.boundaries[col].partition_point(|&b| b <= idx);
            let original_affected_interval = state.interval_range(col, interval_idx);
            let (u_plus_l, u_minus_l) = split_candidate.update_left;
            let (u_plus_r, u_minus_r) = split_candidate.update_right;

            let v_min = crate::grid_tensor::two_tensor_solver::DEFAULT_V_MIN;
            let v_max = crate::grid_tensor::two_tensor_solver::DEFAULT_V_MAX;

            let is_stage1 = state.is_stage1_positive_only();
            let (v_b_l, delta_d_l, v_b_r, delta_d_r) = if is_stage1 {
                // Stage 1 positive-only: v_b = 1 + u directly, Δd = 0
                let v_b_l = (1.0 + u_plus_l).clamp(v_min, v_max);
                let v_b_r = (1.0 + u_plus_r).clamp(v_min, v_max);
                (v_b_l, 0.0, v_b_r, 0.0)
            } else {
                // Full two-tensor: convert (v_+, v_-) to (v_b, Δd)
                let v_plus_l = (1.0 + u_plus_l).clamp(v_min, v_max);
                let v_minus_l = (1.0 + u_minus_l).clamp(v_min, v_max);
                let v_plus_r = (1.0 + u_plus_r).clamp(v_min, v_max);
                let v_minus_r = (1.0 + u_minus_r).clamp(v_min, v_max);

                let (v_b_l, delta_d_l) =
                    crate::grid_tensor::two_tensor_solver::convert_multipliers_to_backbone_tilt(
                        v_plus_l, v_minus_l,
                    );
                let (v_b_r, delta_d_r) =
                    crate::grid_tensor::two_tensor_solver::convert_multipliers_to_backbone_tilt(
                        v_plus_r, v_minus_r,
                    );
                (v_b_l, delta_d_l, v_b_r, delta_d_r)
            };

            // Capture parent (old) axis factors before mutating parameters.
            let insert_at = state.boundaries[col].partition_point(|&b| b <= idx);
            let b_p = state.backbone_values[col][insert_at];
            let d_p = state.tilt_values[col][insert_at];
            let g_plus_old = b_p * d_p.exp();
            let g_minus_old = b_p * (-d_p).exp();

            state =
                reduce_grid_split_two_tensor(state, v_b_l, delta_d_l, v_b_r, delta_d_r, col, idx);
            let affected_ranges = state.get_affected_ranges(col, original_affected_interval);

            // Compute new axis factors after split using the *clamped* child parameters stored in state.
            let b_l = state.backbone_values[col][insert_at];
            let d_l = state.tilt_values[col][insert_at];
            let g_plus_new_l = b_l * d_l.exp();
            let g_minus_new_l = b_l * (-d_l).exp();
            let b_r = state.backbone_values[col][insert_at + 1];
            let d_r = state.tilt_values[col][insert_at + 1];
            let g_plus_new_r = b_r * d_r.exp();
            let g_minus_new_r = b_r * (-d_r).exp();

            state = reduce_grid_predictions_and_residuals_two_tensor_factors(
                state,
                col,
                idx,
                &original_affected_interval,
                g_plus_old,
                g_minus_old,
                g_plus_new_l,
                g_minus_new_l,
                g_plus_new_r,
                g_minus_new_r,
            );

            state = reduce_grid_statistics_and_refresh_caches(
                state,
                col,
                &original_affected_interval,
                &affected_ranges,
                refinement_strategy,
            );

            state = reduce_forbid_around_split(state, &split_candidate, split_strategy);
            state = reduce_loop_state_for_split(state, &split_candidate);
            state
        }
        FittingAction::ApplyMerge {
            merge: merge_candidate,
        } => {
            // Merge=OptimalMerge: compute optimal merged params at merge time using partial products
            let col = merge_candidate.col;
            let interval_idx = merge_candidate.interval_idx;
            let index = merge_candidate.index;
            let interval_left_and_right = state.interval_range_left_and_right(col, interval_idx);
            let original_affected_interval = (interval_left_and_right.0, interval_left_and_right.2);

            // Capture old (child) axis factors before merging
            let b_left = state.backbone_values[col][interval_idx];
            let d_left = state.tilt_values[col][interval_idx];
            let a_plus_left = b_left * d_left.exp(); // a_{+,j}^L = b_L * exp(d_L)
            let a_minus_left = b_left * (-d_left).exp(); // a_{-,j}^L = b_L * exp(-d_L)
            let g_plus_old_left = a_plus_left;
            let g_minus_old_left = a_minus_left;
            let b_right = state.backbone_values[col][interval_idx + 1];
            let d_right = state.tilt_values[col][interval_idx + 1];
            let g_plus_old_right = b_right * d_right.exp();
            let g_minus_old_right = b_right * (-d_right).exp();

            // Reuse stored optimal merged params from merge gain computation
            // These were computed using partest_split_reduces_error_by_correct_amount_over_multiple_splitstial products in update_error_reductions_merge_for_col_range
            // and stored in update_pairs_merge[col][interval_idx]
            let (u_plus_merged, u_minus_merged) =
                state.precomputed_statistics.update_pairs_merge[col][interval_idx];

            let is_stage1 = state.is_stage1_positive_only();
            let (b_merged_optimal, d_merged_optimal) = if is_stage1 {
                // Stage 1 positive-only: v_b = 1 + u directly, d = 0
                // Use left child's backbone as base: b_U = b_L * v_b_U
                let v_min = crate::grid_tensor::two_tensor_solver::DEFAULT_V_MIN;
                let v_max = crate::grid_tensor::two_tensor_solver::DEFAULT_V_MAX;
                let v_b_merged = (1.0 + u_plus_merged).clamp(v_min, v_max);
                let b_merged_optimal = b_left * v_b_merged;
                (b_merged_optimal, 0.0)
            } else {
                // Full two-tensor: convert (v_+, v_-) to (b, d)
                // Option A (per spec): Use left child's parameters as base
                // a_+^U = a_+^L * v_+^U, a_-^U = a_-^L * v_-^U
                let v_min = crate::grid_tensor::two_tensor_solver::DEFAULT_V_MIN;
                let v_max = crate::grid_tensor::two_tensor_solver::DEFAULT_V_MAX;
                let v_plus_merged = (1.0 + u_plus_merged).clamp(v_min, v_max);
                let v_minus_merged = (1.0 + u_minus_merged).clamp(v_min, v_max);

                let a_plus_merged = a_plus_left * v_plus_merged;
                let a_minus_merged = a_minus_left * v_minus_merged;

                // Convert to (b, d): b = sqrt(a_+ * a_-), d = 0.5 * ln(a_+ / a_-)
                let b_merged_optimal = (a_plus_merged * a_minus_merged).sqrt();
                let d_merged_optimal = 0.5 * (a_plus_merged / a_minus_merged).ln();
                (b_merged_optimal, d_merged_optimal)
            };

            // Apply merge: remove boundary and set merged params to absolute optimal values
            state = reduce_grid_merge_optimal(
                state,
                col,
                interval_idx,
                b_merged_optimal,
                d_merged_optimal,
            );
            let affected_ranges = state.get_affected_ranges(col, original_affected_interval);

            // Compute new (merged) axis factors after merge
            let b_merged = state.backbone_values[col][interval_idx];
            let d_merged = state.tilt_values[col][interval_idx];
            let g_plus_new = b_merged * d_merged.exp();
            let g_minus_new = b_merged * (-d_merged).exp();

            // Update per-point caches using factor ratios
            state = reduce_grid_predictions_and_residuals_merge_two_tensor_factors(
                state,
                col,
                index,
                &original_affected_interval,
                g_plus_old_left,
                g_minus_old_left,
                g_plus_old_right,
                g_minus_old_right,
                g_plus_new,
                g_minus_new,
            );

            state = reduce_grid_statistics_and_refresh_caches(
                state,
                col,
                &original_affected_interval,
                &affected_ranges,
                refinement_strategy,
            );

            // Recompute current_error from residuals to ensure accuracy
            state.current_error = state.residuals.iter().map(|r| r * r).sum::<f64>();

            state = reduce_loop_state_for_merge(state, &merge_candidate);
            state
        }
        FittingAction::ApplyResplit {
            resplit: resplit_candidate,
        } => {
            // Resplit doesn't change boundaries; no map updates needed
            let col = resplit_candidate.col;
            let interval_idx = resplit_candidate.interval_idx;
            let index = resplit_candidate.index;
            let interval_left_and_right = state.interval_range_left_and_right(col, interval_idx);
            let original_affected_interval = (interval_left_and_right.0, interval_left_and_right.2);
            // Convert two-tensor updates to (v_b, delta_d) for each side
            let (u_plus_l, u_minus_l) = resplit_candidate.update_left;
            let (u_plus_r, u_minus_r) = resplit_candidate.update_right;
            let v_min = crate::grid_tensor::two_tensor_solver::DEFAULT_V_MIN;
            let v_max = crate::grid_tensor::two_tensor_solver::DEFAULT_V_MAX;

            let is_stage1 = state.is_stage1_positive_only();
            let (v_b_l, delta_d_l, v_b_r, delta_d_r) = if is_stage1 {
                // Stage 1 positive-only: v_b = 1 + u directly, Δd = 0
                let v_b_l = (1.0 + u_plus_l).clamp(v_min, v_max);
                let v_b_r = (1.0 + u_plus_r).clamp(v_min, v_max);
                (v_b_l, 0.0, v_b_r, 0.0)
            } else {
                // Full two-tensor: convert (v_+, v_-) to (v_b, Δd)
                let v_plus_l = (1.0 + u_plus_l).clamp(v_min, v_max);
                let v_minus_l = (1.0 + u_minus_l).clamp(v_min, v_max);
                let v_plus_r = (1.0 + u_plus_r).clamp(v_min, v_max);
                let v_minus_r = (1.0 + u_minus_r).clamp(v_min, v_max);

                let (v_b_l, delta_d_l) =
                    crate::grid_tensor::two_tensor_solver::convert_multipliers_to_backbone_tilt(
                        v_plus_l, v_minus_l,
                    );
                let (v_b_r, delta_d_r) =
                    crate::grid_tensor::two_tensor_solver::convert_multipliers_to_backbone_tilt(
                        v_plus_r, v_minus_r,
                    );
                (v_b_l, delta_d_l, v_b_r, delta_d_r)
            };

            // Capture old axis factors for both intervals before updating.
            let b_left_old = state.backbone_values[col][interval_idx];
            let d_left_old = state.tilt_values[col][interval_idx];
            let g_plus_old_l = b_left_old * d_left_old.exp();
            let g_minus_old_l = b_left_old * (-d_left_old).exp();
            let b_right_old = state.backbone_values[col][interval_idx + 1];
            let d_right_old = state.tilt_values[col][interval_idx + 1];
            let g_plus_old_r = b_right_old * d_right_old.exp();
            let g_minus_old_r = b_right_old * (-d_right_old).exp();

            state = reduce_grid_resplit_two_tensor(
                state,
                v_b_l,
                delta_d_l,
                v_b_r,
                delta_d_r,
                col,
                interval_idx,
            );
            let affected_ranges = state.get_affected_ranges(col, original_affected_interval);

            // Compute new factors (after param update) for both intervals using stored (clamped) params.
            let b_left_new = state.backbone_values[col][interval_idx];
            let d_left_new = state.tilt_values[col][interval_idx];
            let g_plus_new_l = b_left_new * d_left_new.exp();
            let g_minus_new_l = b_left_new * (-d_left_new).exp();
            let b_right_new = state.backbone_values[col][interval_idx + 1];
            let d_right_new = state.tilt_values[col][interval_idx + 1];
            let g_plus_new_r = b_right_new * d_right_new.exp();
            let g_minus_new_r = b_right_new * (-d_right_new).exp();

            state = reduce_grid_predictions_and_residuals_resplit_two_tensor_factors(
                state,
                col,
                index,
                &original_affected_interval,
                g_plus_old_l,
                g_minus_old_l,
                g_plus_old_r,
                g_minus_old_r,
                g_plus_new_l,
                g_minus_new_l,
                g_plus_new_r,
                g_minus_new_r,
            );

            state = reduce_grid_statistics_and_refresh_caches(
                state,
                col,
                &original_affected_interval,
                &affected_ranges,
                refinement_strategy,
            );

            state = reduce_loop_state_for_resplit(state, &resplit_candidate);
            state
        }
        _ => unreachable!(),
    };

    // Last step: apply logging reducer for transformation event
    // Reducer pattern: logging is integrated as last step
    // Takes ownership of logging_state, returns updated state (no cloning)
    // Feature flag is encapsulated inside the reducer
    #[cfg(feature = "evo-logging")]
    {
        let split_event = create_simple_split_event(&state, &action_for_logging, old_error);
        state.logging_state = log_split_reducer(state.logging_state.take(), split_event);
    }

    state
}

/// Update tree structure: boundaries and grid_values
fn reduce_grid_split_two_tensor(
    mut state: FittingState<'_>,
    v_b_l: f64,
    delta_d_l: f64,
    v_b_r: f64,
    delta_d_r: f64,
    col: usize,
    index: usize,
) -> FittingState<'_> {
    let insert_at = state.boundaries[col].partition_point(|&b| b <= index);

    // Insert new boundary at the correct position
    state.boundaries[col].insert(insert_at, index);

    // Parent parameters
    let b_p = state.backbone_values[col][insert_at];
    let d_p = state.tilt_values[col][insert_at];

    // Child params: b_child = b_p * v_b, d_child = d_p + Δd
    // Use same clamping values as resplit for consistency
    const MIN_BACKBONE: f64 = 1e-10;
    const MAX_BACKBONE: f64 = 1e10;
    const MAX_TILT: f64 = 10.0;

    state.backbone_values[col][insert_at] = (b_p * v_b_l).clamp(MIN_BACKBONE, MAX_BACKBONE);
    let new_tilt_l = (d_p + delta_d_l).clamp(-MAX_TILT, MAX_TILT);
    state.tilt_values[col][insert_at] = new_tilt_l;

    state.backbone_values[col].insert(
        insert_at + 1,
        (b_p * v_b_r).clamp(MIN_BACKBONE, MAX_BACKBONE),
    );
    state.tilt_values[col].insert(insert_at + 1, (d_p + delta_d_r).clamp(-MAX_TILT, MAX_TILT));

    // Stage 1 invariants: I20-I27 (debug builds only)
    #[cfg(debug_assertions)]
    {
        if state.is_stage1_positive_only() {
            const EPS: f64 = 1e-10;
            // I20: λ_- = 0
            assert!(
                state.lambda_minus.abs() < EPS,
                "I20 violation: lambda_minus should be 0 in Stage 1 mode, got {}",
                state.lambda_minus
            );
            // I21: all d_j = 0
            for (j, d_vec) in state.tilt_values.iter().enumerate() {
                for (k, &d) in d_vec.iter().enumerate() {
                    assert!(
                        d.abs() < EPS,
                        "I21 violation: tilt[{j}][{k}] should be 0 in Stage 1 mode, got {}",
                        d
                    );
                }
            }
            // I22: f_- = 0
            for (i, &fm) in state.f_minus.iter().enumerate() {
                assert!(
                    fm.abs() < EPS,
                    "I22 violation: f_minus[{}] should be 0 in Stage 1 mode, got {}",
                    i,
                    fm
                );
            }
            // I23: f >= 0
            for (i, &f) in state.f.iter().enumerate() {
                assert!(
                    f >= -EPS,
                    "I23 violation: f[{}] should be >= 0 in Stage 1 mode, got {}",
                    i,
                    f
                );
            }
        }
    }

    // Insert placeholders for resplit/merge statistics aligned with the new boundary
    state.precomputed_statistics.error_reductions_resplit[col].insert(insert_at, f64::NAN);
    state.precomputed_statistics.update_pairs_resplit_left[col]
        .insert(insert_at, (f64::NAN, f64::NAN));
    state.precomputed_statistics.update_pairs_resplit_right[col]
        .insert(insert_at, (f64::NAN, f64::NAN));
    state.precomputed_statistics.error_reductions_resplit_pairs[col]
        .insert(insert_at, (f64::NAN, f64::NAN));
    state.precomputed_statistics.error_reductions_merge[col].insert(insert_at, f64::NAN);
    state.precomputed_statistics.update_pairs_merge[col].insert(insert_at, (f64::NAN, f64::NAN));

    // Split interval stats: insert placeholder for the new right interval
    // (actual values will be computed in update_statistics)
    state.precomputed_statistics.interval_stats[col]
        .insert(insert_at + 1, crate::grid_tensor::state::IntervalStats::default());

    state
}

/// Update per-point caches (f_±, f, y_hat, residuals, r_tilde) for a split using explicit axis factors.
///
/// Here:
/// - `g_plus_old/g_minus_old` are the parent axis factors on this column before the split
/// - `g_plus_new_* / g_minus_new_*` are the child axis factors for left/right after the split
fn reduce_grid_predictions_and_residuals_two_tensor_factors<'a>(
    mut state: FittingState<'a>,
    col: usize,
    index: usize,
    original_affected_interval: &(usize, usize),
    g_plus_old: f64,
    g_minus_old: f64,
    g_plus_new_l: f64,
    g_minus_new_l: f64,
    g_plus_new_r: f64,
    g_minus_new_r: f64,
) -> FittingState<'a> {
    let &(start, end) = original_affected_interval;

    let left_grid_points = &state.precomputed_statistics.sorted_indices[col][start..index];
    let right_grid_points = &state.precomputed_statistics.sorted_indices[col][index..end];

    let ratio_plus_left = if g_plus_old.abs() > 1e-12 {
        g_plus_new_l / g_plus_old
    } else {
        1.0
    };
    let ratio_minus_left = if g_minus_old.abs() > 1e-12 {
        g_minus_new_l / g_minus_old
    } else {
        1.0
    };
    let ratio_plus_right = if g_plus_old.abs() > 1e-12 {
        g_plus_new_r / g_plus_old
    } else {
        1.0
    };
    let ratio_minus_right = if g_minus_old.abs() > 1e-12 {
        g_minus_new_r / g_minus_old
    } else {
        1.0
    };

    for &i in left_grid_points {
        state.f_plus[i] *= ratio_plus_left;
        state.f_minus[i] *= ratio_minus_left;
        state.f[i] = state.f_plus[i] - state.f_minus[i];
        state.y_hat[i] = state.f[i];
        state.residuals[i] = state.labels[i] - state.y_hat[i];
        state.r_tilde[i] = state.residuals[i];
    }

    for &i in right_grid_points {
        state.f_plus[i] *= ratio_plus_right;
        state.f_minus[i] *= ratio_minus_right;
        state.f[i] = state.f_plus[i] - state.f_minus[i];
        state.y_hat[i] = state.f[i];
        state.residuals[i] = state.labels[i] - state.y_hat[i];
        state.r_tilde[i] = state.residuals[i];
    }

    state
}

/// Update per-point caches for a resplit using explicit old/new axis factors per side.
fn reduce_grid_predictions_and_residuals_resplit_two_tensor_factors<'a>(
    mut state: FittingState<'a>,
    col: usize,
    index: usize,
    original_affected_interval: &(usize, usize),
    g_plus_old_l: f64,
    g_minus_old_l: f64,
    g_plus_old_r: f64,
    g_minus_old_r: f64,
    g_plus_new_l: f64,
    g_minus_new_l: f64,
    g_plus_new_r: f64,
    g_minus_new_r: f64,
) -> FittingState<'a> {
    let &(start, end) = original_affected_interval;

    let left_grid_points = &state.precomputed_statistics.sorted_indices[col][start..index];
    let right_grid_points = &state.precomputed_statistics.sorted_indices[col][index..end];

    let ratio_plus_left = if g_plus_old_l.abs() > 1e-12 {
        g_plus_new_l / g_plus_old_l
    } else {
        1.0
    };
    let ratio_minus_left = if g_minus_old_l.abs() > 1e-12 {
        g_minus_new_l / g_minus_old_l
    } else {
        1.0
    };
    let ratio_plus_right = if g_plus_old_r.abs() > 1e-12 {
        g_plus_new_r / g_plus_old_r
    } else {
        1.0
    };
    let ratio_minus_right = if g_minus_old_r.abs() > 1e-12 {
        g_minus_new_r / g_minus_old_r
    } else {
        1.0
    };

    for &i in left_grid_points {
        state.f_plus[i] *= ratio_plus_left;
        state.f_minus[i] *= ratio_minus_left;
        state.f[i] = state.f_plus[i] - state.f_minus[i];
        state.y_hat[i] = state.f[i];
        state.residuals[i] = state.labels[i] - state.y_hat[i];
        state.r_tilde[i] = state.residuals[i];
    }

    for &i in right_grid_points {
        state.f_plus[i] *= ratio_plus_right;
        state.f_minus[i] *= ratio_minus_right;
        state.f[i] = state.f_plus[i] - state.f_minus[i];
        state.y_hat[i] = state.f[i];
        state.residuals[i] = state.labels[i] - state.y_hat[i];
        state.r_tilde[i] = state.residuals[i];
    }

    state
}

/// Update backbone and tilt values for a resplit using two-tensor (v_b, Δd) conversion
fn reduce_grid_resplit_two_tensor(
    mut state: FittingState<'_>,
    v_b_l: f64,
    delta_d_l: f64,
    v_b_r: f64,
    delta_d_r: f64,
    col: usize,
    interval_idx: usize,
) -> FittingState<'_> {
    // Get parent parameters (for resplit, we update existing intervals)
    let b_p_left = state.backbone_values[col][interval_idx];
    let d_p_left = state.tilt_values[col][interval_idx];
    let b_p_right = state.backbone_values[col][interval_idx + 1];
    let d_p_right = state.tilt_values[col][interval_idx + 1];

    // Update left interval: b_L = b_p * v_b_L, d_L = d_p + Δd_L
    const MIN_BACKBONE: f64 = 1e-10;
    const MAX_BACKBONE: f64 = 1e10;
    const MAX_TILT: f64 = 10.0;
    state.backbone_values[col][interval_idx] = (b_p_left * v_b_l).clamp(MIN_BACKBONE, MAX_BACKBONE);
    state.tilt_values[col][interval_idx] = (d_p_left + delta_d_l).clamp(-MAX_TILT, MAX_TILT);

    // Update right interval: b_R = b_p * v_b_R, d_R = d_p + Δd_R
    state.backbone_values[col][interval_idx + 1] =
        (b_p_right * v_b_r).clamp(MIN_BACKBONE, MAX_BACKBONE);
    state.tilt_values[col][interval_idx + 1] = (d_p_right + delta_d_r).clamp(-MAX_TILT, MAX_TILT);

    // Stage 1 invariants: I21 (debug builds only)
    #[cfg(debug_assertions)]
    {
        if state.is_stage1_positive_only() {
            const EPS: f64 = 1e-10;
            // I21: all d_j = 0
            for (j, d_vec) in state.tilt_values.iter().enumerate() {
                for (k, &d) in d_vec.iter().enumerate() {
                    assert!(
                        d.abs() < EPS,
                        "I21 violation: tilt[{j}][{k}] should be 0 in Stage 1 mode, got {}",
                        d
                    );
                }
            }
        }
    }

    state
}

/// Merge=OptimalMerge: apply optimal merged parameters and remove boundary
///
/// # Arguments
/// * `b_merged_optimal` - Absolute optimal backbone value for merged interval
/// * `d_merged_optimal` - Absolute optimal tilt value for merged interval
fn reduce_grid_merge_optimal<'a>(
    mut state: FittingState<'a>,
    col: usize,
    interval_idx: usize,
    b_merged_optimal: f64,
    d_merged_optimal: f64,
) -> FittingState<'a> {
    // Remove the boundary inserted by the split
    state.boundaries[col].remove(interval_idx);

    // Apply merged params: set to absolute optimal values (not relative to left child)
    const MIN_BACKBONE: f64 = 1e-10;
    const MAX_BACKBONE: f64 = 1e10;
    const MAX_TILT: f64 = 10.0;
    state.backbone_values[col][interval_idx] = b_merged_optimal.clamp(MIN_BACKBONE, MAX_BACKBONE);
    state.tilt_values[col][interval_idx] = d_merged_optimal.clamp(-MAX_TILT, MAX_TILT);

    // Remove right child interval
    state.backbone_values[col].remove(interval_idx + 1);
    state.tilt_values[col].remove(interval_idx + 1);

    // Remove resplit/merge statistics aligned with the removed boundary
    state.precomputed_statistics.error_reductions_resplit[col].remove(interval_idx);
    state.precomputed_statistics.update_pairs_resplit_left[col].remove(interval_idx);
    state.precomputed_statistics.update_pairs_resplit_right[col].remove(interval_idx);
    state.precomputed_statistics.error_reductions_resplit_pairs[col].remove(interval_idx);
    state.precomputed_statistics.error_reductions_merge[col].remove(interval_idx);
    state.precomputed_statistics.update_pairs_merge[col].remove(interval_idx);

    // Remove right interval stats
    state.precomputed_statistics.interval_stats[col].remove(interval_idx + 1);

    state
}

/// Update predictions and residuals for affected points after merge using two-tensor per-axis factor ratios
/// Merges left and right intervals back into parent interval
fn reduce_grid_predictions_and_residuals_merge_two_tensor_factors<'a>(
    mut state: FittingState<'a>,
    col: usize,
    index: usize,
    original_affected_interval: &(usize, usize),
    g_plus_old_left: f64,
    g_minus_old_left: f64,
    g_plus_old_right: f64,
    g_minus_old_right: f64,
    g_plus_new: f64,
    g_minus_new: f64,
) -> FittingState<'a> {
    let &(start, end) = original_affected_interval;

    // Compute ratios: from children to parent
    // For left interval: ratio = g_new / g_old_left
    let ratio_plus_left = if g_plus_old_left.abs() > 1e-10 {
        g_plus_new / g_plus_old_left
    } else {
        1.0
    };
    let ratio_minus_left = if g_minus_old_left.abs() > 1e-10 {
        g_minus_new / g_minus_old_left
    } else {
        1.0
    };

    // For right interval: ratio = g_new / g_old_right
    let ratio_plus_right = if g_plus_old_right.abs() > 1e-10 {
        g_plus_new / g_plus_old_right
    } else {
        1.0
    };
    let ratio_minus_right = if g_minus_old_right.abs() > 1e-10 {
        g_minus_new / g_minus_old_right
    } else {
        1.0
    };

    let left_grid_points = &state.precomputed_statistics.sorted_indices[col][start..index];
    let right_grid_points = &state.precomputed_statistics.sorted_indices[col][index..end];

    // Update per-point caches for left interval (merge back to parent)
    for &i in left_grid_points {
        // Update f_plus, f_minus using ratios
        state.f_plus[i] *= ratio_plus_left;
        state.f_minus[i] *= ratio_minus_left;
        // Update f = f_plus - f_minus
        state.f[i] = state.f_plus[i] - state.f_minus[i];
        state.y_hat[i] = state.f[i];
        state.residuals[i] = state.labels[i] - state.y_hat[i];
        state.r_tilde[i] = state.residuals[i];
    }

    // Update per-point caches for right interval (merge back to parent)
    for &i in right_grid_points {
        state.f_plus[i] *= ratio_plus_right;
        state.f_minus[i] *= ratio_minus_right;
        state.f[i] = state.f_plus[i] - state.f_minus[i];
        state.y_hat[i] = state.f[i];
        state.residuals[i] = state.labels[i] - state.y_hat[i];
        state.r_tilde[i] = state.residuals[i];
    }

    state
}

/// Forbid splits around the given position (ported from SplitStrategy::forbid_around_split)
fn reduce_forbid_around_split<'a>(
    state: FittingState<'a>,
    split_candidate: &SplitCandidate,
    split_strategy: &SplitStrategy,
) -> FittingState<'a> {
    split_strategy.forbid_around_split(state, split_candidate)
}

/// Wrapper: update statistics and refresh ER caches for affected ranges
fn reduce_grid_statistics_and_refresh_caches<'a>(
    mut state: FittingState<'a>,
    col: usize,
    original_affected_interval: &(usize, usize),
    affected_ranges: &[AffectedRange],
    refinement: &RefinementStrategy,
) -> FittingState<'a> {
    let &(start, end) = original_affected_interval;
    // Convert AffectedRange to point ranges for update_statistics
    // update_statistics expects a Vec indexed by column number, so we need to build
    // a full vector with empty ranges (1, 0) for unaffected columns
    let mut affected_points_range = vec![(1usize, 0usize); state.p];
    for affected_range in affected_ranges {
        affected_points_range[affected_range.col] = affected_range.point_range;
    }
    state = refinement.update_statistics(state, col, start, end, &affected_points_range);
    state = refinement.refresh_error_reduction_caches_for_affected(state, affected_ranges);
    state
}

fn reduce_loop_state_for_split<'a>(
    mut state: FittingState<'a>,
    split_candidate: &SplitCandidate,
) -> FittingState<'a> {
    state.loop_state.split_count += 1;
    state.loop_state.fineness += 1;
    state.loop_state.consecutive_resplits = 0;
    state.current_error -= split_candidate.error_reduction;
    state
}

fn reduce_loop_state_for_resplit<'a>(
    mut state: FittingState<'a>,
    resplit_candidate: &ResplitCandidate,
) -> FittingState<'a> {
    state.loop_state.resplit_count += 1;
    state.loop_state.consecutive_resplits += 1;
    state.current_error -= resplit_candidate.error_reduction;
    state
}

fn reduce_loop_state_for_merge<'a>(
    mut state: FittingState<'a>,
    _merge_candidate: &MergeCandidate,
) -> FittingState<'a> {
    state.loop_state.merge_count += 1;
    state.loop_state.fineness = state.loop_state.fineness.saturating_sub(1);
    state.loop_state.consecutive_resplits = 0;
    // Note: current_error is recomputed from residuals in the merge reducer above
    // (not using merge_candidate.error_reduction because we restore parent params, not optimal merged params)
    state
}

/// Terminate the fitting process
fn reduce_terminate(mut state: FittingState<'_>, reason: String) -> FittingState<'_> {
    // For now, just return the state
    // In the future, we could set a termination flag
    // The reason is consumed but not currently used
    log::debug!("Terminating; reason='{}'", reason);
    state.terminated = true;
    state
}

/// Apply multiple actions in sequence
fn reduce_composite<'a>(
    mut state: FittingState<'a>,
    actions: Vec<FittingAction>,
    refinement_strategy: &RefinementStrategy,
    split_strategy: &SplitStrategy,
) -> FittingState<'a> {
    for action in actions {
        state = fitting_reducer(state, action, refinement_strategy, split_strategy);
    }
    state
}

/// Multiply the model's global scaling by a factor
fn reduce_apply_scaling(mut state: FittingState<'_>, factor: f64) -> FittingState<'_> {
    log::debug!(
        "Applying scaling factor: {} (old scaling: {})",
        factor,
        state.scaling
    );
    state.scaling *= factor;
    state.y_hat *= factor;
    state.residuals = &state.labels - &state.y_hat;
    state
}

// ============================================================================
// Helper functions for merge/resplit using axis parameters and partial products
// ============================================================================

/// Compute partial products g_{\pm}^{(-j)} for all points by dividing out axis j factor.
///
/// Uses the efficient approach: g_{+,i}^{(-j)} = f_{+,i} / a_{+,j}^{k_j(i)}
/// where a_{+,j}^{k_j(i)} = b_{j,i} * exp(d_{j,i})
///
/// # Arguments
/// * `col` - The axis index j
/// * `state` - The current fitting state
///
/// # Returns
/// `(g_plus, g_minus)` vectors of length n, where:
/// - `g_plus[i] = f_{+,i} / (b_{j,i} * exp(d_{j,i}))`
/// - `g_minus[i] = f_{-,i} / (b_{j,i} * exp(-d_{j,i}))`
pub(crate) fn compute_partial_products_for_axis(
    col: usize,
    state: &FittingState,
) -> (Vec<f64>, Vec<f64>) {
    let n = state.n;
    let mut g_plus = Vec::with_capacity(n);
    let mut g_minus = Vec::with_capacity(n);

    for i in 0..n {
        // Get current interval index for point i on axis col
        let interval_idx = state.interval_id[col][i];

        // Get axis parameters for this interval
        let b_j = state.backbone_values[col][interval_idx];
        let d_j = state.tilt_values[col][interval_idx];

        // Compute axis factors: a_{+,j}^{k_j(i)} = b_j * exp(d_j), a_{-,j}^{k_j(i)} = b_j * exp(-d_j)
        let a_plus_j = b_j * d_j.exp();
        let a_minus_j = b_j * (-d_j).exp();

        // Divide out axis j factor from cached f_plus and f_minus
        // g_{+,i}^{(-j)} = f_{+,i} / a_{+,j}^{k_j(i)}
        // g_{-,i}^{(-j)} = f_{-,i} / a_{-,j}^{k_j(i)}
        g_plus.push(state.f_plus[i] / a_plus_j);
        g_minus.push(state.f_minus[i] / a_minus_j);
    }

    (g_plus, g_minus)
}

/// Compute sufficient statistics for a region using partial products as regressors.
///
/// Uses g_{\pm}^{(-j)} as regressors instead of f_{\pm}, which is appropriate
/// for merge/resplit operations on axis j.
///
/// # Arguments
/// * `_col` - The axis index j (currently unused, kept for future use with weights)
/// * `region` - Indices of points in the region (sorted by x_{i,j})
/// * `g_plus` - Partial products g_{+}^{(-j)} for all points
/// * `g_minus` - Partial products g_{-}^{(-j)} for all points
/// * `state` - The current fitting state
///
/// # Returns
/// `IntervalStats` with stats computed using partial product regressors:
/// - S_{11} = sum_{i in region} w_i * (g_{+,i}^{(-j)})^2
/// - S_{22} = sum_{i in region} w_i * (g_{-,i}^{(-j)})^2
/// - S_{12} = -sum_{i in region} w_i * g_{+,i}^{(-j)} * g_{-,i}^{(-j)}
/// - t_1 = sum_{i in region} w_i * r_tilde_i * g_{+,i}^{(-j)}
/// - t_2 = -sum_{i in region} w_i * r_tilde_i * g_{-,i}^{(-j)}
pub(crate) fn compute_stats_using_partial_products(
    _col: usize,
    region: &[usize],
    g_plus: &[f64],
    g_minus: &[f64],
    state: &FittingState,
) -> crate::grid_tensor::state::IntervalStats {
    use crate::grid_tensor::state::IntervalStats;

    let mut sum_s11 = 0.0;
    let mut sum_s22 = 0.0;
    let mut sum_s12 = 0.0;
    let mut sum_t1 = 0.0;
    let mut sum_t2 = 0.0;

    // For now, assume uniform weights (w_i = 1.0)
    // TODO: Support non-uniform weights if needed
    for &i in region {
        let g_p = g_plus[i];
        let g_m = g_minus[i];
        let r_tilde = state.r_tilde[i];

        // S_{11} = sum w_i * (g_{+,i}^{(-j)})^2
        sum_s11 += g_p * g_p;

        // S_{22} = sum w_i * (g_{-,i}^{(-j)})^2
        sum_s22 += g_m * g_m;

        // S_{12} = -sum w_i * g_{+,i}^{(-j)} * g_{-,i}^{(-j)}
        sum_s12 -= g_p * g_m;

        // t_1 = sum w_i * r_tilde_i * g_{+,i}^{(-j)}
        sum_t1 += r_tilde * g_p;

        // t_2 = -sum w_i * r_tilde_i * g_{-,i}^{(-j)}
        sum_t2 -= r_tilde * g_m;
    }

    IntervalStats {
        sum_s11,
        sum_s22,
        sum_s12,
        sum_t1,
        sum_t2,
        n: region.len(),
    }
}

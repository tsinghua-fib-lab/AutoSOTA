use std::sync::mpsc::Sender;

use crate::grid_tensor::{action::FittingAction, state::FittingState};
use crate::logging::types::{
    Action, ComponentStateSnapshot, FComponentStats, GridError, LoggingMessage, SplitEvent, Update,
};
use ndarray::Array1;

/// Encode f64 array to binary blob
pub fn encode_f64_array(values: &[f64]) -> Vec<u8> {
    unsafe {
        let len = std::mem::size_of_val(values);
        let ptr = values.as_ptr() as *const u8;
        std::slice::from_raw_parts(ptr, len).to_vec()
    }
}

/// Decode binary blob to f64 array
pub fn decode_f64_array(data: &[u8]) -> Vec<f64> {
    unsafe {
        let len = data.len() / std::mem::size_of::<f64>();
        let ptr = data.as_ptr() as *const f64;
        std::slice::from_raw_parts(ptr, len).to_vec()
    }
}

/// Compute percentile using linear interpolation (standard method)
fn percentile(array: &Array1<f64>, p: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&p), "Percentile must be in [0, 1]");

    if array.is_empty() {
        return f64::NAN;
    }

    if array.len() == 1 {
        return array[0];
    }

    // Create sorted copy
    let mut sorted: Vec<f64> = array.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Linear interpolation method
    let n = sorted.len();
    let position = p * (n - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
    }
}

/// Compute statistics for f+ and f- components
pub fn compute_f_component_stats(
    f_plus: &Array1<f64>,
    f_minus: &Array1<f64>,
) -> (FComponentStats, FComponentStats) {
    debug_assert_eq!(
        f_plus.len(),
        f_minus.len(),
        "f_plus and f_minus must have same length"
    );
    debug_assert!(
        f_plus.iter().all(|&x| x >= 0.0),
        "f_plus must be non-negative (I21)"
    );
    debug_assert!(
        f_minus.iter().all(|&x| x >= 0.0),
        "f_minus must be non-negative (I21)"
    );

    // Compute statistics for f_plus
    let mean_plus = f_plus.mean().unwrap_or(0.0);
    let variance_plus =
        f_plus.iter().map(|&x| (x - mean_plus).powi(2)).sum::<f64>() / f_plus.len() as f64;
    let std_plus = variance_plus.sqrt();

    let stats_plus = FComponentStats {
        min: f_plus.iter().copied().fold(f64::INFINITY, f64::min),
        max: f_plus.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        mean: mean_plus,
        std: std_plus,
        p25: percentile(f_plus, 0.25),
        p50: percentile(f_plus, 0.50),
        p75: percentile(f_plus, 0.75),
        p95: percentile(f_plus, 0.95),
        p99: percentile(f_plus, 0.99),
        n_samples: f_plus.len(),
    };

    // Compute statistics for f_minus
    let mean_minus = f_minus.mean().unwrap_or(0.0);
    let variance_minus = f_minus
        .iter()
        .map(|&x| (x - mean_minus).powi(2))
        .sum::<f64>()
        / f_minus.len() as f64;
    let std_minus = variance_minus.sqrt();

    let stats_minus = FComponentStats {
        min: f_minus.iter().copied().fold(f64::INFINITY, f64::min),
        max: f_minus.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        mean: mean_minus,
        std: std_minus,
        p25: percentile(f_minus, 0.25),
        p50: percentile(f_minus, 0.50),
        p75: percentile(f_minus, 0.75),
        p95: percentile(f_minus, 0.95),
        p99: percentile(f_minus, 0.99),
        n_samples: f_minus.len(),
    };

    (stats_plus, stats_minus)
}

pub fn create_raw_split_event(
    state: &FittingState,
    action: &FittingAction,
    old_error: f64,
) -> RawLoggingEvent {
    let split_event = create_simple_split_event(state, action, old_error);
    RawLoggingEvent::Split(split_event)
}

pub fn create_simple_split_event(
    state: &FittingState,
    action: &FittingAction,
    old_error: f64,
) -> SplitEvent {
    let (
        action_type,
        col,
        interval_idx,
        update_a,
        update_b,
        error_reduction,
        split_value,
        left_count,
        right_count,
    ) = match action {
        FittingAction::ApplySplit { split } => {
            // For splits, we need to compute the actual split value and counts
            let col = split.col;
            let split_value = state.x[[
                state.precomputed_statistics.sorted_indices[col][split.index],
                col,
            ]];

            // Find the interval index where the split was applied
            // After split, the split index should be at the boundary position
            let interval_idx = state.boundaries[col]
                .iter()
                .position(|&b| b == split.index)
                .unwrap_or_else(|| {
                    // Fallback: find the interval that contains the split point
                    state.boundaries[col].partition_point(|&b| b <= split.index)
                });

            let (start, _, end) = state.interval_range_left_and_right(col, interval_idx);
            let left_count = split.index.saturating_sub(start);
            let right_count = end.saturating_sub(split.index);

            // Use actual state values after split is applied
            // For splits, the left and right values should be at interval_idx and interval_idx + 1
            // Derive grid_values from backbone/tilt for logging
            let update_a = state.backbone_values[col][interval_idx]
                * state.tilt_values[col][interval_idx].cosh();
            let update_b = if interval_idx + 1 < state.backbone_values[col].len() {
                state.backbone_values[col][interval_idx + 1]
                    * state.tilt_values[col][interval_idx + 1].cosh()
            } else {
                // This shouldn't happen for splits, but let's be safe
                0.0
            };

            (
                Action::Split,
                split.col,
                interval_idx,
                update_a,
                update_b,
                split.error_reduction,
                Some(split_value),
                Some(left_count),
                Some(right_count),
            )
        }
        FittingAction::ApplyMerge { merge } => {
            // For merges, we need to compute the actual split value and counts
            let col = merge.col;
            let split_value = state.x[[
                state.precomputed_statistics.sorted_indices[col][merge.index],
                col,
            ]];
            let (start, _, end) = state.interval_range_left_and_right(col, merge.interval_idx);
            let left_count = merge.index.saturating_sub(start);
            let right_count = end.saturating_sub(merge.index);

            // Use actual state values after merge is applied (derive from backbone/tilt)
            let update_a = state.backbone_values[col][merge.interval_idx]
                * state.tilt_values[col][merge.interval_idx].cosh();
            let update_b = 0.0; // Dummy value for merge operations

            (
                Action::Merge,
                merge.col,
                merge.interval_idx,
                update_a,
                update_b,
                merge.error_reduction,
                Some(split_value),
                Some(left_count),
                Some(right_count),
            )
        }
        FittingAction::ApplyResplit { resplit } => {
            // For resplits, we need to compute the actual split value and counts
            let col = resplit.col;
            let split_value = state.x[[
                state.precomputed_statistics.sorted_indices[col][resplit.index],
                col,
            ]];
            let (start, _, end) = state.interval_range_left_and_right(col, resplit.interval_idx);
            let left_count = resplit.index.saturating_sub(start);
            let right_count = end.saturating_sub(resplit.index);

            // Use actual state values after resplit is applied (derive from backbone/tilt)
            let update_a = state.backbone_values[col][resplit.interval_idx]
                * state.tilt_values[col][resplit.interval_idx].cosh();
            let update_b = if resplit.interval_idx + 1 < state.backbone_values[col].len() {
                state.backbone_values[col][resplit.interval_idx + 1]
                    * state.tilt_values[col][resplit.interval_idx + 1].cosh()
            } else {
                // This shouldn't happen for resplits, but let's be safe
                0.0
            };

            (
                Action::Resplit,
                resplit.col,
                resplit.interval_idx,
                update_a,
                update_b,
                resplit.error_reduction,
                Some(split_value),
                Some(left_count),
                Some(right_count),
            )
        }
        _ => unreachable!(),
    };

    // Compute n_cells_after (total number of cells after the operation)
    // Use backbone_values instead of grid_values (same length)
    let n_cells_after: usize = state
        .backbone_values
        .iter()
        .map(|col_values| col_values.len())
        .product();

    // Create residual updates for affected samples
    let mut residual_updates = Vec::new();
    let (start, _, end) = state.interval_range_left_and_right(col, interval_idx);
    for &sample_id in &state.precomputed_statistics.sorted_indices[col][start..end] {
        residual_updates.push(Update {
            sample_id: sample_id as u32,
            residual: state.residuals[sample_id] as f32,
            y_hat: Some(state.y_hat[sample_id] as f32),
        });
    }

    SplitEvent {
        iter_no: state.iteration,
        action: action_type,
        col,
        left_interval_idx: interval_idx,
        split_value,
        update_a: Some(update_a),
        update_b: if action_type == Action::Merge {
            None
        } else {
            Some(update_b)
        },
        left_count,
        right_count,
        gain: Some(error_reduction),
        err_before: Some(old_error),
        err_after: Some(state.current_error),
        n_cells_after,
        residual_updates,
    }
}

/// Internal helper to create component snapshot from intervals and grid values
#[cfg(feature = "evo-logging")]
pub fn create_component_snapshot_from_data(
    intervals: &[(f64, f64)],
    grid_values: &[f64],
    col: usize,
    iter_no: usize,
) -> ComponentStateSnapshot {
    let intervals_count = intervals.len();

    // Handle case where there are no intervals
    if intervals_count == 0 {
        return ComponentStateSnapshot {
            iter_no,
            col,
            intervals_count: 1,
            data: vec![], // Empty data for initial state
            backbone_data: None,
            tilt_data: None,
            lambda_plus: None,
            lambda_minus: None,
        };
    }

    // Build lossless [starts (N), ends (N), values (N)] as f64 and copy to bytes
    // This matches the legacy grid system encoding exactly
    let mut payload: Vec<f64> = Vec::with_capacity(intervals_count * 3);

    // starts - from intervals
    for (start, _) in intervals {
        payload.push(*start);
    }

    // ends - from intervals
    for (_, end) in intervals {
        payload.push(*end);
    }

    // values - the actual grid values
    payload.extend(grid_values.iter().copied());

    // Convert to raw bytes (not JSON!) - matches legacy system
    let data: Vec<u8> = unsafe {
        let len = payload.len() * std::mem::size_of::<f64>();
        let ptr = payload.as_ptr() as *const u8;
        std::slice::from_raw_parts(ptr, len).to_vec()
    };

    ComponentStateSnapshot {
        iter_no,
        col,
        intervals_count,
        data,
        backbone_data: None,
        tilt_data: None,
        lambda_plus: None,
        lambda_minus: None,
    }
}

#[cfg(feature = "evo-logging")]
pub fn create_raw_component_event(
    state: &FittingState,
    col: usize,
    iter_no: usize,
) -> RawLoggingEvent {
    let intervals_count = state.boundaries[col].len() + 1;

    // Handle case where there are no boundaries yet (initial state)
    if state.boundaries[col].is_empty() {
        let snapshot = ComponentStateSnapshot {
            iter_no,
            col,
            intervals_count: 1,
            data: vec![], // Empty data for initial state
            backbone_data: Some(encode_f64_array(&state.backbone_values[col])),
            tilt_data: Some(encode_f64_array(&state.tilt_values[col])),
            lambda_plus: Some(state.lambda_plus),
            lambda_minus: Some(state.lambda_minus),
        };
        return RawLoggingEvent::Component(snapshot);
    }

    // Compute intervals from state (same logic as in convert_state_to_result)
    let mut intervals = Vec::with_capacity(intervals_count);
    let mut prev = 0usize;
    for &b in &state.boundaries[col] {
        let left = if prev == 0 {
            f64::NEG_INFINITY
        } else {
            state.x[[state.precomputed_statistics.sorted_indices[col][prev], col]]
        };
        let right = if b == state.precomputed_statistics.sorted_indices[col].len() {
            f64::INFINITY
        } else {
            state.x[[state.precomputed_statistics.sorted_indices[col][b], col]]
        };
        intervals.push((left, right));
        prev = b;
    }
    // Add final interval
    let left = if prev == 0 {
        f64::NEG_INFINITY
    } else {
        state.x[[state.precomputed_statistics.sorted_indices[col][prev], col]]
    };
    intervals.push((left, f64::INFINITY));

    // Derive grid_values from backbone/tilt for snapshot (backward compatibility)
    let grid_values: Vec<f64> = state.backbone_values[col]
        .iter()
        .zip(state.tilt_values[col].iter())
        .map(|(b, d)| b * d.cosh())
        .collect();
    let mut snapshot = create_component_snapshot_from_data(&intervals, &grid_values, col, iter_no);

    // Populate two-tensor fields
    snapshot.backbone_data = Some(encode_f64_array(&state.backbone_values[col]));
    snapshot.tilt_data = Some(encode_f64_array(&state.tilt_values[col]));
    snapshot.lambda_plus = Some(state.lambda_plus);
    snapshot.lambda_minus = Some(state.lambda_minus);

    // Assert invariants (debug builds)
    #[cfg(debug_assertions)]
    {
        let backbone_decoded = decode_f64_array(snapshot.backbone_data.as_ref().unwrap());
        for &b in &backbone_decoded {
            debug_assert!(b > 0.0, "Backbone must be positive (I20)");
        }
        debug_assert!(
            snapshot.lambda_plus.unwrap() > 0.0,
            "lambda_plus must be positive"
        );
        // λ_- = 0 is legal in Stage 1 positive-only mode (spec I20); otherwise > 0.
        debug_assert!(
            snapshot.lambda_minus.unwrap() >= 0.0,
            "lambda_minus must be non-negative"
        );
    }

    RawLoggingEvent::Component(snapshot)
}

/// Helper function to wrap raw events with epoch/tree_id context (called at family level)
///
/// This function converts raw logging events (without epoch/tree_id) into full LoggingMessage
/// events that can be persisted to the SQLite database.
#[cfg(feature = "evo-logging")]
pub fn wrap_raw_events_with_context(
    raw_events: Vec<RawLoggingEvent>,
    epoch: usize,
    tree_id: usize,
) -> Vec<LoggingMessage> {
    raw_events
        .into_iter()
        .map(|raw_event| match raw_event {
            RawLoggingEvent::Split(split_event) => LoggingMessage::Split {
                epoch,
                tree_id,
                event: split_event,
            },
            RawLoggingEvent::Component(snapshot) => LoggingMessage::Component {
                epoch,
                tree_id,
                snapshot,
            },
            RawLoggingEvent::GridError(grid_error) => LoggingMessage::GridErrCombined {
                epoch,
                err: grid_error.err,
                variant: grid_error.variant,
            },
        })
        .collect()
}

// Raw event types without epoch/tree_id - these get wrapped in LoggingMessage at family level
#[derive(Debug, Clone)]
pub enum RawLoggingEvent {
    Split(SplitEvent),
    Component(ComponentStateSnapshot),
    GridError(GridError),
}

// Buffered events collected during fitting
#[derive(Debug)]
pub struct LoggingState {
    /// Raw events collected (without epoch/tree_id context)
    pub events: Vec<RawLoggingEvent>,
    /// f+/f- component statistics per iteration
    pub f_component_stats: Vec<(usize, FComponentStats, FComponentStats)>,
    /// Optional: Event channel sender (for parallel contexts)
    /// When Some, events are sent immediately through channel
    /// When None, events are buffered in `events` for later wrapping
    pub event_sender: Option<Sender<LoggingMessage>>,
}

impl Default for LoggingState {
    fn default() -> Self {
        Self::new()
    }
}

impl LoggingState {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            f_component_stats: Vec::new(),
            event_sender: None,
        }
    }

    /// Create with event sender (for parallel contexts)
    pub fn with_event_sender(sender: Sender<LoggingMessage>) -> Self {
        Self {
            events: Vec::new(),
            f_component_stats: Vec::new(),
            event_sender: Some(sender),
        }
    }
}

/// Log final component state snapshots after identification
/// Always callable - feature flag is encapsulated inside
/// Consumes logging_state and sends all buffered events
/// If logging_state is None, does nothing (no-op)
pub fn log_final_component_states(
    logging_state: Option<LoggingState>,
    grid_tensor: &crate::grid_tensor::GridTensor,
    final_iteration: usize,
) {
    #[cfg(feature = "evo-logging")]
    {
        // If logging disabled, return early (no-op)
        let mut logging_state = match logging_state {
            Some(state) => state,
            None => return,
        };
        use crate::logging::types::LoggingMessage;
        use crate::logging::{current_epoch, current_tree_id, try_send_event};

        // Buffer final component state snapshots
        let mean_factor = grid_tensor.get_mean_factor();
        for col in 0..mean_factor.len() {
            let mut snapshot = create_component_snapshot_from_data(
                &grid_tensor.intervals[col],
                &mean_factor[col],
                col,
                final_iteration,
            );

            // Populate two-tensor fields and identified lambdas so the final snapshot
            // reflects the post-identification state.
            snapshot.backbone_data = Some(encode_f64_array(&grid_tensor.backbone_values[col]));
            snapshot.tilt_data = Some(encode_f64_array(&grid_tensor.tilt_values[col]));
            snapshot.lambda_plus = Some(grid_tensor.lambda_plus);
            snapshot.lambda_minus = Some(grid_tensor.lambda_minus);

            // Always buffer for later
            logging_state
                .events
                .push(RawLoggingEvent::Component(snapshot));
        }

        // Wrap all buffered events with context and send them
        // Check context before wrapping - this should always be set at the family level
        let epoch = match current_epoch() {
            Some(e) => e,
            None => {
                log::warn!("Logging events dropped: epoch context not set");
                return;
            }
        };
        let tree_id = match current_tree_id() {
            Some(t) => t,
            None => {
                log::warn!("Logging events dropped: tree_id context not set");
                return;
            }
        };
        let events = wrap_raw_events_with_context(logging_state.events, epoch, tree_id);
        for event in events {
            if !try_send_event(event) {
                log::warn!("Failed to send logging event (channel may be closed)");
            }
        }

        // Send f_component_stats with proper context checking
        for (iter_no, stats_plus, stats_minus) in logging_state.f_component_stats {
            if !try_send_event(LoggingMessage::FComponentStats {
                epoch,
                tree_id,
                iter_no,
                stats_plus,
                stats_minus,
            }) {
                log::warn!("Failed to send logging event (channel may be closed)");
            }
        }
    }
    #[cfg(not(feature = "evo-logging"))]
    {
        // No-op when feature disabled
        let _ = (logging_state, grid_tensor, final_iteration);
    }
}

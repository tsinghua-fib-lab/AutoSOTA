use crate::grid_tensor::logging_helpers::LoggingState;
use crate::grid_tensor::state::FittingState;
use crate::logging::types::{GridErrorVariant, SplitEvent};

/// Log f+/f- component statistics reducer
///
/// Takes optional logging state and fitting state, computes stats internally,
/// and returns updated logging state. If logging disabled, returns None (no-op).
pub fn log_f_component_reducer(
    logging_state: Option<LoggingState>,
    state: &FittingState,
    iter_no: usize,
) -> Option<LoggingState> {
    // If logging disabled, return None (no-op)
    let mut log_state = logging_state?;

    // Compute statistics inside reducer
    use crate::grid_tensor::logging_helpers::compute_f_component_stats;
    let (stats_plus, stats_minus) = compute_f_component_stats(&state.f_plus, &state.f_minus);

    // Always buffer for later
    log_state
        .f_component_stats
        .push((iter_no, stats_plus, stats_minus));
    Some(log_state)
}

/// Log component state snapshots reducer
///
/// Takes optional logging state and fitting state, creates snapshots for all columns
/// internally, and returns updated logging state. If logging disabled, returns None (no-op).
pub fn log_component_reducer(
    logging_state: Option<LoggingState>,
    state: &FittingState,
    iter_no: usize,
) -> Option<LoggingState> {
    // If logging disabled, return None (no-op)
    let mut log_state = logging_state?;

    // Loop through all columns and create snapshots
    use crate::grid_tensor::logging_helpers::create_raw_component_event;
    for col in 0..state.p {
        let raw_event = create_raw_component_event(state, col, iter_no);
        // Always buffer for later
        log_state.events.push(raw_event);
    }
    Some(log_state)
}

/// Log split/resplit/merge event reducer
///
/// Takes optional logging state and event, returns updated logging state.
/// If logging disabled, returns None (no-op).
pub fn log_split_reducer(
    logging_state: Option<LoggingState>,
    event: SplitEvent,
) -> Option<LoggingState> {
    // If logging disabled, return None (no-op)
    let mut log_state = logging_state?;

    // Always buffer for later (will be wrapped with context at family level)
    use crate::grid_tensor::logging_helpers::RawLoggingEvent;
    log_state.events.push(RawLoggingEvent::Split(event));
    Some(log_state)
}

/// Log grid error reducer
///
/// Takes optional logging state, error, and variant, returns updated logging state.
/// If logging disabled, returns None (no-op).
pub fn log_grid_error_reducer(
    logging_state: Option<LoggingState>,
    _tree_id: Option<usize>,
    err: f64,
    variant: GridErrorVariant,
) -> Option<LoggingState> {
    // If logging disabled, return None (no-op)
    let mut log_state = logging_state?;

    // Always buffer grid errors for later
    use crate::grid_tensor::logging_helpers::RawLoggingEvent;
    use crate::logging::types::GridError;
    log_state
        .events
        .push(RawLoggingEvent::GridError(GridError { err, variant }));
    Some(log_state)
}

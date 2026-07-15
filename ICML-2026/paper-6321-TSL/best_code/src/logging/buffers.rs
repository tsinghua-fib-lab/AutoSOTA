use crate::logging::types::{
    CombinedGridSnapshot, ComponentStateSnapshot, EpochScalingSnapshot, GridErrorVariant,
    LoggingMessage, SplitEvent,
};

use super::context::{current_epoch, current_tree_id};
use super::manager::{is_logging_enabled, try_send_event};

/// Helper function for messages that need both epoch and tree_id
pub(crate) fn send_with_tree_id<F>(f: F)
where
    F: FnOnce(usize, usize) -> LoggingMessage,
{
    // Skip if logging is not enabled (no database path configured)
    if !is_logging_enabled() {
        return;
    }
    let epoch = match current_epoch() {
        Some(e) => e,
        None => {
            log::warn!("Logging event dropped: epoch context not set");
            return;
        }
    };
    let tree_id = match current_tree_id() {
        Some(t) => t,
        None => {
            log::warn!("Logging event dropped: tree_id context not set");
            return;
        }
    };
    if !try_send_event(f(epoch, tree_id)) {
        log::warn!("Failed to send logging event (channel may be closed)");
    }
}

/// Helper function for messages that only need epoch
fn send_with_epoch<F>(f: F)
where
    F: FnOnce(usize) -> LoggingMessage,
{
    // Skip if logging is not enabled (no database path configured)
    if !is_logging_enabled() {
        return;
    }
    let epoch = match current_epoch() {
        Some(e) => e,
        None => {
            log::warn!("Logging event dropped: epoch context not set");
            return;
        }
    };
    if !try_send_event(f(epoch)) {
        log::warn!("Failed to send logging event (channel may be closed)");
    }
}

/// Helper function for messages that need no context (snapshots already have epoch)
fn send_message(msg: LoggingMessage) {
    // Skip if logging is not enabled (no database path configured)
    if !is_logging_enabled() {
        return;
    }
    if !try_send_event(msg) {
        log::warn!("Failed to send logging event (channel may be closed)");
    }
}

pub fn log_split_event(event: SplitEvent) {
    send_with_tree_id(|epoch, tree_id| LoggingMessage::Split {
        epoch,
        tree_id,
        event,
    });
}

pub fn log_combined_grid(snapshot: CombinedGridSnapshot) {
    // No epoch needed - snapshot already has it
    send_message(LoggingMessage::CombinedGrid { snapshot });
}

pub fn log_epoch_scaling(snapshot: EpochScalingSnapshot) {
    // No epoch needed - snapshot already has it
    send_message(LoggingMessage::EpochScaling { snapshot });
}

pub fn log_component_state(snapshot: ComponentStateSnapshot) {
    send_with_tree_id(|epoch, tree_id| LoggingMessage::Component {
        epoch,
        tree_id,
        snapshot,
    });
}

pub fn log_grid_error_combined(err: f64, variant: GridErrorVariant) {
    send_with_epoch(|epoch| LoggingMessage::GridErrCombined {
        epoch,
        err,
        variant,
    });
}

pub fn log_grid_error_fitted(err: f64, variant: GridErrorVariant) {
    send_with_tree_id(|epoch, tree_id| LoggingMessage::GridErrFitted {
        epoch,
        tree_id,
        err,
        variant,
    });
}

pub fn log_combination_choice(
    method: &str,
    best_index: Option<usize>,
    candidates: &Vec<(usize, f64)>,
) {
    send_with_epoch(|epoch| LoggingMessage::CombinationChoice {
        epoch,
        method: method.to_string(),
        best_index,
        candidate_indices: candidates.clone(),
    });
}

use std::sync::Mutex;

use crate::logging::types::LoggingMessage;

use once_cell::sync::Lazy;

// Global unified event sender for all logging (main thread and worker threads).
// The receiver lives on the training thread and drains events before flush.
// Use an unbounded channel to avoid dropping events under load.
static EVENT_SENDER: Lazy<Mutex<Option<std::sync::mpsc::Sender<LoggingMessage>>>> =
    Lazy::new(|| Mutex::new(None));

pub fn set_event_sender(sender: std::sync::mpsc::Sender<LoggingMessage>) {
    let mut guard = EVENT_SENDER.lock().expect("lock poisoned");
    *guard = Some(sender);
}

pub fn clear_event_sender() {
    let mut guard = EVENT_SENDER.lock().expect("lock poisoned");
    *guard = None;
}

pub fn is_logging_enabled() -> bool {
    let guard = EVENT_SENDER.lock().expect("lock poisoned");
    guard.is_some()
}

pub fn try_send_event(event: LoggingMessage) -> bool {
    let guard = EVENT_SENDER.lock().expect("lock poisoned");
    if let Some(sender) = &*guard {
        // Unbounded channel: send never blocks; returns false only if receiver is dropped
        sender.send(event).is_ok()
    } else {
        false
    }
}

#[cfg(feature = "evo-logging")]
pub fn drain_logging_events(event_channel: &crate::logging::EventChannel) {
    let (rx, logger_rc) = event_channel;
    while let Ok(msg) = rx.try_recv() {
        match msg {
            LoggingMessage::Split {
                epoch,
                tree_id,
                event,
            } => {
                logger_rc
                    .borrow_mut()
                    .push_split_event(epoch, tree_id, event);
            }
            LoggingMessage::Component {
                epoch,
                tree_id,
                snapshot,
            } => {
                logger_rc
                    .borrow_mut()
                    .push_component_state(epoch, tree_id, snapshot);
            }
            LoggingMessage::GridErrCombined {
                epoch,
                err,
                variant,
            } => {
                logger_rc
                    .borrow_mut()
                    .push_grid_error_combined(epoch, err, variant);
            }
            LoggingMessage::GridErrFitted {
                epoch,
                tree_id,
                err,
                variant,
            } => {
                logger_rc
                    .borrow_mut()
                    .push_grid_error_fitted(epoch, tree_id, err, variant);
            }
            LoggingMessage::FComponentStats {
                epoch,
                tree_id,
                iter_no,
                stats_plus,
                stats_minus,
            } => {
                logger_rc.borrow_mut().push_f_component_stats(
                    epoch,
                    tree_id,
                    iter_no,
                    stats_plus,
                    stats_minus,
                );
            }
            LoggingMessage::CombinationChoice {
                epoch,
                method,
                best_index,
                candidate_indices,
            } => {
                logger_rc.borrow_mut().push_combination_choice(
                    epoch,
                    &method,
                    best_index,
                    candidate_indices,
                );
            }
            LoggingMessage::CombinedGrid { snapshot } => {
                // snapshot.epoch already has the epoch
                logger_rc.borrow_mut().push_combined_grid(snapshot);
            }
            LoggingMessage::EpochScaling { snapshot } => {
                // snapshot.epoch already has the epoch
                logger_rc.borrow_mut().push_epoch_scaling(snapshot);
            }
        }
    }
}

#[cfg(not(feature = "evo-logging"))]
pub fn drain_logging_events(_event_channel: &crate::logging::EventChannel) {
    // No-op when evo-logging is disabled
}

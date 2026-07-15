// Root of logging module. Split into submodules under `src/logging/`.

#[cfg(feature = "evo-logging")]
mod buffers;
mod context;
mod evo;
mod manager;
#[cfg(feature = "evo-logging")]
pub mod reducer;
mod simple;
#[cfg(feature = "evo-logging")]
mod sqlite_logger;
pub mod types;

// Type alias for event channel (receiver + logger)
#[cfg(feature = "evo-logging")]
pub type EventChannel = (
    std::sync::mpsc::Receiver<LoggingMessage>,
    std::rc::Rc<std::cell::RefCell<Box<dyn EvoLogger>>>,
);

// Stub type when evo-logging is disabled (never actually constructed)
#[cfg(not(feature = "evo-logging"))]
pub type EventChannel = ();

// Public re-exports to preserve crate::logging::* API
#[cfg(feature = "evo-logging")]
pub use buffers::{
    log_combination_choice, log_combined_grid, log_component_state, log_epoch_scaling,
    log_grid_error_combined, log_grid_error_fitted, log_split_event,
};

// No-op stubs when evo-logging is disabled
#[cfg(not(feature = "evo-logging"))]
pub fn log_split_event(_event: SplitEvent) {}

#[cfg(not(feature = "evo-logging"))]
pub fn log_combined_grid(_snapshot: CombinedGridSnapshot) {}

#[cfg(not(feature = "evo-logging"))]
pub fn log_epoch_scaling(_snapshot: EpochScalingSnapshot) {}

#[cfg(not(feature = "evo-logging"))]
pub fn log_component_state(_snapshot: ComponentStateSnapshot) {}

#[cfg(not(feature = "evo-logging"))]
pub fn log_grid_error_combined(_err: f64, _variant: GridErrorVariant) {}

#[cfg(not(feature = "evo-logging"))]
pub fn log_grid_error_fitted(_err: f64, _variant: GridErrorVariant) {}

#[cfg(not(feature = "evo-logging"))]
pub fn log_combination_choice(
    _method: &str,
    _best_index: Option<usize>,
    _candidates: &Vec<(usize, f64)>,
) {
}
pub use context::{current_epoch, current_tree_id, with_epoch, with_tree_id};
pub use evo::{create_logger, EvoLogger, NoopLogger};
pub use manager::{
    clear_event_sender, drain_logging_events, is_logging_enabled, set_event_sender, try_send_event,
};
pub use simple::init_logging;
#[cfg(feature = "evo-logging")]
pub use sqlite_logger::SqliteBufferedLogger;
pub use types::{
    Action, CombinedGridSnapshot, ComponentStateSnapshot, EpochScalingSnapshot, GridError,
    GridErrorVariant, LoggingConfig, LoggingMessage, SplitEvent, Update,
};

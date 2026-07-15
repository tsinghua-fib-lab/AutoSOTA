use crate::logging::types::{
    CombinedGridSnapshot, ComponentStateSnapshot, EpochScalingSnapshot, FComponentStats,
    GridErrorVariant, LoggingConfig, SplitEvent,
};

pub trait EvoLogger {
    fn start_run(&mut self, params_json: &str, n_rows: usize, n_cols: usize) -> i64;
    fn push_split_event(&mut self, epoch: usize, tree_id: usize, event: SplitEvent);
    fn push_combined_grid(&mut self, snapshot: CombinedGridSnapshot);
    fn push_epoch_scaling(&mut self, snapshot: EpochScalingSnapshot);
    fn push_component_state(
        &mut self,
        epoch: usize,
        tree_id: usize,
        snapshot: ComponentStateSnapshot,
    );
    fn push_grid_error_combined(&mut self, epoch: usize, err: f64, variant: GridErrorVariant);
    fn push_grid_error_fitted(
        &mut self,
        epoch: usize,
        tree_id: usize,
        err: f64,
        variant: GridErrorVariant,
    );
    fn push_combination_choice(
        &mut self,
        epoch: usize,
        method: &str,
        best_index: Option<usize>,
        candidate_indices: Vec<(usize, f64)>,
    );
    fn push_f_component_stats(
        &mut self,
        epoch: usize,
        tree_id: usize,
        iter_no: usize,
        stats_plus: FComponentStats,
        stats_minus: FComponentStats,
    );
    fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>>;
}

#[derive(Debug, Default)]
pub struct NoopLogger;

impl EvoLogger for NoopLogger {
    #[inline]
    fn start_run(&mut self, _params_json: &str, _n_rows: usize, _n_cols: usize) -> i64 {
        0
    }
    #[inline]
    fn push_split_event(&mut self, _epoch: usize, _tree_id: usize, _event: SplitEvent) {}
    #[inline]
    fn push_combined_grid(&mut self, _snapshot: CombinedGridSnapshot) {}
    #[inline]
    fn push_epoch_scaling(&mut self, _snapshot: EpochScalingSnapshot) {}
    #[inline]
    fn push_component_state(
        &mut self,
        _epoch: usize,
        _tree_id: usize,
        _snapshot: ComponentStateSnapshot,
    ) {
    }
    #[inline]
    fn push_grid_error_combined(&mut self, _epoch: usize, _err: f64, _variant: GridErrorVariant) {}
    #[inline]
    fn push_grid_error_fitted(
        &mut self,
        _epoch: usize,
        _tree_id: usize,
        _err: f64,
        _variant: GridErrorVariant,
    ) {
    }
    #[inline]
    fn push_f_component_stats(
        &mut self,
        _epoch: usize,
        _tree_id: usize,
        _iter_no: usize,
        _stats_plus: FComponentStats,
        _stats_minus: FComponentStats,
    ) {
    }
    #[inline]
    fn push_combination_choice(
        &mut self,
        _epoch: usize,
        _method: &str,
        _best_index: Option<usize>,
        _candidate_indices: Vec<(usize, f64)>,
    ) {
    }
    #[inline]
    fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}

#[cfg(feature = "evo-logging")]
use crate::logging::SqliteBufferedLogger;

pub fn create_logger(config: Option<LoggingConfig>) -> Box<dyn EvoLogger> {
    #[cfg(feature = "evo-logging")]
    {
        if let Some(config) = config {
            Box::new(SqliteBufferedLogger::new(config))
        } else {
            Box::new(NoopLogger)
        }
    }

    #[cfg(not(feature = "evo-logging"))]
    {
        let _ = config;
        Box::new(NoopLogger)
    }
}

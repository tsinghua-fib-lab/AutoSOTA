use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Action {
    Split,
    Resplit,
    Merge,
}

impl Action {
    pub fn as_str(&self) -> &'static str {
        match self {
            Action::Split => "split",
            Action::Resplit => "resplit",
            Action::Merge => "merge",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SplitEvent {
    pub iter_no: usize,
    pub action: Action,
    pub col: usize,
    pub left_interval_idx: usize,
    pub split_value: Option<f64>,
    pub update_a: Option<f64>,
    pub update_b: Option<f64>,
    pub left_count: Option<usize>,
    pub right_count: Option<usize>,
    pub gain: Option<f64>,
    pub err_before: Option<f64>,
    pub err_after: Option<f64>,
    pub n_cells_after: usize,
    pub residual_updates: Vec<Update>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Update {
    pub sample_id: u32,
    pub residual: f32,
    pub y_hat: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct LoggingConfig {
    pub db_path: String,
    pub run_label: Option<String>,
    pub record_residual_updates: bool,
    pub pack_updates_as_blob: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            db_path: "target/tg_logs.sqlite".to_string(),
            run_label: None,
            record_residual_updates: true,
            pack_updates_as_blob: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CombinedGridSnapshot {
    pub epoch: usize,
    pub energy: Option<f64>,        // Energy of the unscaled predictions
    pub scaling: Option<f64>, // Scaling for this epoch (if available) - legacy field, use scaling_plus
    pub scaling_plus: Option<f64>, // Scaling for f+ component (if available)
    pub scaling_minus: Option<f64>, // Scaling for f- component (if available)
    pub grid_json: String,    // JSON with splits/grid_values/intervals/two-tensor fields
    pub f_plus: Option<Vec<f64>>, // f+ component values (if available)
    pub f_minus: Option<Vec<f64>>, // f- component values (if available)
}

/// Represents a scaling value for a specific epoch, potentially re-optimized at a later epoch
#[derive(Debug, Clone)]
pub struct EpochScalingSnapshot {
    pub epoch: usize,              // The epoch this scaling belongs to
    pub scaling: f64,              // The scaling value
    pub optimization_epoch: usize, // The epoch when this scaling was computed/updated
}

// Lossless per-component snapshot (exact intervals/values encoded as binary)
#[derive(Debug, Clone)]
pub struct ComponentStateSnapshot {
    pub iter_no: usize,
    pub col: usize,
    pub intervals_count: usize,
    pub data: Vec<u8>, // Backward compat: grid_values = b * cosh(d)

    // New two-tensor fields
    pub backbone_data: Option<Vec<u8>>, // b_j^k per interval
    pub tilt_data: Option<Vec<u8>>,     // d_j^k per interval
    pub lambda_plus: Option<f64>,       // Stage-level λ_+
    pub lambda_minus: Option<f64>,      // Stage-level λ_-
}

/// Statistics for f+ or f- component (aggregated per iteration)
#[derive(Debug, Clone)]
pub struct FComponentStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p95: f64,
    pub p99: f64,
    pub n_samples: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GridErrorVariant {
    Train,
    Test,
}

impl GridErrorVariant {
    pub fn as_str(&self) -> &'static str {
        match self {
            GridErrorVariant::Train => "train",
            GridErrorVariant::Test => "test",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GridError {
    pub err: f64,
    pub variant: GridErrorVariant,
}

#[derive(Debug, Clone)]
pub enum LoggingMessage {
    Split {
        epoch: usize,
        tree_id: usize,
        event: SplitEvent,
    },
    Component {
        epoch: usize,
        tree_id: usize,
        snapshot: ComponentStateSnapshot,
    },
    GridErrCombined {
        epoch: usize,
        err: f64,
        variant: GridErrorVariant,
    },
    GridErrFitted {
        epoch: usize,
        tree_id: usize,
        err: f64,
        variant: GridErrorVariant,
    },
    FComponentStats {
        epoch: usize,
        tree_id: usize,
        iter_no: usize,
        stats_plus: FComponentStats,
        stats_minus: FComponentStats,
    },
    CombinationChoice {
        epoch: usize,
        method: String,
        best_index: Option<usize>,
        candidate_indices: Vec<(usize, f64)>, // (tree_id, score)
    },
    CombinedGrid {
        snapshot: CombinedGridSnapshot,
    },
    EpochScaling {
        snapshot: EpochScalingSnapshot,
    },
}

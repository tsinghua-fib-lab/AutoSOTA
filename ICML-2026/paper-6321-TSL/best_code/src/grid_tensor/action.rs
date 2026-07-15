use crate::grid_tensor::splitting::{MergeCandidate, ResplitCandidate, SplitCandidate};

/// Actions that can be applied to FittingState through the reducer
///
/// This enum represents all possible state mutations during fitting.
/// The action-based architecture separates concerns:
/// - Strategies propose actions (what to do)
/// - Reducer applies actions (how to do it)
///
/// This pattern makes the fit loop clean and testable.
#[derive(Debug, Clone)]
pub enum FittingAction {
    /// Apply a split with all associated updates
    ApplySplit { split: SplitCandidate },
    /// Apply a merge with all associated updates
    ApplyMerge { merge: MergeCandidate },
    /// Apply a resplit with all associated updates
    ApplyResplit { resplit: ResplitCandidate },
    /// Multiply the model's global scaling by a factor
    ApplyScaling { factor: f64 },
    /// Terminate the fitting process
    Terminate { reason: String },
    /// Composite action that chains multiple actions
    Composite { actions: Vec<FittingAction> },
}

/// Information needed to refine/update the tree after a split
#[derive(Debug, Clone)]
pub struct RefineInfo {
    /// Multiplier for left interval (or current interval for merge)
    pub left_multiplier: f64,
    /// Multiplier for right interval (or NaN for merge)
    pub right_multiplier: f64,
    /// Error reduction from this split
    pub error_reduction: f64,
    /// Affected interval in the split column (start, end)
    pub original_affected_interval: (usize, usize),
    /// Ranges of affected points in other columns [(min_pos, max_pos)]
    pub affected_points_range: Vec<(usize, usize)>,
}

impl RefineInfo {
    pub fn new(
        left_multiplier: f64,
        right_multiplier: f64,
        error_reduction: f64,
        original_affected_interval: (usize, usize),
        affected_points_range: Vec<(usize, usize)>,
    ) -> Self {
        Self {
            left_multiplier,
            right_multiplier,
            error_reduction,
            original_affected_interval,
            affected_points_range,
        }
    }
}

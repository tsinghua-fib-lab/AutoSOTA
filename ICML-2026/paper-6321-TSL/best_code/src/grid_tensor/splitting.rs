use std::cmp::Ordering;

use rand::{seq::index::sample, Rng};

use crate::grid_tensor::{
    action::FittingAction,
    state::{FittingState, MAX_CONSECUTIVE_RESPLIT},
};

fn sample_best_split(
    allowed_intervals: &[Vec<Interval>],
    error_reductions: &[Vec<f64>],
) -> Option<(usize, usize, usize, f64)> {
    let mut best_split_candidate: Option<(usize, usize, usize, f64)> = None;

    for (col, intervals) in allowed_intervals.iter().enumerate() {
        for (interval_idx, interval) in intervals.iter().enumerate() {
            for index in interval.start..interval.end() {
                let err_reduction = error_reductions[col][index];
                if !err_reduction.is_nan()
                    && best_split_candidate
                        .as_ref()
                        .is_none_or(|current_best| err_reduction > current_best.3)
                {
                    best_split_candidate = Some((col, index, interval_idx, err_reduction));
                }
            }
        }
    }
    best_split_candidate
}

fn sample_random_split<R: Rng + ?Sized>(
    allowed_intervals: &[Vec<Interval>],
    total_positions: &[usize],
    colsample_bytree: f64,
    split_try: usize,
    rng: &mut R,
    error_reductions: &[Vec<f64>],
) -> Option<(usize, usize, usize, f64)> {
    let cols_can_be_sampled: Vec<usize> = total_positions
        .iter()
        .enumerate()
        .filter_map(|(idx, &positions)| if positions == 0 { None } else { Some(idx) })
        .collect();

    let ncols = error_reductions.len();
    let ncols_to_sample = (colsample_bytree * ncols as f64) as usize;

    let cols = if cols_can_be_sampled.len() < ncols_to_sample {
        sample(rng, ncols, ncols_to_sample).into_vec()
    } else {
        cols_can_be_sampled
    };

    let mut best_split: Option<(usize, usize, usize, f64)> = None;
    for col in cols {
        // Use cached total positions for this feature
        let total_positions = total_positions[col];

        let abstract_indices = if total_positions < split_try {
            (0..total_positions).collect()
        } else {
            sample(rng, total_positions, split_try).into_vec()
        };

        // Map abstract indices to actual split positions
        for abstract_idx in abstract_indices {
            let (split_position, interval_idx) =
                map_abstract_to_position(allowed_intervals, col, abstract_idx);

            // Check if this position has a valid error reduction
            if split_position < error_reductions[col].len()
                && !error_reductions[col][split_position].is_nan()
                && best_split
                    .as_ref()
                    .is_none_or(|candidate| error_reductions[col][split_position] > candidate.3)
            {
                best_split = Some((
                    col,
                    split_position,
                    interval_idx,
                    error_reductions[col][split_position],
                ));
            }
        }
    }

    best_split
}

fn sample_top_k_split<R: Rng + ?Sized>(
    allowed_intervals: &[Vec<Interval>],
    top_k: usize,
    must_fill_all_k: bool,
    rng: &mut R,
    error_reductions: &[Vec<f64>],
) -> Option<(usize, usize, usize, f64)> {
    let mut candidates = Vec::new();

    for (col, intervals) in allowed_intervals.iter().enumerate() {
        for (interval_idx, interval) in intervals.iter().enumerate() {
            for index in interval.start..interval.end() {
                let err_reduction = error_reductions[col][index];
                if !err_reduction.is_nan() {
                    candidates.push((col, index, interval_idx, err_reduction));
                }
            }
        }
    }

    if candidates.is_empty() || (must_fill_all_k && candidates.len() < top_k) {
        return None;
    }

    // Sort by error reduction (descending) and take top k
    candidates.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(Ordering::Equal));
    candidates.truncate(top_k);

    // Randomly select one from the top k
    let random_index = rng.gen_range(0..candidates.len());
    Some(candidates[random_index])
}

#[derive(Debug, Clone, PartialEq)]
pub enum SplitType {
    Split,
    Merge,
    Resplit,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub start: usize,
    pub length: usize,
}

impl Interval {
    pub fn new(start: usize, length: usize) -> Self {
        Self { start, length }
    }

    pub fn end(&self) -> usize {
        self.start + self.length
    }

    pub fn contains(&self, position: usize) -> bool {
        position >= self.start && position < self.end()
    }

    pub fn overlaps(&self, other: &Interval) -> bool {
        self.start < other.end() && other.start < self.end()
    }

    pub fn subtract(&self, forbidden: &Interval) -> Vec<Interval> {
        if !self.overlaps(forbidden) {
            return vec![*self];
        }

        let mut result = Vec::new();

        // Left segment (before forbidden interval)
        if self.start < forbidden.start {
            result.push(Interval::new(self.start, forbidden.start - self.start));
        }

        // Right segment (after forbidden interval)
        if self.end() > forbidden.end() {
            result.push(Interval::new(forbidden.end(), self.end() - forbidden.end()));
        }

        result
    }
}

#[derive(Debug, Clone)]
pub struct SplitCandidate {
    pub col: usize,
    pub error_reduction: f64,
    pub allowed_interval_idx: usize,
    pub index: usize,
    /// Two-tensor updates for left side: (u_plus_L, u_minus_L)
    pub update_left: (f64, f64),
    /// Two-tensor updates for right side: (u_plus_R, u_minus_R)
    pub update_right: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct MergeCandidate {
    pub col: usize,
    pub error_reduction: f64,
    pub interval_idx: usize,
    pub index: usize,
}

#[derive(Debug, Clone)]
pub struct ResplitCandidate {
    pub col: usize,
    pub error_reduction: f64,
    pub interval_idx: usize,
    pub index: usize,
    /// Two-tensor updates for left side: (u_plus_L, u_minus_L)
    pub update_left: (f64, f64),
    /// Two-tensor updates for right side: (u_plus_R, u_minus_R)
    pub update_right: (f64, f64),
}

/// Split strategy state
#[derive(Debug, Clone)]
pub struct SplitStrategyState {
    /// Valid split intervals per feature [col][n_intervals]
    pub allowed_intervals: Vec<Vec<Interval>>,
    /// Total positions per feature [col]
    pub total_positions: Vec<usize>,
    pub last_transformation: Option<FittingAction>,
    pub resplit_enabled: bool,
    pub merge_enabled: bool,
}

impl Default for SplitStrategyState {
    fn default() -> Self {
        Self::new()
    }
}

impl SplitStrategyState {
    pub fn new() -> Self {
        Self {
            allowed_intervals: vec![],
            total_positions: vec![],
            last_transformation: None,
            resplit_enabled: true,
            // Merge is currently single-tensor/legacy-only; disable by default for two-tensor correctness.
            merge_enabled: false,
        }
    }
}

/// Enum dispatch for all split strategy types (matching actual implementation)
#[derive(Debug, Clone)]
pub enum SplitStrategy {
    Random {
        split_try: usize,
        colsample_bytree: f64,
        min_interval_samples: usize,
        /// Complexity penalty (lambda) for adaptive merge bonus.
        complexity_penalty: f64,
        min_split_loss: f64,
    },
    Best {
        min_interval_samples: usize,
        complexity_penalty: f64,
        min_split_loss: f64,
    },
    TopK {
        top_k: usize,
        must_fill_all_k: bool,
        min_interval_samples: usize,
        complexity_penalty: f64,
        min_split_loss: f64,
    },
}

impl SplitStrategy {
    pub fn initialize<'a>(&self, mut state: FittingState<'a>) -> FittingState<'a> {
        if !state.precomputed_statistics.initialized {
            panic!("Statistics are not initialized yet!")
        }
        let min_interval_samples = self.min_interval_samples();
        let error_reductions = &state.precomputed_statistics.error_reductions_split;

        let (allowed_intervals, total_positions) =
            initialize_intervals(error_reductions, min_interval_samples);

        state.split_strategy_state.allowed_intervals = allowed_intervals;
        state.split_strategy_state.total_positions = total_positions;

        state
    }

    pub fn propose_next_action<R: Rng + ?Sized>(
        &self,
        state: &FittingState,
        rng: &mut R,
    ) -> Option<FittingAction> {
        let best_split_candidate = self.propose_best_split(state, rng);
        // TODO: merges are disabled — the OptimalMerge path produces NaN
        // residuals (see ignored tests in fit.rs::tests and reducer.rs:134).
        // Re-enable once the precomputed update_pairs_merge / applied-params
        // mismatch is fixed.
        let best_merge_candidate: Option<MergeCandidate> = None;
        let best_resplit_candidate = self.propose_best_resplit(state);

        let split_action = best_split_candidate
            .map(|sc| (sc.error_reduction, FittingAction::ApplySplit { split: sc }));

        // Compute adaptive merge bonus for the best merge candidate
        let merge_action = best_merge_candidate.map(|sc| {
            let adaptive_bonus = self.compute_adaptive_merge_bonus(state, &sc);
            (
                sc.error_reduction + adaptive_bonus,
                FittingAction::ApplyMerge { merge: sc },
            )
        });

        let resplit_action = best_resplit_candidate.map(|sc| {
            (
                sc.error_reduction,
                FittingAction::ApplyResplit { resplit: sc },
            )
        });

        [split_action, merge_action, resplit_action]
            .into_iter()
            .flatten()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
            .map(|(_, action)| action)
    }

    /// Compute adaptive merge bonus for a merge candidate.
    ///
    /// The bonus is BIC-inspired and scale-invariant:
    ///   bonus = lambda * MSE * (log(n)/n + 1/harmonic_mean(n_left, n_right))
    ///
    /// Properties:
    /// - Scale-invariant: proportional to MSE
    /// - Larger bonus for smaller n (less data → prefer simpler models)
    /// - Larger bonus when merging small intervals (uncertain estimates)
    fn compute_adaptive_merge_bonus(&self, state: &FittingState, merge: &MergeCandidate) -> f64 {
        let lambda = self.complexity_penalty();
        if lambda <= 0.0 {
            return 0.0;
        }

        let n_total = state.n as f64;
        let current_mse = state.current_error / n_total;

        // Get interval sample counts for the merge
        let col = merge.col;
        let interval_idx = merge.interval_idx;

        // Compute n_left and n_right from boundaries
        let boundaries = &state.boundaries[col];
        let n_points = state.precomputed_statistics.sorted_indices[col].len();

        let start_left = if interval_idx == 0 {
            0
        } else {
            boundaries[interval_idx - 1]
        };
        let split_point = boundaries[interval_idx];
        let end_right = boundaries
            .get(interval_idx + 1)
            .copied()
            .unwrap_or(n_points);

        let n_left = (split_point - start_left) as f64;
        let n_right = (end_right - split_point) as f64;

        // BIC component: complexity cost of having an extra split
        let bic_bonus = lambda * current_mse * n_total.ln() / n_total;

        // Small interval bonus: harmonic mean penalizes small intervals
        let harmonic_mean = if n_left > 0.0 && n_right > 0.0 {
            2.0 * n_left * n_right / (n_left + n_right)
        } else {
            1.0 // Avoid division by zero
        };
        let small_interval_bonus = lambda * current_mse / harmonic_mean.max(1.0);

        bic_bonus + small_interval_bonus
    }

    /// Forbids splits around the given position by updating only the specific interval that contained the split
    pub fn forbid_around_split<'a>(
        &self,
        mut state: FittingState<'a>,
        split_candidate: &SplitCandidate,
    ) -> FittingState<'a> {
        let margin = self.min_interval_samples();
        let position = split_candidate.index;
        let interval_idx = split_candidate.allowed_interval_idx;
        let col = split_candidate.col;
        let allowed_intervals = &mut state.split_strategy_state.allowed_intervals;
        let total_positions = &mut state.split_strategy_state.total_positions;
        let forbidden_start = position.saturating_sub(margin - 1);
        let forbidden_end = position + margin;
        let forbidden_interval = Interval::new(forbidden_start, forbidden_end - forbidden_start);

        // Only update the specific interval that contained the split
        let original_interval = &allowed_intervals[col][interval_idx];
        let new_intervals = original_interval.subtract(&forbidden_interval);

        // Remove empty intervals
        let filtered_intervals: Vec<Interval> = new_intervals
            .into_iter()
            .filter(|interval| interval.length > 0)
            .collect();

        // Calculate the change in total positions
        let original_length = original_interval.length;
        let new_total: usize = filtered_intervals
            .iter()
            .map(|interval| interval.length)
            .sum();
        let change = new_total as i32 - original_length as i32;

        // Replace the original interval with the new ones
        if filtered_intervals.is_empty() {
            // If no intervals remain, remove this interval entirely
            allowed_intervals[col].remove(interval_idx);
        } else {
            // Replace the original interval with the new ones
            allowed_intervals[col].remove(interval_idx);
            allowed_intervals[col].extend(filtered_intervals);

            // Re-sort to maintain the sorted order
            allowed_intervals[col].sort_by_key(|interval| interval.start);
        }

        // Update the cached total positions
        total_positions[col] = (total_positions[col] as i32 + change) as usize;

        state
    }

    pub fn min_interval_samples(&self) -> usize {
        match self {
            SplitStrategy::Random {
                min_interval_samples,
                ..
            } => *min_interval_samples,
            SplitStrategy::Best {
                min_interval_samples,
                ..
            } => *min_interval_samples,
            SplitStrategy::TopK {
                min_interval_samples,
                ..
            } => *min_interval_samples,
        }
    }

    /// Returns the complexity penalty (lambda) for adaptive merge bonus.
    ///
    /// The merge bonus is computed as:
    ///   bonus = lambda * MSE * (log(n)/n + 1/harmonic_mean(n_left, n_right))
    ///
    /// This is BIC-inspired and scale-invariant.
    pub fn complexity_penalty(&self) -> f64 {
        match self {
            SplitStrategy::Random {
                complexity_penalty, ..
            } => *complexity_penalty,
            SplitStrategy::Best {
                complexity_penalty, ..
            } => *complexity_penalty,
            SplitStrategy::TopK {
                complexity_penalty, ..
            } => *complexity_penalty,
        }
    }

    pub fn min_split_loss(&self) -> f64 {
        match self {
            SplitStrategy::Random { min_split_loss, .. } => *min_split_loss,
            SplitStrategy::Best { min_split_loss, .. } => *min_split_loss,
            SplitStrategy::TopK { min_split_loss, .. } => *min_split_loss,
        }
    }

    #[allow(dead_code)] // TODO: re-enable in propose_next_action once merge NaN is fixed
    fn propose_best_merge(&self, state: &FittingState) -> Option<MergeCandidate> {
        if !state.split_strategy_state.merge_enabled {
            return None;
        }
        let mut best_merge_candidate: Option<MergeCandidate> = None;
        for (col_idx, (err_reductions, boundaries)) in state
            .precomputed_statistics
            .error_reductions_merge
            .iter()
            .zip(state.boundaries.iter())
            .enumerate()
        {
            for (index, (&err_reduction, &data_index)) in
                err_reductions.iter().zip(boundaries.iter()).enumerate()
            {
                // Skip NaN values to prevent panic in partial_cmp
                if err_reduction.is_nan() {
                    continue;
                }
                // Merge=OptimalMerge: all boundaries with valid gains are mergeable
                // (no undo_map check needed - we compute optimal merged params at merge time)

                if best_merge_candidate
                    .as_ref()
                    .is_none_or(|sc| err_reduction > sc.error_reduction)
                {
                    let _update = state.precomputed_statistics.update_pairs_merge[col_idx][index];
                    best_merge_candidate = Some(MergeCandidate {
                        col: col_idx,
                        error_reduction: err_reduction,
                        interval_idx: index,
                        index: data_index,
                    });
                }
            }
        }

        best_merge_candidate
    }

    fn propose_best_resplit(&self, state: &FittingState) -> Option<ResplitCandidate> {
        if !state.split_strategy_state.resplit_enabled {
            return None;
        }
        // Prevent too many consecutive resplits to avoid infinite loops
        if state.loop_state.consecutive_resplits >= MAX_CONSECUTIVE_RESPLIT {
            return None;
        }
        let mut best_resplit_candidate: Option<ResplitCandidate> = None;
        for (col_idx, (err_reductions, boundaries)) in state
            .precomputed_statistics
            .error_reductions_resplit
            .iter()
            .zip(state.boundaries.iter())
            .enumerate()
        {
            for (index, (&err_reduction, &data_index)) in
                err_reductions.iter().zip(boundaries.iter()).enumerate()
            {
                // Skip NaN values to prevent panic in partial_cmp
                if err_reduction.is_nan() {
                    continue;
                }
                if let Some(last_transformation) =
                    state.split_strategy_state.last_transformation.as_ref()
                {
                    match last_transformation {
                        FittingAction::ApplySplit { split } => {
                            if col_idx == split.col && data_index == split.index {
                                continue;
                            }
                        }
                        FittingAction::ApplyResplit { resplit } => {
                            if col_idx == resplit.col && data_index == resplit.index {
                                continue;
                            }
                        }
                        _ => {}
                    }
                }
                if best_resplit_candidate
                    .as_ref()
                    .is_none_or(|sc| err_reduction > sc.error_reduction)
                {
                    let update_left =
                        state.precomputed_statistics.update_pairs_resplit_left[col_idx][index];
                    let update_right =
                        state.precomputed_statistics.update_pairs_resplit_right[col_idx][index];
                    let candidate = ResplitCandidate {
                        col: col_idx,
                        index: data_index,
                        error_reduction: err_reduction,
                        interval_idx: index,
                        update_left,
                        update_right,
                    };
                    best_resplit_candidate = Some(candidate);
                }
            }
        }
        best_resplit_candidate
    }

    fn propose_best_split<R: Rng + ?Sized>(
        &self,
        state: &FittingState,
        rng: &mut R,
    ) -> Option<SplitCandidate> {
        let best_split_info = match self {
            SplitStrategy::Random {
                colsample_bytree,
                split_try,
                ..
            } => {
                // Random strategy implementation - uses state.strategy_state for allowed intervals
                sample_random_split(
                    &state.split_strategy_state.allowed_intervals,
                    &state.split_strategy_state.total_positions,
                    *colsample_bytree,
                    *split_try,
                    rng,
                    &state.precomputed_statistics.error_reductions_split,
                )
            }
            SplitStrategy::Best { .. } => {
                // Best strategy implementation - uses state.working_buffers for error reductions
                sample_best_split(
                    &state.split_strategy_state.allowed_intervals,
                    &state.precomputed_statistics.error_reductions_split,
                )
            }
            SplitStrategy::TopK {
                top_k,
                must_fill_all_k,
                ..
            } => {
                // Top-k strategy implementation
                sample_top_k_split(
                    &state.split_strategy_state.allowed_intervals,
                    *top_k,
                    *must_fill_all_k,
                    rng,
                    &state.precomputed_statistics.error_reductions_split,
                )
            }
        };

        if let Some((col, index, interval_idx, err_reduction)) = best_split_info {
            if err_reduction < self.min_split_loss() {
                return None;
            }

            let update_left = state.precomputed_statistics.update_pairs_split_left[col][index];
            let update_right = state.precomputed_statistics.update_pairs_split_right[col][index];
            Some(SplitCandidate {
                col,
                index,
                error_reduction: err_reduction,
                allowed_interval_idx: interval_idx,
                update_left,
                update_right,
            })
        } else {
            None
        }
    }
}

fn map_abstract_to_position(
    allowed_intervals: &[Vec<Interval>],
    col: usize,
    abstract_idx: usize,
) -> (usize, usize) {
    let mut remaining = abstract_idx;
    for (interval_idx, interval) in allowed_intervals[col].iter().enumerate() {
        if remaining < interval.length {
            return (interval.start + remaining, interval_idx);
        }
        remaining -= interval.length;
    }
    // This should never happen if abstract_idx is valid
    panic!("Invalid abstract index: {}", abstract_idx);
}

fn initialize_intervals(
    error_reductions: &[Vec<f64>],
    min_interval_samples: usize,
) -> (Vec<Vec<Interval>>, Vec<usize>) {
    let mut allowed_intervals = vec![Vec::new(); error_reductions.len()];
    let mut total_positions = vec![0; error_reductions.len()];

    for (col, col_error_reductions) in error_reductions.iter().enumerate() {
        let mut intervals = Vec::new();
        let mut current_start = None;

        for (pos, &err_reduction) in col_error_reductions.iter().enumerate() {
            if !err_reduction.is_nan() {
                if current_start.is_none() {
                    current_start = Some(pos);
                }
            } else if let Some(start) = current_start {
                intervals.push(Interval::new(start, pos - start));
                current_start = None;
            }
        }

        if let Some(start) = current_start {
            intervals.push(Interval::new(start, col_error_reductions.len() - start));
        }

        intervals.retain(|interval| interval.length > 0);

        let n = col_error_reductions.len();
        let margin = min_interval_samples;
        let global_start = margin;
        let global_end = n.saturating_sub(margin - 1);
        let new_intervals: Vec<Interval> = intervals
            .into_iter()
            .filter_map(|interval| {
                let old_end = interval.end();
                let start = interval.start.max(global_start);
                let end = old_end.min(global_end);
                if end > start {
                    Some(Interval::new(start, end - start))
                } else {
                    None
                }
            })
            .collect();

        let total = new_intervals.iter().map(|interval| interval.length).sum();
        allowed_intervals[col] = new_intervals;
        total_positions[col] = total;
    }

    (allowed_intervals, total_positions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_tensor::state::FittingState;
    use ndarray::{Array1, Array2};
    use rand::{rngs::StdRng, SeedableRng};

    // Helper to create a mock FittingState for testing
    fn create_mock_state(error_reductions: Vec<Vec<f64>>) -> FittingState<'static> {
        let n = 20;
        let p = error_reductions.len();

        // Create dummy data
        let x_data = Array2::zeros((n, p));
        let y_data = Array1::zeros(n);

        // Leak to get 'static lifetime (only for tests)
        let x_static = Box::leak(Box::new(x_data));
        let y_static = Box::leak(Box::new(y_data));

        let mut state = FittingState::new(x_static.view(), y_static.view());

        // Set up precomputed statistics with our error reductions
        state.precomputed_statistics.error_reductions_split = error_reductions;
        state.precomputed_statistics.initialized = true;

        state
    }

    #[test]
    fn test_top_k_splits_selects_from_top_k() {
        let error_reductions = vec![
            vec![f64::NAN, 1.0, 5.0, 2.0, 4.0, 3.0], // top 2 are: 5.0, 4.0
        ];
        let strategy = SplitStrategy::TopK {
            top_k: 2,
            must_fill_all_k: false,
            min_interval_samples: 1,
            complexity_penalty: 0.0,
            min_split_loss: 0.0,
        };
        let mut state = create_mock_state(error_reductions);
        state = strategy.initialize(state);

        // Enable merge for testing
        state.split_strategy_state.merge_enabled = true;

        let mut rng = StdRng::seed_from_u64(42);
        let action = strategy.propose_next_action(&state, &mut rng);

        // Should select one of the top 2 splits
        assert!(action.is_some());
        if let Some(FittingAction::ApplySplit { split }) = action {
            let (col, index) = (split.col, split.index);
            assert_eq!(col, 0);
            assert!(index == 2 || index == 4); // Either 5.0 at index 2 or 4.0 at index 4
        }
    }

    #[test]
    fn test_top_k_splits_must_fill_all_k_true() {
        let error_reductions = vec![
            vec![f64::NAN, 1.0, 2.0, 3.0, f64::NAN], // Only 3 valid splits
        ];
        let strategy = SplitStrategy::TopK {
            top_k: 5,
            must_fill_all_k: true,
            min_interval_samples: 1,
            complexity_penalty: 0.0,
            min_split_loss: 0.0,
        };
        let mut state = create_mock_state(error_reductions);
        state = strategy.initialize(state);

        // Enable merge for testing
        state.split_strategy_state.merge_enabled = true;

        let mut rng = StdRng::seed_from_u64(42);
        let action = strategy.propose_next_action(&state, &mut rng);

        // Should return None because we can't fill all k=5 splits (only 3 valid)
        assert!(action.is_none());
    }

    #[test]
    fn test_top_k_splits_must_fill_all_k_false() {
        let error_reductions = vec![
            vec![f64::NAN, 1.0, 2.0, 3.0, f64::NAN], // Only 3 valid splits
        ];
        let strategy = SplitStrategy::TopK {
            top_k: 5,
            must_fill_all_k: false,
            min_interval_samples: 1,
            complexity_penalty: 0.0,
            min_split_loss: 0.0,
        };
        let mut state = create_mock_state(error_reductions);
        state = strategy.initialize(state);

        // Enable merge for testing
        state.split_strategy_state.merge_enabled = true;

        let mut rng = StdRng::seed_from_u64(42);
        let action = strategy.propose_next_action(&state, &mut rng);

        // Should return a result because must_fill_all_k is false
        assert!(action.is_some());
    }

    #[test]
    fn test_top_k_splits_multiple_calls() {
        let error_reductions = vec![vec![f64::NAN, 1.0, 5.0, 2.0, 4.0, 3.0]];
        let strategy = SplitStrategy::TopK {
            top_k: 2,
            must_fill_all_k: false,
            min_interval_samples: 1,
            complexity_penalty: 0.0,
            min_split_loss: 0.0,
        };
        let mut state = create_mock_state(error_reductions);
        state = strategy.initialize(state);

        let mut rng = StdRng::seed_from_u64(42);

        // First split from top-k
        let action1 = strategy.propose_next_action(&state, &mut rng);
        assert!(action1.is_some());
        if let Some(FittingAction::ApplySplit { split }) = action1 {
            // Forbid around the selected split
            state = strategy.forbid_around_split(state, &split);
        }

        // Second split should still work
        let action2 = strategy.propose_next_action(&state, &mut rng);
        assert!(action2.is_some());
    }

    #[test]
    fn test_top_k_splits_empty_heap() {
        let error_reductions = vec![
            vec![f64::NAN, f64::NAN, f64::NAN], // No valid splits
        ];
        let strategy = SplitStrategy::TopK {
            top_k: 2,
            must_fill_all_k: false,
            min_interval_samples: 1,
            complexity_penalty: 0.0,
            min_split_loss: 0.0,
        };
        let mut state = create_mock_state(error_reductions);
        state = strategy.initialize(state);

        // Enable merge for testing
        state.split_strategy_state.merge_enabled = true;

        let mut rng = StdRng::seed_from_u64(42);
        let action = strategy.propose_next_action(&state, &mut rng);

        assert!(action.is_none());
    }

    #[test]
    fn test_top_k_splits_nan_handling() {
        let error_reductions = vec![vec![f64::NAN, 1.0, f64::NAN, 3.0, f64::NAN, 2.0]];
        let strategy = SplitStrategy::TopK {
            top_k: 2,
            must_fill_all_k: false,
            min_interval_samples: 1,
            complexity_penalty: 0.0,
            min_split_loss: 0.0,
        };
        let mut state = create_mock_state(error_reductions);
        state = strategy.initialize(state);

        // Enable merge for testing
        state.split_strategy_state.merge_enabled = true;

        let mut rng = StdRng::seed_from_u64(42);
        let action = strategy.propose_next_action(&state, &mut rng);

        assert!(action.is_some());
        if let Some(FittingAction::ApplySplit { split }) = action {
            let (col, index) = (split.col, split.index);
            let err_reductions = &state.precomputed_statistics.error_reductions_split;
            assert!(!err_reductions[col][index].is_nan());
            // Should be one of the valid non-NaN positions
            assert!(index == 1 || index == 3 || index == 5);
        }
    }

    // random_split_tests

    #[test]
    fn test_random_split_exact_split_try_less_than_total_positions() {
        let error_reductions = vec![vec![f64::NAN, 1.0, 2.0, 3.0, 4.0]];
        let strategy = SplitStrategy::Random {
            split_try: 10,
            colsample_bytree: 1.0,
            min_interval_samples: 1,
            complexity_penalty: 0.0,
            min_split_loss: 0.0,
        };
        let mut state = create_mock_state(error_reductions);
        state = strategy.initialize(state);

        // Enable merge for testing
        state.split_strategy_state.merge_enabled = true;

        let mut rng = StdRng::seed_from_u64(42);
        let action = strategy.propose_next_action(&state, &mut rng);
        assert!(action.is_some());
    }

    #[test]
    fn test_random_split_exact_split_try() {
        let error_reductions = vec![vec![
            f64::NAN,
            1.0,
            2.0,
            3.0,
            4.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
        ]];
        let min_interval_samples = 4;
        let (allowed_intervals, total_positions) =
            initialize_intervals(&error_reductions, min_interval_samples);

        // Check initialization creates correct intervals
        assert_eq!(total_positions, vec![3]);
        assert_eq!(
            allowed_intervals,
            vec![vec![Interval::new(4, 1), Interval::new(8, 2)]]
        );

        let mut rng = StdRng::seed_from_u64(42);
        let result = sample_random_split(
            &allowed_intervals,
            &total_positions,
            1.0,
            1,
            &mut rng,
            &error_reductions,
        );
        assert!(result.is_some());
    }

    #[test]
    fn test_random_split_exact_initialization() {
        let error_reductions = vec![
            vec![
                f64::NAN,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                f64::NAN,
                f64::NAN,
                8.0,
                9.0,
                10.0,
            ], // column 0
            vec![f64::NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], // column 1
        ];
        let min_interval_samples = 2;
        let (allowed_intervals, total_positions) =
            initialize_intervals(&error_reductions, min_interval_samples);

        // Should have 2 features, each with intervals
        assert_eq!(allowed_intervals.len(), 2);
        assert_eq!(allowed_intervals[0].len(), 2);
        assert_eq!(allowed_intervals[0][0], Interval::new(2, 4));
        assert_eq!(allowed_intervals[0][1], Interval::new(8, 2));
        assert_eq!(allowed_intervals[1].len(), 1);
        assert_eq!(allowed_intervals[1][0], Interval::new(2, 7));
        assert_eq!(total_positions, vec![6, 7]);
    }

    #[test]
    fn test_random_split_exact_multiple_splits() {
        let error_reductions = vec![
            vec![f64::NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], // column 0
        ];
        let min_interval_samples = 2;
        let (allowed_intervals, total_positions) =
            initialize_intervals(&error_reductions, min_interval_samples);

        let mut rng = StdRng::seed_from_u64(42);

        // Multiple splits should work
        let result1 = sample_random_split(
            &allowed_intervals,
            &total_positions,
            1.0,
            1,
            &mut rng,
            &error_reductions,
        );
        assert!(result1.is_some());

        let result2 = sample_random_split(
            &allowed_intervals,
            &total_positions,
            1.0,
            1,
            &mut rng,
            &error_reductions,
        );
        assert!(result2.is_some());

        // Third split should still work (we're not modifying intervals between calls)
        let result3 = sample_random_split(
            &allowed_intervals,
            &total_positions,
            1.0,
            1,
            &mut rng,
            &error_reductions,
        );
        assert!(result3.is_some());
    }

    #[test]
    fn test_random_split_exact_nan_handling() {
        let error_reductions = vec![
            vec![f64::NAN, f64::NAN, 2.0, f64::NAN, 4.0], // column 0 with NaN values
            vec![f64::NAN, 1.0, f64::NAN, 3.0, f64::NAN], // column 1 with NaN values
        ];
        let min_interval_samples = 1;
        let (allowed_intervals, total_positions) =
            initialize_intervals(&error_reductions, min_interval_samples);

        let mut rng = StdRng::seed_from_u64(42);
        let result = sample_random_split(
            &allowed_intervals,
            &total_positions,
            1.0,
            2,
            &mut rng,
            &error_reductions,
        );
        let (col, index, _interval_idx, _err) = result.unwrap();
        assert!(!error_reductions[col][index].is_nan());
    }

    #[test]
    fn test_random_split_filters_splits_below_min_split_loss() {
        // Create minimal test data
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // Create state
        let mut state = FittingState::new(x.view(), y.view());

        // Manually set error reductions with some below min_split_loss
        state.precomputed_statistics.error_reductions_split[0] =
            vec![f64::NAN, 0.1, 0.2, f64::NAN, 0.4, 0.5, f64::NAN, 0.3];
        state.precomputed_statistics.initialized = true;

        // Create strategy with high min_split_loss
        let strategy = SplitStrategy::Random {
            split_try: 1,
            colsample_bytree: 1.0,
            min_interval_samples: 1,
            complexity_penalty: 0.0,
            min_split_loss: 0.5, // Higher than all error reductions
        };

        state = strategy.initialize(state);

        // Enable merge for testing
        state.split_strategy_state.merge_enabled = true;

        let mut rng = StdRng::seed_from_u64(42);
        let action = strategy.propose_next_action(&state, &mut rng);

        // Should return None because all splits are below min_split_loss
        assert!(action.is_none());
    }

    // best_split_tests

    #[test]
    fn test_best_split_picks_resplit_if_better() {
        use crate::grid_tensor::state::FittingState;
        use ndarray::{Array1, Array2};

        // Create minimal test data
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // Create state
        let mut state = FittingState::new(x.view(), y.view());

        // Set up boundaries to enable resplit
        state.boundaries[0] = vec![3, 7]; // Two intervals: [0,3) and [3,7) and [7,10)

        // Set error reductions: split=1.0, resplit=2.0 (better)
        state.precomputed_statistics.error_reductions_split[0] =
            vec![f64::NAN, 1.0, 1.0, f64::NAN, 1.0, 1.0, f64::NAN, 1.0];
        state.precomputed_statistics.error_reductions_resplit[0] = vec![2.0, 2.0]; // Two resplit candidates
        state.precomputed_statistics.update_pairs_resplit_left[0] = vec![(0.0, 0.0), (0.0, 0.0)];
        state.precomputed_statistics.update_pairs_resplit_right[0] = vec![(0.0, 0.0), (0.0, 0.0)];
        state.precomputed_statistics.initialized = true;

        // Create strategy
        let strategy = SplitStrategy::Best {
            min_interval_samples: 1,
            complexity_penalty: 0.0,
            min_split_loss: 0.0,
        };

        state = strategy.initialize(state);

        // Enable merge for testing
        state.split_strategy_state.merge_enabled = true;

        let mut rng = StdRng::seed_from_u64(42);
        let action = strategy.propose_next_action(&state, &mut rng);

        // Should return a resplit action because resplit (2.0) > split (1.0)
        match action {
            Some(crate::grid_tensor::action::FittingAction::ApplyResplit { .. }) => {
                // Expected: resplit action selected
            }
            _ => panic!("Expected resplit action, got: {:?}", action),
        }
    }

    #[test]
    fn test_best_split_prevents_consecutive_resplits() {
        use crate::grid_tensor::{action::FittingAction, state::FittingState};
        use ndarray::{Array1, Array2};

        // Create minimal test data
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // Create state
        let mut state = FittingState::new(x.view(), y.view());

        // Set up boundaries to enable resplit
        state.boundaries[0] = vec![3]; // Two intervals: [0,3) and [3,10)

        // Set error reductions: split=3.0 (best), resplit=2.0 (but consecutive prevention should make split win)
        state.precomputed_statistics.error_reductions_split[0] =
            vec![f64::NAN, 3.0, 3.0, f64::NAN, 3.0, 3.0, f64::NAN, 3.0];
        state.precomputed_statistics.error_reductions_resplit[0] = vec![20.0]; // resplit with very high error reduction
        state.precomputed_statistics.update_pairs_resplit_left[0] = vec![(0.0, 0.0)]; // u_plus=0, u_minus=0 means no change
        state.precomputed_statistics.update_pairs_resplit_right[0] = vec![(0.0, 0.0)];
        state.precomputed_statistics.initialized = true;

        // Set last transformation to a resplit at the same position
        let last_resplit = crate::grid_tensor::splitting::ResplitCandidate {
            col: 0,
            index: 3, // Same position as one of our resplit candidates
            error_reduction: 2.0,
            interval_idx: 0,
            update_left: (0.0, 0.0),
            update_right: (0.0, 0.0),
        };
        state.split_strategy_state.last_transformation = Some(FittingAction::ApplyResplit {
            resplit: last_resplit,
        });

        // Create strategy
        let strategy = SplitStrategy::Best {
            min_interval_samples: 1,
            complexity_penalty: 0.0,
            min_split_loss: 0.0,
        };

        state = strategy.initialize(state);

        // Enable merge for testing
        state.split_strategy_state.merge_enabled = true;

        let mut rng = StdRng::seed_from_u64(42);
        let action = strategy.propose_next_action(&state, &mut rng);

        // Should return a split action instead of resplit because consecutive resplit is prevented
        match action {
            Some(FittingAction::ApplySplit { .. }) => {
                // Expected: split action selected instead of resplit
            }
            _ => panic!(
                "Expected split action (consecutive resplit prevented), got: {:?}",
                action
            ),
        }
    }

    #[test]
    fn test_best_split_handles_forbidding_correctly() {
        use crate::grid_tensor::state::FittingState;
        use ndarray::{Array1, Array2};

        // Create minimal test data
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // Create state
        let mut state = FittingState::new(x.view(), y.view());

        // Set error reductions with clear best choice
        state.precomputed_statistics.error_reductions_split[0] =
            vec![f64::NAN, 1.0, 5.0, 3.0, 4.0, 2.0, f64::NAN, 1.0];
        state.precomputed_statistics.initialized = true;

        // Create strategy
        let strategy = SplitStrategy::Best {
            min_interval_samples: 1,
            complexity_penalty: 0.0,
            min_split_loss: 0.0,
        };

        state = strategy.initialize(state);

        let mut rng = StdRng::seed_from_u64(42);
        let action1 = strategy.propose_next_action(&state, &mut rng);

        // Should pick the best split (5.0 at index 2)
        match action1 {
            Some(crate::grid_tensor::action::FittingAction::ApplySplit { split }) => {
                assert_eq!(split.col, 0);
                assert_eq!(split.index, 2);
                assert_eq!(split.error_reduction, 5.0);
            }
            _ => panic!("Expected split action, got: {:?}", action1),
        }

        // After forbidding around the split, the same position should not be available
        let split_candidate = crate::grid_tensor::splitting::SplitCandidate {
            col: 0,
            index: 2,
            error_reduction: 5.0,
            allowed_interval_idx: 0,
            update_left: (0.0, 0.0),
            update_right: (0.0, 0.0),
        };
        state = strategy.forbid_around_split(state, &split_candidate);
        let action2 = strategy.propose_next_action(&state, &mut rng);

        // Should pick the next best available split (4.0 at index 4)
        match action2 {
            Some(crate::grid_tensor::action::FittingAction::ApplySplit { split }) => {
                assert_eq!(split.col, 0);
                assert_eq!(split.index, 4); // Next best after forbidding index 2
                assert_eq!(split.error_reduction, 4.0);
            }
            _ => panic!(
                "Expected split action at different position, got: {:?}",
                action2
            ),
        }
    }

    #[test]
    fn test_best_split_nan_handling() {
        let error_reductions = vec![vec![f64::NAN, f64::NAN, 3.0, f64::NAN, 2.0]];
        let min_interval_samples = 1;
        let (allowed_intervals, _) = initialize_intervals(&error_reductions, min_interval_samples);

        let result = sample_best_split(&allowed_intervals, &error_reductions);
        assert_eq!(result, Some((0, 2, 0, 3.0))); // Should pick 3.0 as it's better than 2.0
    }
}

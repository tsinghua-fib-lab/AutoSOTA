//! Histogram binning prologue for the per-tree split scan.
//!
//! The exact per-tree fit scans every sorted position as a candidate split
//! threshold. Histogram binning restricts the candidate set to a deterministic
//! set of quantile bin boundaries per feature, reducing the inner-loop solver
//! call count from O(n_interval) to O(n_bins).
//!
//! In addition, the per-tree sufficient-stat prefix sums (s11, s22, s12, t1,
//! t2) are *replaced* by length-B bin-cumulative arrays for binned columns:
//! `binned_prefix_sums_*[col][k]` = sum of `c_*` over sorted positions in
//! bins `0..=k`. This shrinks the per-call bucket scatter + segment
//! prefix-sum sweep inside `update_statistics` from O(span) ≈ O(n) to O(B),
//! which is the actual hot path identified by profiling (~68% self-time).
//!
//! For non-binned columns (`bin_edges[col].is_empty()`, including the entire
//! `max_bins = None` parity path), the length-n `prefix_sums_*[col]` storage
//! is retained and used unchanged.
//!
//! Determinism: bin-edge positions are derived from sorted-position indices,
//! so the same data + the same max_bins yield the same bins on every run.

use crate::grid_tensor::state::FittingState;

/// Compute deterministic bin-edge positions in sorted order for a feature.
///
/// Returns the sorted-position indices (`pos ∈ [1, n-1]`) at which a bin
/// boundary lies. Positions where the underlying feature value equals the
/// preceding sorted value (duplicates) are skipped — those positions are
/// already disallowed as splits.
///
/// When `n_bins >= unique-value count`, every distinct boundary is a
/// candidate (effectively the exact path).
fn quantile_bin_edge_positions(
    sorted_indices: &[usize],
    feature_values: &[f64],
    n_bins: usize,
) -> Vec<usize> {
    let n = sorted_indices.len();
    if n < 2 || n_bins < 2 {
        return Vec::new();
    }
    let n_bins = n_bins.min(n);

    let mut edges = Vec::with_capacity(n_bins.saturating_sub(1));
    let mut last_pushed: Option<usize> = None;
    // Bin boundary k lands at sorted-position floor(k * n / n_bins) for k in 1..n_bins.
    for k in 1..n_bins {
        let pos = (k * n) / n_bins;
        if pos == 0 || pos >= n {
            continue;
        }
        // Skip duplicates: if x[sorted[pos]] == x[sorted[pos-1]], the position
        // is already NaN-marked as a forbidden split.
        let v_here = feature_values[sorted_indices[pos]];
        let v_prev = feature_values[sorted_indices[pos - 1]];
        if v_here == v_prev {
            continue;
        }
        if last_pushed != Some(pos) {
            edges.push(pos);
            last_pushed = Some(pos);
        }
    }
    edges
}

/// Apply histogram-binning NaN mask to the split-candidate caches.
///
/// For each feature, marks every sorted position that is NOT a quantile
/// bin boundary as NaN in `error_reductions_split`, `update_pairs_split_left`,
/// `update_pairs_split_right`, and `error_reductions_split_pairs`. Bin
/// boundary positions retain their values (just computed by
/// `RefinementStrategy::initialize`).
///
/// Must be called AFTER `RefinementStrategy::initialize` and BEFORE
/// `SplitStrategy::initialize` (so that `allowed_intervals` is built from the
/// binned mask).
pub fn apply_histogram_binning_mask<'a>(
    mut state: FittingState<'a>,
    max_bins: Option<u16>,
) -> FittingState<'a> {
    let Some(max_bins) = max_bins else {
        return state;
    };
    let n_bins = max_bins as usize;
    if n_bins < 2 {
        return state;
    }

    let p = state.p;
    let n = state.n;

    for col in 0..p {
        let sorted_indices = &state.precomputed_statistics.sorted_indices[col];
        // Build a boolean keep-mask over [0, n).
        // Position 0 and position n-1 are always NaN (no split before-first /
        // after-last), so we just need to mark the interior.
        let feature_values: Vec<f64> = (0..n).map(|i| state.x[[i, col]]).collect();
        let edges = quantile_bin_edge_positions(sorted_indices, &feature_values, n_bins);

        // If we have at least as many distinct candidates as max_bins, mask
        // out non-edge positions. Otherwise, every distinct value is its own
        // bin already — no masking needed (equivalent to exact).
        if edges.is_empty() {
            continue;
        }

        let mut keep = vec![false; n];
        for &e in &edges {
            keep[e] = true;
        }

        let er = &mut state.precomputed_statistics.error_reductions_split[col];
        let pl = &mut state.precomputed_statistics.update_pairs_split_left[col];
        let pr = &mut state.precomputed_statistics.update_pairs_split_right[col];
        let pp = &mut state.precomputed_statistics.error_reductions_split_pairs[col];
        for pos in 0..n {
            if !keep[pos] {
                er[pos] = f64::NAN;
                pl[pos] = (f64::NAN, f64::NAN);
                pr[pos] = (f64::NAN, f64::NAN);
                pp[pos] = (f64::NAN, f64::NAN);
            }
        }

        // Populate length-B bin-cumulative prefix sums by sampling the
        // length-n prefix sums at each bin's last position. These will be the
        // sole storage maintained by `update_statistics` for binned columns.
        let n_bins = edges.len() + 1;
        let mut binned = Vec::with_capacity(n_bins);
        {
            let ps = &state.precomputed_statistics.prefix_sums[col];
            for k in 0..n_bins {
                let bin_end_pos = if k < edges.len() { edges[k] } else { n };
                let idx = bin_end_pos - 1;
                binned.push(ps[idx]);
            }
        }
        state.precomputed_statistics.binned_prefix_sums[col] = binned;

        // Free the length-n prefix sums for binned columns — they are no
        // longer read once `bin_edges[col]` is non-empty.
        state.precomputed_statistics.prefix_sums[col] = Vec::new();

        // Publish bin-edge positions so the inner split-scan can iterate them
        // directly instead of walking every sorted position.
        state.precomputed_statistics.bin_edges[col] = edges;
    }

    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_edge_positions_uniform_data() {
        // Feature values 0..10 sorted -> sorted_indices = [0,1,2,...,9].
        let sorted = (0..10).collect::<Vec<_>>();
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        // 5 bins over 10 positions -> edges at positions 2, 4, 6, 8.
        let edges = quantile_bin_edge_positions(&sorted, &values, 5);
        assert_eq!(edges, vec![2, 4, 6, 8]);
    }

    #[test]
    fn bin_edge_positions_skip_duplicates() {
        // Feature values: 0, 0, 0, 1, 2, 3, 4, 5, 6, 7
        // sorted_indices preserves order.
        let sorted = (0..10).collect::<Vec<_>>();
        let values: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        // 5 bins -> would put edge at position 2 (between two 0s) -> skip.
        let edges = quantile_bin_edge_positions(&sorted, &values, 5);
        assert!(!edges.contains(&2)); // Position 2 has same value as position 1
        assert!(edges.contains(&4));
        assert!(edges.contains(&6));
        assert!(edges.contains(&8));
    }

    #[test]
    fn bin_edge_positions_max_bins_too_large() {
        // 4 points, n_bins = 100 -> n_bins capped at 4 -> 3 edges at floor(k*4/4)
        let sorted = vec![0, 1, 2, 3];
        let values: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let edges = quantile_bin_edge_positions(&sorted, &values, 100);
        // For n=4, n_bins=4: edges at k*4/4 = 1, 2, 3.
        assert_eq!(edges, vec![1, 2, 3]);
    }
}

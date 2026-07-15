//! Refinement module for updating tree statistics after splits
//!
//! This module contains the refinement strategies that update precomputed statistics
//! after each split/resplit/merge operation. The actual tree updates are handled by
//! the reducer in `grid/reducer.rs`.

use std::iter::once;

use crate::grid_tensor::state::{AffectedRange, FittingState, PrefixStats};
use crate::grid_tensor::two_tensor_solver::{solve_two_tensor, DEFAULT_V_MAX, DEFAULT_V_MIN};

pub enum RefinementStrategy {
    L2Refinement {
        alpha: f64,
        /// Two-tensor L2 coupling between u_+ and u_-.
        tilt_tau: f64,
        /// Two-tensor L1 coupling on (u_+ - u_-).
        tilt_rho: f64,
        /// Prior sample size for parent anchoring (tau_0).
        /// Interpreted as "how many samples worth of confidence in the parent".
        prior_sample_size: f64,
        update_clamp: f64,
    },
    HuberRefinement {
        alpha: f64,
        c: f64,
        tilt_tau: f64,
        tilt_rho: f64,
        prior_sample_size: f64,
        update_clamp: f64,
    },
}

impl RefinementStrategy {
    #[inline]
    pub fn tilt_tau(&self) -> f64 {
        match self {
            RefinementStrategy::L2Refinement { tilt_tau, .. }
            | RefinementStrategy::HuberRefinement { tilt_tau, .. } => *tilt_tau,
        }
    }

    #[inline]
    pub fn tilt_rho(&self) -> f64 {
        match self {
            RefinementStrategy::L2Refinement { tilt_rho, .. }
            | RefinementStrategy::HuberRefinement { tilt_rho, .. } => *tilt_rho,
        }
    }
}

#[inline]
pub fn prefix_range(prefix: &[f64], start: usize, end: usize) -> f64 {
    if start == 0 {
        prefix[end - 1]
    } else {
        prefix[end - 1] - prefix[start - 1]
    }
}

/// Subtract two `PrefixStats` componentwise.
#[inline]
fn sub_stats(a: PrefixStats, b: PrefixStats) -> PrefixStats {
    PrefixStats {
        s11: a.s11 - b.s11,
        s22: a.s22 - b.s22,
        s12: a.s12 - b.s12,
        t1: a.t1 - b.t1,
        t2: a.t2 - b.t2,
    }
}

/// Sum the per-stat prefix sums over the half-open sorted range `[start, end)`.
#[inline]
pub fn prefix_stats_range(prefix: &[PrefixStats], start: usize, end: usize) -> PrefixStats {
    if start == 0 {
        prefix[end - 1]
    } else {
        sub_stats(prefix[end - 1], prefix[start - 1])
    }
}

/// Read a bin-cumulative prefix sum at sorted position `pos`. `pos` must be
/// the last position of some bin (i.e., `pos = bin_edges[k] - 1` for some `k`
/// or `pos = n - 1`).
#[inline]
pub fn binned_prefix_stats_at(
    binned: &[PrefixStats],
    bin_edges: &[usize],
    pos: usize,
) -> PrefixStats {
    let k = bin_edges.partition_point(|&e| e <= pos);
    binned[k]
}

/// Bin-cumulative equivalent of `prefix_stats_range`: sum over sorted
/// positions `[start, end)`. Both endpoints must be bin-aligned.
#[inline]
pub fn binned_prefix_stats_range(
    binned: &[PrefixStats],
    bin_edges: &[usize],
    start: usize,
    end: usize,
) -> PrefixStats {
    let end_v = binned_prefix_stats_at(binned, bin_edges, end - 1);
    if start == 0 {
        end_v
    } else {
        sub_stats(end_v, binned_prefix_stats_at(binned, bin_edges, start - 1))
    }
}

#[inline]
pub fn l2_update_unanchored(alpha: f64, s_rb: f64, s_bb: f64) -> (f64, f64, f64) {
    let numerator = s_rb;
    let denominator = s_bb + alpha;
    let u = if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    };
    (u, denominator, numerator)
}

#[inline]
pub fn l2_gain_raw(update: f64, s_rb: f64, s_bb: f64) -> f64 {
    2.0 * update * s_rb - s_bb * update * update
}

#[inline]
pub fn huber_weighting(res: f64, c: f64) -> f64 {
    if res.abs() <= c {
        1.0
    } else {
        c / res.abs()
    }
}

impl RefinementStrategy {
    pub fn initialize<'a>(&'a self, mut state: FittingState<'a>) -> FittingState<'a> {
        // Check if Stage 1 positive-only mode: all residuals are nonnegative
        let is_stage1 = state.residuals.iter().all(|&r| r >= 0.0);

        // Initialize two-tensor lambdas from current outer residuals (labels - stage prediction).
        const EPS_LAMBDA: f64 = 1e-10;

        if is_stage1 {
            // Stage 1 positive-only initialization
            // Set λ_+ = 1 (canonical starting point) or scale-matched value
            // Set λ_- = 0 (fixed, no negative component)
            state.lambda_plus = 1.0;
            state.lambda_minus = 0.0;

            // Ensure all tilt values are zero (already initialized in FittingState::new)
            // But explicitly set them to be safe
            for d_vec in state.tilt_values.iter_mut() {
                for d in d_vec.iter_mut() {
                    *d = 0.0;
                }
            }
        } else {
            // Full two-tensor initialization
            let mut sum_w = 0.0;
            let mut sum_pos = 0.0;
            let mut sum_neg = 0.0;
            for i in 0..state.n {
                let w = self.weight(state.residuals[i]);
                let r = state.residuals[i];
                sum_w += w;
                if r > 0.0 {
                    sum_pos += w * r;
                } else {
                    sum_neg += w * (-r);
                }
            }
            let denom = sum_w.max(EPS_LAMBDA);
            state.lambda_plus = (sum_pos / denom).max(EPS_LAMBDA);
            state.lambda_minus = (sum_neg / denom).max(EPS_LAMBDA);
        }

        // Initialize per-point caches from (b=1, d=0) and lambdas.
        // In Stage 1: f_+ = λ_+ * prod_j b_j, f_- = 0, f = f_+
        // In full mode: f_+ = λ_+ * prod_j b_j * exp(d_j), f_- = λ_- * prod_j b_j * exp(-d_j), f = f_+ - f_-
        state.f_plus.fill(state.lambda_plus);
        state.f_minus.fill(state.lambda_minus);
        state.f.assign(&(&state.f_plus - &state.f_minus));
        state.y_hat.assign(&state.f);
        state
            .residuals
            .assign(&(state.labels.to_owned() - state.y_hat.view()));
        state.r_tilde.assign(&state.residuals);

        for col in 0..state.p {
            let mut indices = (0..state.n).collect::<Vec<_>>();
            indices.sort_by(|&a, &b| state.x[[a, col]].partial_cmp(&state.x[[b, col]]).unwrap());

            for (pos, &i) in indices.iter().enumerate() {
                state.precomputed_statistics.sort_order[col][i] = pos;
            }

            // Compute per-point two-tensor statistics and build prefix sums
            // For two-tensor, we need 5 sufficient statistics:
            //   c_s11[i] = w_i * f_plus[i]²
            //   c_s22[i] = w_i * f_minus[i]²
            //   c_s12[i] = w_i * f_plus[i] * f_minus[i]
            //   c_t1[i] = w_i * r_tilde[i] * f_plus[i]
            //   c_t2[i] = w_i * r_tilde[i] * f_minus[i]
            // We only need to populate per-point contributions once (col == 0), since they're point-specific
            let mut acc = PrefixStats::ZERO;
            let mut prefix: Vec<PrefixStats> = Vec::with_capacity(indices.len());

            for &i in indices.iter() {
                let fp = state.f_plus[i];
                let fm = state.f_minus[i];
                let rt = state.r_tilde[i];
                let w = self.weight(state.residuals[i]);

                // In Stage 1 mode: c_s22 = 0, c_s12 = 0, c_t2 = 0
                // Reuse existing caches but set unused ones to zero
                let c_s11_val = w * fp * fp;
                let c_s22_val = if is_stage1 { 0.0 } else { w * fm * fm };
                let c_s12_val = if is_stage1 { 0.0 } else { w * fp * fm };
                let c_t1_val = w * rt * fp;
                let c_t2_val = if is_stage1 { 0.0 } else { w * rt * fm };

                // Store per-point values (only on first column to avoid redundant writes)
                if col == 0 {
                    state.precomputed_statistics.c_s11[i] = c_s11_val;
                    state.precomputed_statistics.c_s22[i] = c_s22_val;
                    state.precomputed_statistics.c_s12[i] = c_s12_val;
                    state.precomputed_statistics.c_t1[i] = c_t1_val;
                    state.precomputed_statistics.c_t2[i] = c_t2_val;
                }

                acc.s11 += c_s11_val;
                acc.s22 += c_s22_val;
                acc.s12 += c_s12_val;
                acc.t1 += c_t1_val;
                acc.t2 += c_t2_val;
                prefix.push(acc);
            }

            // Store prefix sums and sorted indices BEFORE computing error reductions
            state.precomputed_statistics.prefix_sums[col] = prefix;
            state.precomputed_statistics.sorted_indices[col] = indices;

            // Initialize interval stats: one interval covering all points
            // Note: S12 and t2 have negative signs in the sufficient stats definition
            state.precomputed_statistics.interval_stats[col] =
                vec![crate::grid_tensor::state::IntervalStats {
                    sum_s11: acc.s11,
                    sum_s22: acc.s22,
                    sum_s12: acc.s12, // Will be negated when used: S12 = -sum(c_s12)
                    sum_t1: acc.t1,
                    sum_t2: acc.t2, // Will be negated when used: t2 = -sum(c_t2)
                    n: state.n,
                }];

            // Mark duplicate values as NaN in error_reductions
            state.precomputed_statistics.sorted_indices[col]
                .iter()
                .enumerate()
                .scan(None, |last_value: &mut Option<f64>, (i, &idx)| {
                    let current_value = state.x[[idx, col]];
                    let is_duplicate = last_value
                        .map(|val| (current_value - val).abs() < f64::EPSILON)
                        .unwrap_or(false);
                    *last_value = Some(current_value);
                    Some((i, is_duplicate))
                })
                .filter(|&(_i, is_duplicate)| is_duplicate)
                .for_each(|(i, _)| {
                    state.precomputed_statistics.error_reductions_split[col][i] = f64::NAN;
                    state.precomputed_statistics.update_pairs_split_left[col][i] =
                        (f64::NAN, f64::NAN);
                    state.precomputed_statistics.update_pairs_split_right[col][i] =
                        (f64::NAN, f64::NAN);
                    state.precomputed_statistics.error_reductions_split_pairs[col][i] =
                        (f64::NAN, f64::NAN);
                });
        }

        // Compute error reductions AFTER all columns have been initialized
        state = self.update_error_reductions_split_for_all_cols(state);

        state.precomputed_statistics.initialized = true;
        state
    }

    pub fn update_statistics<'a>(
        &self,
        mut state: FittingState<'a>,
        col: usize,
        start: usize,
        end: usize,
        affected_points_range: &[(usize, usize)],
    ) -> FittingState<'a> {
        let updated_points = &state.precomputed_statistics.sorted_indices[col][start..end];
        if updated_points.is_empty() {
            return state;
        }

        // 1) Compute new per-point two-tensor statistics and their deltas vs stored c_*.
        // Precomputing deltas avoids re-loading old c_* values for every affected column.
        let f_plus = &state.f_plus;
        let f_minus = &state.f_minus;
        let r_tilde = &state.r_tilde;
        let residuals = &state.residuals;

        let (c_s11_new, c_s22_new, c_s12_new, c_t1_new, c_t2_new, d_s11, d_s22, d_s12, d_t1, d_t2) =
            {
                let old_s11 = &state.precomputed_statistics.c_s11;
                let old_s22 = &state.precomputed_statistics.c_s22;
                let old_s12 = &state.precomputed_statistics.c_s12;
                let old_t1 = &state.precomputed_statistics.c_t1;
                let old_t2 = &state.precomputed_statistics.c_t2;

                let mut c_s11_new = Vec::with_capacity(updated_points.len());
                let mut c_s22_new = Vec::with_capacity(updated_points.len());
                let mut c_s12_new = Vec::with_capacity(updated_points.len());
                let mut c_t1_new = Vec::with_capacity(updated_points.len());
                let mut c_t2_new = Vec::with_capacity(updated_points.len());
                let mut d_s11 = Vec::with_capacity(updated_points.len());
                let mut d_s22 = Vec::with_capacity(updated_points.len());
                let mut d_s12 = Vec::with_capacity(updated_points.len());
                let mut d_t1 = Vec::with_capacity(updated_points.len());
                let mut d_t2 = Vec::with_capacity(updated_points.len());

                for &pt in updated_points.iter() {
                    let fp = f_plus[pt];
                    let fm = f_minus[pt];
                    let rt = r_tilde[pt];
                    let w = self.weight(residuals[pt]);

                    let new_s11 = w * fp * fp;
                    let new_s22 = w * fm * fm;
                    let new_s12 = w * fp * fm;
                    let new_t1 = w * rt * fp;
                    let new_t2 = w * rt * fm;

                    c_s11_new.push(new_s11);
                    c_s22_new.push(new_s22);
                    c_s12_new.push(new_s12);
                    c_t1_new.push(new_t1);
                    c_t2_new.push(new_t2);

                    d_s11.push(new_s11 - old_s11[pt]);
                    d_s22.push(new_s22 - old_s22[pt]);
                    d_s12.push(new_s12 - old_s12[pt]);
                    d_t1.push(new_t1 - old_t1[pt]);
                    d_t2.push(new_t2 - old_t2[pt]);
                }

                (
                    c_s11_new, c_s22_new, c_s12_new, c_t1_new, c_t2_new, d_s11, d_s22, d_s12,
                    d_t1, d_t2,
                )
            };

        // 2) For other (non-split) columns, apply delta updates.
        //    Binned columns: scatter deltas into B-length bin buckets and
        //    sweep cumulants over the affected bin range (~B work instead of
        //    ~n work — the main optimization).
        //    Exact columns: original O(span) scatter + sweep over length-n
        //    prefix sums (parity path).
        //
        //    The bucket scatter feeds 5 deltas at offset `off`; we coalesce
        //    those into one `PrefixStats` per bucket slot so the prefix-sum
        //    sweep touches a single 40-byte struct per index instead of
        //    five independent `Vec<f64>`. Order of additions is unchanged.
        let mut bucket_d: Vec<PrefixStats> = Vec::new();
        for (c, &(min_pos, max_pos)) in affected_points_range
            .iter()
            .enumerate()
            .filter(|&(c, _)| c != col)
        {
            if min_pos > max_pos {
                continue;
            }
            let is_binned = !state.precomputed_statistics.bin_edges[c].is_empty();

            if is_binned {
                // ---- Binned path ----
                let bin_edges = &state.precomputed_statistics.bin_edges[c];
                let min_bin = bin_edges.partition_point(|&e| e <= min_pos);
                let max_bin = bin_edges.partition_point(|&e| e <= max_pos);
                let bin_span = max_bin - min_bin + 1;

                bucket_d.clear();
                bucket_d.resize(bin_span, PrefixStats::ZERO);

                let sort_order = &state.precomputed_statistics.sort_order[c];
                for (j, &pt) in updated_points.iter().enumerate() {
                    let pos_full = sort_order[pt];
                    let bin = bin_edges.partition_point(|&e| e <= pos_full);
                    let off = bin - min_bin;
                    let slot = &mut bucket_d[off];
                    slot.s11 += d_s11[j];
                    slot.s22 += d_s22[j];
                    slot.s12 += d_s12[j];
                    slot.t1 += d_t1[j];
                    slot.t2 += d_t2[j];
                }

                let b = &mut state.precomputed_statistics.binned_prefix_sums[c];
                let mut acc_d = PrefixStats::ZERO;
                for off in 0..bin_span {
                    acc_d += bucket_d[off];
                    b[min_bin + off] += acc_d;
                }
                if let Some(tail) = b.get_mut((max_bin + 1)..) {
                    for v in tail {
                        *v += acc_d;
                    }
                }
            } else {
                // ---- Exact path (length-n prefix sums) ----
                let span = max_pos - min_pos + 1;
                bucket_d.clear();
                bucket_d.resize(span, PrefixStats::ZERO);

                let sort_order = &state.precomputed_statistics.sort_order[c];
                let prefix_sums = &mut state.precomputed_statistics.prefix_sums[c];

                for (j, &pt) in updated_points.iter().enumerate() {
                    let pos_full = sort_order[pt];
                    let off = pos_full - min_pos;
                    let slot = &mut bucket_d[off];
                    slot.s11 += d_s11[j];
                    slot.s22 += d_s22[j];
                    slot.s12 += d_s12[j];
                    slot.t1 += d_t1[j];
                    slot.t2 += d_t2[j];
                }

                let mut acc_d = PrefixStats::ZERO;
                let range = &mut prefix_sums[min_pos..=max_pos];
                for off in 0..span {
                    acc_d += bucket_d[off];
                    range[off] += acc_d;
                }

                if let Some(tail) = prefix_sums.get_mut((max_pos + 1)..) {
                    for v in tail {
                        *v += acc_d;
                    }
                }
            }
        }

        // 3) Split column: same path-split as non-split columns, but the
        //    affected range is the full interval [start..end).
        //    Binned path uses the same bin-indexed delta scatter as above.
        //    Exact path keeps the original rebuild-from-new-values logic for
        //    parity with the prior implementation.
        let split_is_binned =
            !state.precomputed_statistics.bin_edges[col].is_empty();

        if split_is_binned {
            // Binned split column: bin-indexed delta scatter, identical to
            // the non-split binned loop above but using (start, end-1).
            let min_pos = start;
            let max_pos = end - 1;

            let bin_edges = &state.precomputed_statistics.bin_edges[col];
            let min_bin = bin_edges.partition_point(|&e| e <= min_pos);
            let max_bin = bin_edges.partition_point(|&e| e <= max_pos);
            let bin_span = max_bin - min_bin + 1;

            bucket_d.clear();
            bucket_d.resize(bin_span, PrefixStats::ZERO);

            let sort_order = &state.precomputed_statistics.sort_order[col];
            for (j, &pt) in updated_points.iter().enumerate() {
                let pos_full = sort_order[pt];
                let bin = bin_edges.partition_point(|&e| e <= pos_full);
                let off = bin - min_bin;
                let slot = &mut bucket_d[off];
                slot.s11 += d_s11[j];
                slot.s22 += d_s22[j];
                slot.s12 += d_s12[j];
                slot.t1 += d_t1[j];
                slot.t2 += d_t2[j];
            }

            let b = &mut state.precomputed_statistics.binned_prefix_sums[col];
            let mut acc_d = PrefixStats::ZERO;
            for off in 0..bin_span {
                acc_d += bucket_d[off];
                b[min_bin + off] += acc_d;
            }
            if let Some(tail) = b.get_mut((max_bin + 1)..) {
                for v in tail {
                    *v += acc_d;
                }
            }
        } else {
            let prefix_sums = &mut state.precomputed_statistics.prefix_sums[col];

            // Exact split column: rebuild the interval [start..end) and shift the tail.
            // Each per-stat prefix is rebuilt by accumulating the matching
            // c_*_new vector — same order of additions as before.
            let min_pos = start;
            let max_pos = end - 1;
            let prev = if min_pos > 0 {
                prefix_sums[min_pos - 1]
            } else {
                PrefixStats::ZERO
            };
            let old_last = prefix_sums[max_pos];

            let mut acc = prev;
            for (k, v) in c_s11_new.iter().enumerate() {
                acc.s11 += *v;
                prefix_sums[min_pos + k].s11 = acc.s11;
            }
            acc.s22 = prev.s22;
            for (k, v) in c_s22_new.iter().enumerate() {
                acc.s22 += *v;
                prefix_sums[min_pos + k].s22 = acc.s22;
            }
            acc.s12 = prev.s12;
            for (k, v) in c_s12_new.iter().enumerate() {
                acc.s12 += *v;
                prefix_sums[min_pos + k].s12 = acc.s12;
            }
            acc.t1 = prev.t1;
            for (k, v) in c_t1_new.iter().enumerate() {
                acc.t1 += *v;
                prefix_sums[min_pos + k].t1 = acc.t1;
            }
            acc.t2 = prev.t2;
            for (k, v) in c_t2_new.iter().enumerate() {
                acc.t2 += *v;
                prefix_sums[min_pos + k].t2 = acc.t2;
            }

            let diff = sub_stats(prefix_sums[max_pos], old_last);
            if let Some(tail) = prefix_sums.get_mut((max_pos + 1)..) {
                for v in tail {
                    *v += diff;
                }
            }
        }

        // 4) Update stored per-point c_* values
        {
            let c_s11 = &mut state.precomputed_statistics.c_s11;
            let c_s22 = &mut state.precomputed_statistics.c_s22;
            let c_s12 = &mut state.precomputed_statistics.c_s12;
            let c_t1 = &mut state.precomputed_statistics.c_t1;
            let c_t2 = &mut state.precomputed_statistics.c_t2;
            for (j, &pt) in updated_points.iter().enumerate() {
                c_s11[pt] = c_s11_new[j];
                c_s22[pt] = c_s22_new[j];
                c_s12[pt] = c_s12_new[j];
                c_t1[pt] = c_t1_new[j];
                c_t2[pt] = c_t2_new[j];
            }
        }

        state
    }

    pub fn refresh_error_reduction_caches_for_affected<'a>(
        &self,
        mut state: FittingState<'a>,
        affected_ranges: &[AffectedRange],
    ) -> FittingState<'a> {
        for affected_range in affected_ranges {
            let c = affected_range.col;
            let (lo, hi) = affected_range.point_range;

            // Update split error reductions using point range
            self.update_error_reductions_split_for_col_range(&mut state, c, lo, hi);

            // Update interval stats and boundary caches using boundary interval range
            // Note: interval_range in AffectedRange is for allowed_intervals,
            // but we need boundary indices for cache refreshing, so compute from point range
            let n_splits = state.boundaries[c].len();
            if n_splits > 0 {
                // Compute boundary interval indices from point range
                let (blo, bhi) = state.compute_boundary_index_range(c, lo, hi);

                // Update interval stats for affected intervals (O(1) lookup for resplit/merge)
                Self::update_interval_stats_for_col_range(&mut state, c, blo, bhi);

                // Convert interval range to boundary range
                // Intervals [blo, bhi] affect boundaries [blo-1, bhi] (clamped to valid boundaries)
                let n_boundaries = state.boundaries[c].len();
                if n_boundaries > 0 {
                    let boundary_lo = blo.saturating_sub(1);
                    let boundary_hi = bhi.min(n_boundaries - 1);

                    self.update_error_reductions_resplit_for_col_range(
                        &mut state,
                        c,
                        boundary_lo,
                        boundary_hi,
                    );
                    // Only update merge error reductions if merge is enabled
                    if state.split_strategy_state.merge_enabled {
                        self.update_error_reductions_merge_for_col_range(
                            &mut state,
                            c,
                            boundary_lo,
                            boundary_hi,
                        );
                    }
                }
            }
        }

        state
    }

    /// Recompute interval stats for affected intervals using prefix sums
    fn update_interval_stats_for_col_range(
        state: &mut FittingState,
        col: usize,
        blo: usize,
        bhi: usize,
    ) {
        let boundaries = &state.boundaries[col];
        let n = state.x.nrows();
        let bin_edges = &state.precomputed_statistics.bin_edges[col];
        let is_binned = !bin_edges.is_empty();

        // Update stats for intervals [blo..=bhi+1] since a boundary affects two adjacent intervals
        let start_interval = blo;
        let end_interval = (bhi + 2).min(state.precomputed_statistics.interval_stats[col].len());

        for interval_idx in start_interval..end_interval {
            let start = if interval_idx == 0 {
                0
            } else {
                boundaries[interval_idx - 1]
            };
            let end = boundaries.get(interval_idx).copied().unwrap_or(n);
            let interval_n = end - start;

            let stats = if is_binned {
                // Binned: start and end are always bin-aligned (splits only
                // happen at bin edges in binned mode). Read directly from
                // length-B cumulants.
                let b = &state.precomputed_statistics.binned_prefix_sums[col];
                binned_prefix_stats_range(b, bin_edges, start, end)
            } else {
                let p = &state.precomputed_statistics.prefix_sums[col];
                prefix_stats_range(p, start, end)
            };

            state.precomputed_statistics.interval_stats[col][interval_idx] =
                crate::grid_tensor::state::IntervalStats {
                    sum_s11: stats.s11,
                    sum_s22: stats.s22,
                    sum_s12: stats.s12,
                    sum_t1: stats.t1,
                    sum_t2: stats.t2,
                    n: interval_n,
                };
        }
    }

    fn update_error_reductions_split_for_all_cols<'a>(
        &self,
        mut state: FittingState<'a>,
    ) -> FittingState<'a> {
        let n = state.x.nrows();
        for col in 0..state.boundaries.len() {
            self.update_error_reductions_split_for_col_range(&mut state, col, 0, n);
        }
        state
    }

    fn update_error_reductions_split_for_col_range(
        &self,
        state: &mut FittingState,
        col: usize,
        lo: usize,
        hi: usize,
    ) {
        let n = state.x.nrows();
        let is_stage1 = state.is_stage1_positive_only();
        let mut start = 0usize;
        for &b in state.boundaries[col].iter().chain(once(&n)) {
            // Computes the error reductions for any interval that intersects with the updated points range

            let end = b;
            if end > lo {
                self.update_error_reductions_split_single_interval(
                    &state.precomputed_statistics.prefix_sums[col],
                    &state.precomputed_statistics.binned_prefix_sums[col],
                    &mut state.precomputed_statistics.update_pairs_split_left[col],
                    &mut state.precomputed_statistics.update_pairs_split_right[col],
                    &mut state.precomputed_statistics.error_reductions_split[col],
                    &mut state.precomputed_statistics.error_reductions_split_pairs[col],
                    &state.precomputed_statistics.bin_edges[col],
                    (start, end),
                    is_stage1,
                );
            }
            start = end;
            if start > hi {
                break;
            }
        }
    }

    fn update_error_reductions_resplit_for_col_range(
        &self,
        state: &mut FittingState,
        col: usize,
        lo_boundary_idx: usize,
        hi_boundary_idx: usize,
    ) {
        let boundary_pos = &state.boundaries[col];
        if boundary_pos.is_empty() {
            return;
        }

        // Ensure all resplit caches are sized to the current number of boundaries
        let target_len = boundary_pos.len();
        if state.precomputed_statistics.error_reductions_resplit[col].len() < target_len {
            state.precomputed_statistics.error_reductions_resplit[col].resize(target_len, f64::NAN);
        }
        if state.precomputed_statistics.update_pairs_resplit_left[col].len() < target_len {
            state.precomputed_statistics.update_pairs_resplit_left[col]
                .resize(target_len, (f64::NAN, f64::NAN));
        }
        if state.precomputed_statistics.update_pairs_resplit_right[col].len() < target_len {
            state.precomputed_statistics.update_pairs_resplit_right[col]
                .resize(target_len, (f64::NAN, f64::NAN));
        }
        if state.precomputed_statistics.error_reductions_resplit_pairs[col].len() < target_len {
            state.precomputed_statistics.error_reductions_resplit_pairs[col]
                .resize(target_len, (f64::NAN, f64::NAN));
        }

        let is_stage1 = state.is_stage1_positive_only();
        let alpha = self.alpha();
        let v_min = DEFAULT_V_MIN;
        let v_max = DEFAULT_V_MAX;

        for i in lo_boundary_idx..=hi_boundary_idx {
            // Use interval stats directly (O(1) lookup instead of prefix_range computation)
            let left_stats = &state.precomputed_statistics.interval_stats[col][i];
            let right_stats = &state.precomputed_statistics.interval_stats[col][i + 1];

            if is_stage1 {
                // Stage 1 positive-only: use 1D ridge solver
                // H^L = S_{11}^L, g^L = t_1^L
                let h_l = left_stats.sum_s11;
                let g_l = left_stats.sum_t1;
                let (u_l, _denom_l, _num_l) = l2_update_unanchored(alpha, g_l, h_l);
                let v_b_l = (1.0 + u_l).clamp(v_min, v_max);
                let u_l_clamped = v_b_l - 1.0;
                let gain_l = l2_gain_raw(u_l_clamped, g_l, h_l);

                // H^R = S_{11}^R, g^R = t_1^R
                let h_r = right_stats.sum_s11;
                let g_r = right_stats.sum_t1;
                let (u_r, _denom_r, _num_r) = l2_update_unanchored(alpha, g_r, h_r);
                let v_b_r = (1.0 + u_r).clamp(v_min, v_max);
                let u_r_clamped = v_b_r - 1.0;
                let gain_r = l2_gain_raw(u_r_clamped, g_r, h_r);

                // Store updates: (u_plus, u_minus) = (u, 0) for Stage 1
                state.precomputed_statistics.error_reductions_resplit[col][i] = gain_l + gain_r;
                state.precomputed_statistics.update_pairs_resplit_left[col][i] = (u_l_clamped, 0.0);
                state.precomputed_statistics.update_pairs_resplit_right[col][i] =
                    (u_r_clamped, 0.0);
                state.precomputed_statistics.error_reductions_resplit_pairs[col][i] =
                    (gain_l, gain_r);
            } else {
                // Full two-tensor: use 2×2 solver
                let tau = self.tilt_tau();
                let rho = self.tilt_rho();

                // Solve for left side
                let (u_plus_l, u_minus_l, gain_l) = solve_two_tensor(
                    left_stats.sum_s11,
                    left_stats.sum_s22,
                    -left_stats.sum_s12,
                    left_stats.sum_t1,
                    -left_stats.sum_t2,
                    alpha,
                    tau,
                    rho,
                    v_min,
                    v_max,
                );

                // Solve for right side
                let (u_plus_r, u_minus_r, gain_r) = solve_two_tensor(
                    right_stats.sum_s11,
                    right_stats.sum_s22,
                    -right_stats.sum_s12,
                    right_stats.sum_t1,
                    -right_stats.sum_t2,
                    alpha,
                    tau,
                    rho,
                    v_min,
                    v_max,
                );

                // Store updates as (u_plus, u_minus) pairs for each side
                // Note: We'll convert to (v_b, delta_d) when applying the split
                state.precomputed_statistics.error_reductions_resplit[col][i] = gain_l + gain_r;
                state.precomputed_statistics.update_pairs_resplit_left[col][i] =
                    (u_plus_l, u_minus_l);
                state.precomputed_statistics.update_pairs_resplit_right[col][i] =
                    (u_plus_r, u_minus_r);
                state.precomputed_statistics.error_reductions_resplit_pairs[col][i] =
                    (gain_l, gain_r);
            }
        }
    }

    fn update_error_reductions_merge_for_col_range(
        &self,
        state: &mut FittingState,
        col: usize,
        lo_boundary_idx: usize,
        hi_boundary_idx: usize,
    ) {
        let boundary_pos = &state.boundaries[col];
        if boundary_pos.is_empty() {
            return;
        }

        // Compute partial products for this axis (efficient: O(n) using divide-out approach)
        let (g_plus, g_minus) = crate::grid_tensor::reducer::compute_partial_products_for_axis(col, state);

        let is_stage1 = state.is_stage1_positive_only();
        let alpha = self.alpha();
        let v_min = DEFAULT_V_MIN;
        let v_max = DEFAULT_V_MAX;

        let sorted_indices = &state.precomputed_statistics.sorted_indices[col];
        for i in lo_boundary_idx..=hi_boundary_idx {
            // Get point ranges for left, right, and union intervals
            let (start, index, end) = state.interval_range_left_and_right(col, i);

            // Get sorted indices for each region (avoid per-boundary allocations)
            let left_region = &sorted_indices[start..index];
            let right_region = &sorted_indices[index..end];

            // Compute stats using partial products (g_{\pm}^{(-j)} regressors)
            let left_stats = crate::grid_tensor::reducer::compute_stats_using_partial_products(
                col,
                left_region,
                &g_plus,
                &g_minus,
                state,
            );
            let right_stats = crate::grid_tensor::reducer::compute_stats_using_partial_products(
                col,
                right_region,
                &g_plus,
                &g_minus,
                state,
            );
            // Union stats can be computed by additivity to avoid a third pass.
            let union_stats = crate::grid_tensor::state::IntervalStats::union(&left_stats, &right_stats);

            // Verify union additivity (I18 invariant) - debug builds only
            #[cfg(debug_assertions)]
            {
                const EPS: f64 = 1e-10;
                let union_region = &sorted_indices[start..end];
                let union_stats_direct = crate::grid_tensor::reducer::compute_stats_using_partial_products(
                    col,
                    union_region,
                    &g_plus,
                    &g_minus,
                    state,
                );
                assert!(
                    (union_stats.sum_s11 - union_stats_direct.sum_s11).abs() < EPS,
                    "I18 violation: S11 union additivity failed for col={}, boundary={}",
                    col,
                    i
                );
                assert!(
                    (union_stats.sum_s22 - union_stats_direct.sum_s22).abs() < EPS,
                    "I18 violation: S22 union additivity failed for col={}, boundary={}",
                    col,
                    i
                );
                assert!(
                    (union_stats.sum_s12 - union_stats_direct.sum_s12).abs() < EPS,
                    "I18 violation: S12 union additivity failed for col={}, boundary={}",
                    col,
                    i
                );
                assert!(
                    (union_stats.sum_t1 - union_stats_direct.sum_t1).abs() < EPS,
                    "I18 violation: t1 union additivity failed for col={}, boundary={}",
                    col,
                    i
                );
                assert!(
                    (union_stats.sum_t2 - union_stats_direct.sum_t2).abs() < EPS,
                    "I18 violation: t2 union additivity failed for col={}, boundary={}",
                    col,
                    i
                );
            }

            // Solve for optimal parameters for left, right, and union
            // Note: We compute gain_l and gain_r here even though error_reductions_resplit_pairs
            // stores (gain_l, gain_r) for this boundary. However, those are computed using
            // interval_stats (f_{\pm} regressors), while we need gains computed using partial
            // products (g_{\pm}^{(-j)} regressors). These are different, so we must recompute.
            // TODO: Consider caching these partial-product-based gains if we need them elsewhere.
            let (gain_l, gain_r, u_plus_merged, u_minus_merged, gain_merged) = if is_stage1 {
                // Stage 1 positive-only: use 1D ridge solver
                // Left side: H^L = S_{11}^L, g^L = t_1^L
                let h_l = left_stats.sum_s11;
                let g_l = left_stats.sum_t1;
                let (u_l, _denom_l, _num_l) = l2_update_unanchored(alpha, g_l, h_l);
                let v_b_l = (1.0 + u_l).clamp(v_min, v_max);
                let u_l_clamped = v_b_l - 1.0;
                let gain_l = l2_gain_raw(u_l_clamped, g_l, h_l);

                // Right side: H^R = S_{11}^R, g^R = t_1^R
                let h_r = right_stats.sum_s11;
                let g_r = right_stats.sum_t1;
                let (u_r, _denom_r, _num_r) = l2_update_unanchored(alpha, g_r, h_r);
                let v_b_r = (1.0 + u_r).clamp(v_min, v_max);
                let u_r_clamped = v_b_r - 1.0;
                let gain_r = l2_gain_raw(u_r_clamped, g_r, h_r);

                // Union: H^U = S_{11}^U, g^U = t_1^U
                let h_u = union_stats.sum_s11;
                let g_u = union_stats.sum_t1;
                let (u_u, _denom_u, _num_u) = l2_update_unanchored(alpha, g_u, h_u);
                let v_b_u = (1.0 + u_u).clamp(v_min, v_max);
                let u_u_clamped = v_b_u - 1.0;
                let gain_merged = l2_gain_raw(u_u_clamped, g_u, h_u);

                (gain_l, gain_r, u_u_clamped, 0.0, gain_merged)
            } else {
                // Full two-tensor: use 2×2 solver
                let tau = self.tilt_tau();
                let rho = self.tilt_rho();

                let (_u_plus_l, _u_minus_l, gain_l) = solve_two_tensor(
                    left_stats.sum_s11,
                    left_stats.sum_s22,
                    -left_stats.sum_s12,
                    left_stats.sum_t1,
                    -left_stats.sum_t2,
                    alpha,
                    tau,
                    rho,
                    v_min,
                    v_max,
                );

                let (_u_plus_r, _u_minus_r, gain_r) = solve_two_tensor(
                    right_stats.sum_s11,
                    right_stats.sum_s22,
                    -right_stats.sum_s12,
                    right_stats.sum_t1,
                    -right_stats.sum_t2,
                    alpha,
                    tau,
                    rho,
                    v_min,
                    v_max,
                );

                let (u_plus_merged, u_minus_merged, gain_merged) = solve_two_tensor(
                    union_stats.sum_s11,
                    union_stats.sum_s22,
                    -union_stats.sum_s12,
                    union_stats.sum_t1,
                    -union_stats.sum_t2,
                    alpha,
                    tau,
                    rho,
                    v_min,
                    v_max,
                );

                (gain_l, gain_r, u_plus_merged, u_minus_merged, gain_merged)
            };

            // Verify score dominance (I19 invariant) - debug builds only
            #[cfg(debug_assertions)]
            {
                const EPS: f64 = 1e-8;
                let gain_children = gain_l + gain_r;
                assert!(
                    gain_children >= gain_merged - EPS,
                    "I19 violation: Score dominance failed for col={}, boundary={}: g_A+g_B={}, g_U={}",
                    col, i, gain_children, gain_merged
                );
            }

            // Store boundary benefit as merge gain: Δ_boundary = (g_A + g_B) - g_U
            // Negative values mean merge improves objective (boundary not worth it)
            // Positive values mean keeping boundary improves objective
            let boundary_benefit = (gain_l + gain_r) - gain_merged;

            // Merge gain is the negative of boundary benefit (merge improves when boundary_benefit < 0)
            let merge_gain = -boundary_benefit;

            // Store (u_plus, u_minus) for merged interval (used when applying merge)
            state.precomputed_statistics.error_reductions_merge[col][i] = merge_gain;
            state.precomputed_statistics.update_pairs_merge[col][i] =
                (u_plus_merged, u_minus_merged);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn update_error_reductions_split_single_interval(
        &self,
        ps: &[PrefixStats],
        bs: &[PrefixStats],
        update_left: &mut [(f64, f64)],
        update_right: &mut [(f64, f64)],
        error_reductions: &mut [f64],
        error_reductions_pairs: &mut [(f64, f64)],
        bin_edges: &[usize],
        spanned_interval: (usize, usize),
        is_stage1: bool,
    ) {
        let (start, end) = spanned_interval;
        let len = end.saturating_sub(start);
        if len < 2 {
            return;
        }

        // Disallow splitting at the first position.
        update_left[start] = (f64::NAN, f64::NAN);
        update_right[start] = (f64::NAN, f64::NAN);
        error_reductions[start] = f64::NAN;
        error_reductions_pairs[start] = (f64::NAN, f64::NAN);

        let alpha = self.alpha();
        let v_min = DEFAULT_V_MIN;
        let v_max = DEFAULT_V_MAX;
        let tau = self.tilt_tau();
        let rho = self.tilt_rho();

        // Evaluate a single candidate position given left-side sufficient stats.
        // Closed over the output slices + total stats. Shared between the
        // exact and binned iteration paths.
        let mut eval_pos = |pos: usize,
                            s11_l: f64,
                            s22_l: f64,
                            s12_l: f64,
                            t1_l: f64,
                            t2_l: f64,
                            total_s11: f64,
                            total_s22: f64,
                            total_s12: f64,
                            total_t1: f64,
                            total_t2: f64,
                            update_left: &mut [(f64, f64)],
                            update_right: &mut [(f64, f64)],
                            error_reductions: &mut [f64],
                            error_reductions_pairs: &mut [(f64, f64)]| {
            if error_reductions[pos].is_nan() {
                update_left[pos] = (f64::NAN, f64::NAN);
                update_right[pos] = (f64::NAN, f64::NAN);
                error_reductions_pairs[pos] = (f64::NAN, f64::NAN);
                return;
            }

            if is_stage1 {
                let h_l = s11_l;
                let g_l = t1_l;
                let h_r = total_s11 - h_l;
                let g_r = total_t1 - g_l;

                let (u_l, _denom_l, _num_l) = l2_update_unanchored(alpha, g_l, h_l);
                let v_b_l = (1.0 + u_l).clamp(v_min, v_max);
                let u_l_clamped = v_b_l - 1.0;
                let gain_l = l2_gain_raw(u_l_clamped, g_l, h_l);

                let (u_r, _denom_r, _num_r) = l2_update_unanchored(alpha, g_r, h_r);
                let v_b_r = (1.0 + u_r).clamp(v_min, v_max);
                let u_r_clamped = v_b_r - 1.0;
                let gain_r = l2_gain_raw(u_r_clamped, g_r, h_r);

                update_left[pos] = (u_l_clamped, 0.0);
                update_right[pos] = (u_r_clamped, 0.0);
                error_reductions[pos] = gain_l + gain_r;
                error_reductions_pairs[pos] = (gain_l, gain_r);
            } else {
                let s11_r = total_s11 - s11_l;
                let s22_r = total_s22 - s22_l;
                let s12_r = total_s12 - s12_l;
                let t1_r = total_t1 - t1_l;
                let t2_r = total_t2 - t2_l;

                let (u_plus_l, u_minus_l, gain_l) = solve_two_tensor(
                    s11_l, s22_l, s12_l, t1_l, t2_l, alpha, tau, rho, v_min, v_max,
                );
                let (u_plus_r, u_minus_r, gain_r) = solve_two_tensor(
                    s11_r, s22_r, s12_r, t1_r, t2_r, alpha, tau, rho, v_min, v_max,
                );

                update_left[pos] = (u_plus_l, u_minus_l);
                update_right[pos] = (u_plus_r, u_minus_r);
                error_reductions[pos] = gain_l + gain_r;
                error_reductions_pairs[pos] = (gain_l, gain_r);
            }
        };

        if bin_edges.is_empty() {
            // Exact path: read length-n prefix sums at every candidate position.
            let base = if start == 0 {
                PrefixStats::ZERO
            } else {
                ps[start - 1]
            };

            let total_s11 = ps[end - 1].s11 - base.s11;
            let total_s22 = ps[end - 1].s22 - base.s22;
            let total_s12 = -(ps[end - 1].s12 - base.s12);
            let total_t1 = ps[end - 1].t1 - base.t1;
            let total_t2 = -(ps[end - 1].t2 - base.t2);

            for pos in (start + 1)..end.saturating_sub(1) {
                let pos_end = pos - 1;
                let s11_l = ps[pos_end].s11 - base.s11;
                let s22_l = ps[pos_end].s22 - base.s22;
                let s12_l = -(ps[pos_end].s12 - base.s12);
                let t1_l = ps[pos_end].t1 - base.t1;
                let t2_l = -(ps[pos_end].t2 - base.t2);
                eval_pos(
                    pos,
                    s11_l,
                    s22_l,
                    s12_l,
                    t1_l,
                    t2_l,
                    total_s11,
                    total_s22,
                    total_s12,
                    total_t1,
                    total_t2,
                    update_left,
                    update_right,
                    error_reductions,
                    error_reductions_pairs,
                );
            }
        } else {
            // Binned path: read length-B bin-cumulative prefix sums at bin
            // boundaries. Splits land at `bin_edges[k]` so `pos_end = pos - 1`
            // is the last position of bin k → cumulant at bin index k.
            let base = if start == 0 {
                PrefixStats::ZERO
            } else {
                binned_prefix_stats_at(bs, bin_edges, start - 1)
            };
            let end_v = binned_prefix_stats_at(bs, bin_edges, end - 1);

            let total_s11 = end_v.s11 - base.s11;
            let total_s22 = end_v.s22 - base.s22;
            let total_s12 = -(end_v.s12 - base.s12);
            let total_t1 = end_v.t1 - base.t1;
            let total_t2 = -(end_v.t2 - base.t2);

            let lo_idx = bin_edges.partition_point(|&e| e <= start);
            let hi_idx = bin_edges.partition_point(|&e| e < end.saturating_sub(1));
            for k in lo_idx..hi_idx {
                let pos = bin_edges[k];
                let s11_l = bs[k].s11 - base.s11;
                let s22_l = bs[k].s22 - base.s22;
                let s12_l = -(bs[k].s12 - base.s12);
                let t1_l = bs[k].t1 - base.t1;
                let t2_l = -(bs[k].t2 - base.t2);
                eval_pos(
                    pos,
                    s11_l,
                    s22_l,
                    s12_l,
                    t1_l,
                    t2_l,
                    total_s11,
                    total_s22,
                    total_s12,
                    total_t1,
                    total_t2,
                    update_left,
                    update_right,
                    error_reductions,
                    error_reductions_pairs,
                );
            }
        }

        // Disallow splitting at the last position.
        let last = end - 1;
        update_left[last] = (f64::NAN, f64::NAN);
        update_right[last] = (f64::NAN, f64::NAN);
        error_reductions[last] = f64::NAN;
        error_reductions_pairs[last] = (f64::NAN, f64::NAN);
    }

    pub fn alpha(&self) -> f64 {
        match self {
            RefinementStrategy::L2Refinement { alpha, .. } => *alpha,
            RefinementStrategy::HuberRefinement { alpha, .. } => *alpha,
        }
    }

    pub fn weight(&self, res: f64) -> f64 {
        match self {
            RefinementStrategy::L2Refinement { .. } => 1.0,
            RefinementStrategy::HuberRefinement { c, .. } => huber_weighting(res, *c),
        }
    }

    /// Returns the prior sample size (tau_0) for parent anchoring.
    ///
    /// This represents "how many samples worth of confidence we have that
    /// children should equal their parent". With tau_0 = 30, a child interval
    /// with 10 samples will be heavily shrunk toward the parent, while a child
    /// with 100 samples will mostly trust its own data.
    pub fn prior_sample_size(&self) -> f64 {
        match self {
            RefinementStrategy::L2Refinement {
                prior_sample_size, ..
            } => *prior_sample_size,
            RefinementStrategy::HuberRefinement {
                prior_sample_size, ..
            } => *prior_sample_size,
        }
    }

    pub fn update_clamp(&self) -> f64 {
        match self {
            RefinementStrategy::L2Refinement { update_clamp, .. } => *update_clamp,
            RefinementStrategy::HuberRefinement { update_clamp, .. } => *update_clamp,
        }
    }

    /// Compute the gain for setting the multiplier to zero (u = -1).
    pub fn gain_for_zero_multiplier(&self, s_rb: f64, s_bb: f64, _n: f64) -> f64 {
        // When forcing multiplier to 0, the additive update is u = 0 - 1 = -1.
        // gain = 2u S_rb - u^2 S_bb = 2(-1)S_rb - (-1)^2 S_bb = -2 S_rb - S_bb.
        let u = -1.0;
        l2_gain_raw(u, s_rb, s_bb)
    }
}

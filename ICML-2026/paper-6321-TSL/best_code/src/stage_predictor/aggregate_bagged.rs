//! Bagged Two-Tensor TSL Aggregation
//!
//! The procedure includes:
//! 1. Common grid alignment per axis
//! 2. Gauge fixing / canonicalization per bag
//! 3. Component-shape distance computation
//! 4. Median tensor selection (medoid)
//! 5. Robust averaging of kept components
//! 6. Post-aggregation normalization

use crate::stage_predictor::combine_grids::refine_grids_to_union_two_tensor;
use crate::grid_tensor::identification::l2_identify;
use crate::grid_tensor::GridTensor;
use crate::logging::log_combination_choice;
use ndarray::{ArrayView1, ArrayView2};

#[cfg(feature = "use-rayon")]
use rayon::prelude::*;

/// Numerical stability constants
const EPSILON: f64 = 1e-10;
const LOG_EPSILON: f64 = -23.025850929940457; // ln(1e-10)
const LOG_MAX: f64 = 23.025850929940457; // ln(1e10)
const EPSILON_N: f64 = 1e-12; // For empty bin weights

fn compute_a_factors_from_bd(
    backbone: &[Vec<f64>],
    tilt: &[Vec<f64>],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let num_axes = backbone.len();
    let mut a_plus: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
    let mut a_minus: Vec<Vec<f64>> = Vec::with_capacity(num_axes);

    for axis in 0..num_axes {
        let n_bins = backbone[axis].len();
        let mut a_plus_axis = Vec::with_capacity(n_bins);
        let mut a_minus_axis = Vec::with_capacity(n_bins);

        for bin in 0..n_bins {
            let b = backbone[axis][bin].max(EPSILON);
            let d = tilt[axis][bin];
            let exp_d = d.min(50.0).exp();
            let exp_neg_d = (-d).min(50.0).exp();
            a_plus_axis.push(b * exp_d);
            a_minus_axis.push(b * exp_neg_d);
        }
        a_plus.push(a_plus_axis);
        a_minus.push(a_minus_axis);
    }

    (a_plus, a_minus)
}

fn geometric_mean_combine_a_factors(
    a_plus_candidates: &[Vec<Vec<f64>>],
    a_minus_candidates: &[Vec<Vec<f64>>],
    weights: &[f64],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    if a_plus_candidates.is_empty() {
        panic!("Cannot combine empty candidate grids");
    }

    let num_axes = a_plus_candidates[0].len();
    let total_weight: f64 = weights.iter().sum();

    if total_weight <= 0.0 {
        panic!("Total weight must be positive");
    }

    let mut combined_a_plus: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
    let mut combined_a_minus: Vec<Vec<f64>> = Vec::with_capacity(num_axes);

    for axis in 0..num_axes {
        let n_bins = a_plus_candidates[0][axis].len();
        let mut a_plus_axis = Vec::with_capacity(n_bins);
        let mut a_minus_axis = Vec::with_capacity(n_bins);

        for bin in 0..n_bins {
            let mut log_sum_plus = 0.0;
            for (a_plus_cand, &weight) in a_plus_candidates.iter().zip(weights) {
                let a_val = a_plus_cand[axis][bin].max(EPSILON);
                let log_val = a_val.ln().max(LOG_EPSILON).min(LOG_MAX);
                log_sum_plus += weight * log_val;
            }
            let combined_a_plus_val = (log_sum_plus / total_weight).exp();

            let mut log_sum_minus = 0.0;
            for (a_minus_cand, &weight) in a_minus_candidates.iter().zip(weights) {
                let a_val = a_minus_cand[axis][bin].max(EPSILON);
                let log_val = a_val.ln().max(LOG_EPSILON).min(LOG_MAX);
                log_sum_minus += weight * log_val;
            }
            let combined_a_minus_val = (log_sum_minus / total_weight).exp();

            a_plus_axis.push(combined_a_plus_val);
            a_minus_axis.push(combined_a_minus_val);
        }

        combined_a_plus.push(a_plus_axis);
        combined_a_minus.push(a_minus_axis);
    }

    (combined_a_plus, combined_a_minus)
}

fn convert_a_factors_to_bd(
    a_plus: &[Vec<f64>],
    a_minus: &[Vec<f64>],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let num_axes = a_plus.len();
    let mut backbone: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
    let mut tilt: Vec<Vec<f64>> = Vec::with_capacity(num_axes);

    for axis in 0..num_axes {
        let n_bins = a_plus[axis].len();
        let mut backbone_axis = Vec::with_capacity(n_bins);
        let mut tilt_axis = Vec::with_capacity(n_bins);

        for bin in 0..n_bins {
            let a_p = a_plus[axis][bin].max(EPSILON);
            let a_m = a_minus[axis][bin].max(EPSILON);
            let b = (a_p * a_m).sqrt();
            let ratio = (a_p / a_m).max(EPSILON).min(1.0 / EPSILON);
            let d = 0.5 * ratio.ln();
            backbone_axis.push(b);
            tilt_axis.push(d);
        }

        backbone.push(backbone_axis);
        tilt.push(tilt_axis);
    }

    (backbone, tilt)
}

fn center_tilt_per_axis(tilt_values: &mut [Vec<f64>], observation_counts: &[Vec<usize>]) {
    const EPS: f64 = 1e-12;

    for dim in 0..tilt_values.len() {
        let counts = &observation_counts[dim];
        let d = &mut tilt_values[dim];

        if counts.is_empty() || d.is_empty() {
            continue;
        }

        let weights_sum: f64 = counts.iter().map(|&c| c as f64).sum();
        if weights_sum <= EPS {
            continue;
        }

        let mean = d
            .iter()
            .zip(counts.iter())
            .map(|(&x, &c)| x * (c as f64))
            .sum::<f64>()
            / weights_sum;

        for x in d.iter_mut() {
            *x -= mean;
        }
    }
}

fn canonicalize_backbone_log_space(
    backbone_values: &mut [Vec<f64>],
    observation_counts: &[Vec<usize>],
) {
    const EPS: f64 = 1e-12;

    for dim in 0..backbone_values.len() {
        let counts = &observation_counts[dim];
        let b = &mut backbone_values[dim];

        if counts.is_empty() || b.is_empty() {
            continue;
        }

        let weights_sum: f64 = counts.iter().map(|&c| c as f64).sum();
        if weights_sum <= EPS {
            continue;
        }

        let mean_log_b = b
            .iter()
            .zip(counts.iter())
            .map(|(&x, &c)| {
                let log_x = x.max(EPSILON).ln().max(LOG_EPSILON).min(LOG_MAX);
                log_x * (c as f64)
            })
            .sum::<f64>()
            / weights_sum;

        for x in b.iter_mut() {
            let log_x = x.max(EPSILON).ln().max(LOG_EPSILON).min(LOG_MAX);
            let centered_log_x = log_x - mean_log_b;
            *x = centered_log_x.exp();
        }
    }
}

fn canonicalize_bag_per_axis(grid: &mut GridTensor, observation_counts: &[Vec<usize>]) {
    canonicalize_backbone_log_space(&mut grid.backbone_values, observation_counts);
    center_tilt_per_axis(&mut grid.tilt_values, observation_counts);
}

#[derive(Clone)]
struct PrecomputedGrid {
    log_f_plus: Vec<Vec<f64>>,
    log_f_minus: Vec<Vec<f64>>,
}

fn precompute_bin_weights(observation_counts: &[Vec<usize>]) -> Vec<Vec<f64>> {
    observation_counts
        .iter()
        .map(|axis_counts| {
            axis_counts
                .iter()
                .map(|&count| if count == 0 { EPSILON_N } else { count as f64 })
                .collect()
        })
        .collect()
}

fn precompute_log_components(grid: &GridTensor) -> PrecomputedGrid {
    let mut log_f_plus: Vec<Vec<f64>> = Vec::with_capacity(grid.backbone_values.len());
    let mut log_f_minus: Vec<Vec<f64>> = Vec::with_capacity(grid.backbone_values.len());

    for axis in 0..grid.backbone_values.len() {
        let backbone = &grid.backbone_values[axis];
        let tilt = &grid.tilt_values[axis];
        let mut plus_axis = Vec::with_capacity(backbone.len());
        let mut minus_axis = Vec::with_capacity(backbone.len());

        for bin in 0..backbone.len() {
            let b = backbone[bin].max(EPSILON);
            let d = tilt[bin];
            let log_b = b.ln();

            let p = (log_b + d.min(50.0)).max(LOG_EPSILON).min(LOG_MAX);
            let m = (log_b + (-d).min(50.0)).max(LOG_EPSILON).min(LOG_MAX);

            plus_axis.push(p);
            minus_axis.push(m);
        }

        log_f_plus.push(plus_axis);
        log_f_minus.push(minus_axis);
    }

    PrecomputedGrid {
        log_f_plus,
        log_f_minus,
    }
}

fn compute_component_shape_distance_precomputed(
    grid_a: &PrecomputedGrid,
    grid_b: &PrecomputedGrid,
    bin_weights: &[Vec<f64>],
) -> f64 {
    debug_assert_eq!(grid_a.log_f_plus.len(), grid_b.log_f_plus.len());
    debug_assert_eq!(grid_a.log_f_plus.len(), bin_weights.len());

    let mut total_distance_sq = 0.0;

    for axis in 0..grid_a.log_f_plus.len() {
        let p_a = &grid_a.log_f_plus[axis];
        let m_a = &grid_a.log_f_minus[axis];
        let p_b = &grid_b.log_f_plus[axis];
        let m_b = &grid_b.log_f_minus[axis];
        let weights = &bin_weights[axis];

        debug_assert_eq!(p_a.len(), p_b.len());
        debug_assert_eq!(p_a.len(), weights.len());

        let mut axis_distance_sq = 0.0;

        for bin in 0..p_a.len() {
            let w = weights[bin];
            let delta_p = p_a[bin] - p_b[bin];
            let delta_m = m_a[bin] - m_b[bin];
            axis_distance_sq += w * (delta_p * delta_p + delta_m * delta_m);
        }

        total_distance_sq += axis_distance_sq;
    }

    total_distance_sq.sqrt()
}

#[cfg(not(feature = "use-rayon"))]
fn select_median_tensor_component_shape(
    grids: &[PrecomputedGrid],
    bin_weights: &[Vec<f64>],
) -> (usize, Vec<Vec<f64>>) {
    if grids.is_empty() {
        panic!("Cannot select median from empty grid list");
    }
    if grids.len() == 1 {
        return (0, vec![vec![0.0; 1]; 1]);
    }

    let n = grids.len();
    let mut min_total_distance = f64::INFINITY;
    let mut median_index = 0;

    let mut distance_matrix: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

    log::info!(
        "Computing pairwise component-shape distances for {} bags",
        n
    );
    for i in 0..n {
        for j in (i + 1)..n {
            let dist =
                compute_component_shape_distance_precomputed(&grids[i], &grids[j], bin_weights);
            distance_matrix[i][j] = dist;
            distance_matrix[j][i] = dist;
        }
    }

    log::info!("Finding median bag (medoid) from distance matrix");
    for i in 0..n {
        let total_distance: f64 = distance_matrix[i].iter().sum();
        if total_distance < min_total_distance {
            min_total_distance = total_distance;
            median_index = i;
        }
    }

    log::info!(
        "Selected grid {} as median (medoid) with total distance {:.6}",
        median_index,
        min_total_distance
    );
    (median_index, distance_matrix)
}

#[cfg(feature = "use-rayon")]
fn select_median_tensor_component_shape(
    grids: &[PrecomputedGrid],
    bin_weights: &[Vec<f64>],
) -> (usize, Vec<Vec<f64>>) {
    if grids.is_empty() {
        panic!("Cannot select median from empty grid list");
    }
    if grids.len() == 1 {
        return (0, vec![vec![0.0; 1]; 1]);
    }

    let n = grids.len();
    let mut distance_matrix: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

    log::info!(
        "Computing pairwise component-shape distances for {} bags (parallel)",
        n
    );

    let upper_triangle: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            ((i + 1)..n).into_par_iter().map(move |j| {
                let dist =
                    compute_component_shape_distance_precomputed(&grids[i], &grids[j], bin_weights);
                (i, j, dist)
            })
        })
        .collect();

    for (i, j, dist) in upper_triangle {
        distance_matrix[i][j] = dist;
        distance_matrix[j][i] = dist;
    }

    log::info!("Finding median bag (medoid) from distance matrix");
    let (median_index, min_total_distance) = (0..n)
        .map(|i| {
            let total_distance: f64 = distance_matrix[i].iter().sum();
            (i, total_distance)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    log::info!(
        "Selected grid {} as median (medoid) with total distance {:.6}",
        median_index,
        min_total_distance
    );
    (median_index, distance_matrix)
}

fn trim_outliers(
    grids: &[GridTensor],
    median_index: usize,
    distance_matrix: &[Vec<f64>],
    trim_percentage: f64,
) -> Vec<usize> {
    if grids.is_empty() {
        return Vec::new();
    }
    if grids.len() == 1 {
        return vec![0];
    }

    let n = grids.len();
    let keep_count = (trim_percentage * n as f64).ceil() as usize;
    let keep_count = keep_count.min(n);

    log::info!(
        "Using pre-computed distances to median bag {}",
        median_index
    );
    let mut distances: Vec<(usize, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let dist = distance_matrix[i][median_index];
        distances.push((i, dist));
    }

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let kept: Vec<usize> = distances[..keep_count]
        .iter()
        .map(|(idx, _)| *idx)
        .collect();

    log::info!(
        "Selected {} out of {} grids (trim_percentage={:.2}, keep_count={})",
        kept.len(),
        n,
        trim_percentage,
        keep_count
    );

    kept
}

fn combine_lambdas_geometric_mean(
    lambda_plus_candidates: &[f64],
    lambda_minus_candidates: &[f64],
    weights: Option<&[f64]>,
) -> (f64, f64) {
    if lambda_plus_candidates.is_empty() {
        panic!("Cannot combine empty lambda list");
    }

    let n = lambda_plus_candidates.len();
    let uniform_weights: Vec<f64> = vec![1.0; n];
    let weights = weights.unwrap_or(&uniform_weights);

    if weights.len() != n {
        panic!("Weights length must match lambdas length");
    }

    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        panic!("Total weight must be positive");
    }

    let log_sum_plus: f64 = lambda_plus_candidates
        .iter()
        .zip(weights.iter())
        .map(|(&l, &w)| {
            let log_l = l.max(EPSILON).ln().max(LOG_EPSILON).min(LOG_MAX);
            log_l * w
        })
        .sum();
    let combined_lambda_plus = (log_sum_plus / total_weight).exp();

    let log_sum_minus: f64 = lambda_minus_candidates
        .iter()
        .zip(weights.iter())
        .map(|(&l, &w)| {
            let log_l = l.max(EPSILON).ln().max(LOG_EPSILON).min(LOG_MAX);
            log_l * w
        })
        .sum();
    let combined_lambda_minus = (log_sum_minus / total_weight).exp();

    (combined_lambda_plus, combined_lambda_minus)
}

pub fn aggregate_bagged_two_tensor(
    grids: &[GridTensor],
    points: ArrayView2<f64>,
    _weights: Option<ArrayView1<f64>>,
    trim_percentage: f64,
) -> GridTensor {
    if grids.is_empty() {
        panic!("Cannot aggregate empty grid list");
    }
    if grids.len() == 1 {
        return grids[0].clone();
    }

    // Step 1: Align all grids to union grid
    let mut aligned_grids = refine_grids_to_union_two_tensor(grids);

    // Step 2: Recompute observation counts for the union grid
    let num_axes = aligned_grids[0].intervals.len();
    let mut union_observation_counts: Vec<Vec<usize>> = Vec::with_capacity(num_axes);
    for axis in 0..num_axes {
        let n_bins = aligned_grids[0].intervals[axis].len();
        let mut counts: Vec<usize> = vec![0; n_bins];
        let splits = &aligned_grids[0].splits[axis];

        let mut vals: Vec<f64> = points.column(axis).iter().copied().collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut s_idx: usize = 0;
        let b_len = splits.len();
        for v in vals {
            while s_idx < b_len && v >= splits[s_idx] {
                s_idx += 1;
            }
            counts[s_idx] += 1;
        }

        union_observation_counts.push(counts);
    }

    // Step 3: Per-bag canonicalization
    for grid in &mut aligned_grids {
        canonicalize_bag_per_axis(grid, &union_observation_counts);
    }

    let (_kept_indices, kept_grids) = if trim_percentage >= 1.0 {
        log::info!(
            "trim_percentage is {:.2}, combining all {} grids without distance computation",
            trim_percentage,
            aligned_grids.len()
        );
        let all_indices: Vec<usize> = (0..aligned_grids.len()).collect();
        let candidate_indices: Vec<(usize, f64)> =
            all_indices.iter().map(|&idx| (idx, 0.0)).collect();
        log_combination_choice("BaggedTwoTensor", None, &candidate_indices);
        let kept: Vec<&GridTensor> = aligned_grids.iter().collect();
        (all_indices, kept)
    } else {
        let bin_weights = precompute_bin_weights(&union_observation_counts);
        let precomputed_grids: Vec<PrecomputedGrid> = aligned_grids
            .iter()
            .map(precompute_log_components)
            .collect();

        let (median_index, distance_matrix) =
            select_median_tensor_component_shape(&precomputed_grids, &bin_weights);

        let kept_indices = trim_outliers(
            &aligned_grids,
            median_index,
            &distance_matrix,
            trim_percentage,
        );

        let candidate_indices: Vec<(usize, f64)> = kept_indices
            .iter()
            .map(|&idx| (idx, distance_matrix[idx][median_index]))
            .collect();
        log_combination_choice("BaggedTwoTensor", Some(median_index), &candidate_indices);

        let kept_grids: Vec<&GridTensor> =
            kept_indices.iter().map(|&i| &aligned_grids[i]).collect();
        (kept_indices, kept_grids)
    };

    // Step 7: Geometric mean of a_± factors on kept set
    let mut a_plus_candidates: Vec<Vec<Vec<f64>>> = Vec::with_capacity(kept_grids.len());
    let mut a_minus_candidates: Vec<Vec<Vec<f64>>> = Vec::with_capacity(kept_grids.len());
    let mut lambda_plus_candidates: Vec<f64> = Vec::with_capacity(kept_grids.len());
    let mut lambda_minus_candidates: Vec<f64> = Vec::with_capacity(kept_grids.len());

    for grid in &kept_grids {
        let (a_plus, a_minus) = compute_a_factors_from_bd(&grid.backbone_values, &grid.tilt_values);
        a_plus_candidates.push(a_plus);
        a_minus_candidates.push(a_minus);
        lambda_plus_candidates.push(grid.lambda_plus);
        lambda_minus_candidates.push(grid.lambda_minus);
    }

    let kept_weights: Vec<f64> = vec![1.0; kept_grids.len()];
    let (combined_a_plus, combined_a_minus) =
        geometric_mean_combine_a_factors(&a_plus_candidates, &a_minus_candidates, &kept_weights);

    // Step 8: Reconstruct (b, d)
    let (combined_backbone, combined_tilt) =
        convert_a_factors_to_bd(&combined_a_plus, &combined_a_minus);

    // Step 9: Post-aggregation normalization
    let mut temp_backbone = combined_backbone.clone();
    let mut temp_tilt = combined_tilt.clone();
    let mut temp_lambda_plus = 1.0;
    let mut temp_lambda_minus = 1.0;

    l2_identify(
        &mut temp_backbone,
        &mut temp_tilt,
        &union_observation_counts,
        &mut temp_lambda_plus,
        &mut temp_lambda_minus,
    );

    // Step 10: Geometric mean of lambdas
    let (combined_lambda_plus, combined_lambda_minus) =
        combine_lambdas_geometric_mean(&lambda_plus_candidates, &lambda_minus_candidates, None);

    GridTensor::new_two_tensor(
        aligned_grids[0].splits.clone(),
        union_observation_counts,
        aligned_grids[0].intervals.clone(),
        temp_backbone,
        temp_tilt,
        combined_lambda_plus,
        combined_lambda_minus,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_simple_grid(
        backbone: Vec<Vec<f64>>,
        tilt: Vec<Vec<f64>>,
        lambda_plus: f64,
        lambda_minus: f64,
        splits: Vec<Vec<f64>>,
        intervals: Vec<Vec<(f64, f64)>>,
    ) -> GridTensor {
        let observation_counts: Vec<Vec<usize>> =
            backbone.iter().map(|axis| vec![10; axis.len()]).collect();
        GridTensor::new_two_tensor(
            splits,
            observation_counts,
            intervals,
            backbone,
            tilt,
            lambda_plus,
            lambda_minus,
        )
    }

    #[test]
    fn test_component_shape_distance_identical() {
        let intervals = vec![vec![(0.0, 1.0)], vec![(0.0, 1.0)]];
        let splits = vec![vec![], vec![]];
        let grid1 = create_simple_grid(
            vec![vec![1.0], vec![1.0]],
            vec![vec![0.0], vec![0.0]],
            1.0,
            0.5,
            splits.clone(),
            intervals.clone(),
        );
        let grid2 = create_simple_grid(
            vec![vec![1.0], vec![1.0]],
            vec![vec![0.0], vec![0.0]],
            1.0,
            0.5,
            splits,
            intervals.clone(),
        );
        let bin_weights = precompute_bin_weights(&grid1.observation_counts);
        let grid1_pre = precompute_log_components(&grid1);
        let grid2_pre = precompute_log_components(&grid2);
        let distance =
            compute_component_shape_distance_precomputed(&grid1_pre, &grid2_pre, &bin_weights);
        assert!(
            distance < 1e-10,
            "Identical grids should have distance ≈ 0, got {}",
            distance
        );
    }

    #[test]
    fn test_select_median_single_bag() {
        let intervals = vec![vec![(0.0, 1.0)]];
        let splits = vec![vec![]];
        let grid = create_simple_grid(
            vec![vec![1.0]],
            vec![vec![0.0]],
            1.0,
            0.5,
            splits,
            intervals.clone(),
        );
        let grids = [grid];
        let observation_counts = vec![vec![10]];

        let bin_weights = precompute_bin_weights(&observation_counts);
        let precomputed_grids: Vec<PrecomputedGrid> =
            grids.iter().map(precompute_log_components).collect();
        let (median_index, _distance_matrix) =
            select_median_tensor_component_shape(&precomputed_grids, &bin_weights);
        assert_eq!(median_index, 0);
    }

    #[test]
    fn test_trim_outliers() {
        let intervals = vec![vec![(0.0, 1.0)]];
        let splits = vec![vec![]];
        let grid1 = create_simple_grid(
            vec![vec![1.0]],
            vec![vec![0.0]],
            1.0,
            0.5,
            splits.clone(),
            intervals.clone(),
        );
        let grid2 = create_simple_grid(
            vec![vec![1.0]],
            vec![vec![0.0]],
            1.0,
            0.5,
            splits.clone(),
            intervals.clone(),
        );
        let grid3 = create_simple_grid(
            vec![vec![2.0]],
            vec![vec![0.5]],
            2.0,
            1.0,
            splits,
            intervals,
        );
        let grids = vec![grid1, grid2, grid3];
        let observation_counts = vec![vec![10]];

        let bin_weights = precompute_bin_weights(&observation_counts);
        let precomputed_grids: Vec<PrecomputedGrid> =
            grids.iter().map(precompute_log_components).collect();
        let (median_index, distance_matrix) =
            select_median_tensor_component_shape(&precomputed_grids, &bin_weights);
        let kept = trim_outliers(&grids, median_index, &distance_matrix, 0.9);

        assert!(kept.len() >= 2);
        assert!(kept.contains(&0) || kept.contains(&1));
    }

    #[test]
    fn test_combine_lambdas_geometric_mean() {
        let lambda_plus = vec![1.0, 2.0, 4.0];
        let lambda_minus = vec![0.5, 1.0, 2.0];

        let (combined_plus, combined_minus) =
            combine_lambdas_geometric_mean(&lambda_plus, &lambda_minus, None);

        let product: f64 = 1.0 * 2.0 * 4.0;
        let expected_plus = product.powf(1.0 / 3.0);
        assert!((combined_plus - expected_plus).abs() < 1e-10);
        assert!((combined_minus - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_bagged_two_tensor_single_bag() {
        let intervals = vec![vec![(0.0, 1.0)]];
        let splits = vec![vec![]];
        let grid = create_simple_grid(
            vec![vec![1.0]],
            vec![vec![0.0]],
            1.0,
            0.5,
            splits,
            intervals,
        );
        let grids = vec![grid.clone()];
        let points = Array2::from_shape_vec(
            (10, 1),
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        )
        .unwrap();

        let aggregated = aggregate_bagged_two_tensor(&grids, points.view(), None, 0.9);

        assert_eq!(aggregated.backbone_values.len(), grid.backbone_values.len());
    }
}

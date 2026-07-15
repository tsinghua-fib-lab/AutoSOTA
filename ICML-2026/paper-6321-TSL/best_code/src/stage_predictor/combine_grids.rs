use crate::grid_tensor::GridTensor;

pub(crate) fn refine_grids_to_union_two_tensor(grids: &[GridTensor]) -> Vec<GridTensor> {
    if grids.is_empty() {
        return Vec::new();
    }

    let num_axes = grids[0].intervals.len();
    let mut union_splits: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
    let mut union_intervals: Vec<Vec<(f64, f64)>> = Vec::with_capacity(num_axes);

    for axis in 0..num_axes {
        let mut splits: Vec<f64> = grids
            .iter()
            .flat_map(|grid| grid.splits[axis].iter().copied())
            .collect();

        splits.sort_by(|a, b| a.partial_cmp(b).unwrap());
        splits.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        let mut intervals: Vec<(f64, f64)> = Vec::new();
        if splits.is_empty() {
            intervals.push((f64::NEG_INFINITY, f64::INFINITY));
        } else {
            intervals.push((f64::NEG_INFINITY, splits[0]));
            for i in 0..splits.len() - 1 {
                intervals.push((splits[i], splits[i + 1]));
            }
            intervals.push((splits[splits.len() - 1], f64::INFINITY));
        }

        union_splits.push(splits);
        union_intervals.push(intervals);
    }

    let mut refined_grids = Vec::with_capacity(grids.len());
    for grid in grids {
        let mut refined_backbone: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
        let mut refined_tilt: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
        let mut refined_observation_counts: Vec<Vec<usize>> = Vec::with_capacity(num_axes);

        for axis in 0..num_axes {
            let n_union_bins = union_intervals[axis].len();
            let mut backbone_axis = Vec::with_capacity(n_union_bins);
            let mut tilt_axis = Vec::with_capacity(n_union_bins);
            let counts_axis = vec![0; n_union_bins];

            for &(union_a, union_b) in &union_intervals[axis] {
                let mut found = false;
                for (orig_idx, &(orig_a, orig_b)) in grid.intervals[axis].iter().enumerate() {
                    if union_a >= orig_a && union_b <= orig_b {
                        backbone_axis.push(grid.backbone_values[axis][orig_idx]);
                        tilt_axis.push(grid.tilt_values[axis][orig_idx]);
                        found = true;
                        break;
                    }
                }
                if !found {
                    log::warn!(
                        "Could not find parent interval for union interval [{}, {})",
                        union_a,
                        union_b
                    );
                    backbone_axis.push(1.0);
                    tilt_axis.push(0.0);
                }
            }

            refined_backbone.push(backbone_axis);
            refined_tilt.push(tilt_axis);
            refined_observation_counts.push(counts_axis);
        }

        let refined_grid = GridTensor::new_two_tensor(
            union_splits.clone(),
            refined_observation_counts,
            union_intervals.clone(),
            refined_backbone,
            refined_tilt,
            grid.lambda_plus,
            grid.lambda_minus,
        );

        refined_grids.push(refined_grid);
    }

    refined_grids
}

/// Sign-preserving geometric mean for prediction-space aggregation
/// (`Aggregation::GeometricMean`).
pub(super) fn geometric_mean_combiner(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sign = values.iter().map(|v| v.signum()).sum::<f64>().signum();
    let log_sum = values.iter().map(|v| v.abs().ln()).sum::<f64>();
    let geom_mean = (log_sum / values.len() as f64).exp();
    sign * geom_mean
}

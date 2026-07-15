/// L2 identification / normalization (prediction-preserving).
///
/// - Backbone weighted L2 normalization per axis (pushes scale into λ_±)
/// - Tilt centering per axis (pushes offset into λ_± asymmetrically)
/// - Orientation canonicalization (enforce λ_+ ≥ λ_- by swapping + flipping tilts)
pub fn l2_identify(
    backbone_values: &mut [Vec<f64>],
    tilt_values: &mut [Vec<f64>],
    observation_counts: &[Vec<usize>],
    lambda_plus: &mut f64,
    lambda_minus: &mut f64,
) {
    const EPS: f64 = 1e-12;
    debug_assert_eq!(backbone_values.len(), tilt_values.len());
    debug_assert_eq!(backbone_values.len(), observation_counts.len());

    // 8.1 Backbone scaling normalization: b_j^k <- b_j^k / norm_j, λ_± <- λ_± * Π_j norm_j
    for dim in 0..backbone_values.len() {
        let counts = &observation_counts[dim];
        let b = &mut backbone_values[dim];
        if counts.is_empty() || b.is_empty() {
            continue;
        }
        let weights_sum: f64 = counts.iter().sum::<usize>() as f64;
        if weights_sum <= EPS {
            continue;
        }
        let l2_weighted = b
            .iter()
            .zip(counts.iter())
            .map(|(&x, &w)| x * x * w as f64)
            .sum::<f64>();
        if l2_weighted <= EPS {
            continue;
        }
        let norm = (l2_weighted / weights_sum).sqrt().max(EPS);
        for x in b.iter_mut() {
            *x /= norm;
        }
        *lambda_plus *= norm;
        *lambda_minus *= norm;
    }

    // 8.2 Tilt centering: d_j^k <- d_j^k - c_j, λ_+ *= exp(Σ c_j), λ_- *= exp(-Σ c_j)
    let mut c_sum = 0.0;
    for dim in 0..tilt_values.len() {
        let counts = &observation_counts[dim];
        let d = &mut tilt_values[dim];
        if counts.is_empty() || d.is_empty() {
            continue;
        }
        let weights_sum: f64 = counts.iter().sum::<usize>() as f64;
        if weights_sum <= EPS {
            continue;
        }
        let mean = d
            .iter()
            .zip(counts.iter())
            .map(|(&x, &w)| x * w as f64)
            .sum::<f64>()
            / weights_sum;
        for x in d.iter_mut() {
            *x -= mean;
        }
        c_sum += mean;
    }

    *lambda_plus *= c_sum.exp();
    *lambda_minus *= (-c_sum).exp();

    // NOTE: We intentionally do NOT perform the previously-proposed "orientation canonicalization"
    // (swap (λ_+, λ_-) and negate all tilts).
    //
    // For the ordered difference model:
    //   f(x) = λ_+ Π_j b_j(x_j) exp(+d_j(x_j)) - λ_- Π_j b_j(x_j) exp(-d_j(x_j))
    // that transformation maps f -> -f, i.e. it is NOT prediction-preserving.
    //
    // If we later want a canonical orientation for *ensemble alignment*, it must be done in a way
    // that does not change stage predictions, or by changing the model definition (with an ADR).

    // Safety: ensure positivity
    *lambda_plus = lambda_plus.max(EPS);
    *lambda_minus = lambda_minus.max(EPS);
}

#[cfg(test)]
mod tests {
    use super::l2_identify;

    #[test]
    fn test_identification_preserves_predictions() {
        // Create simple test case: 2 axes, 2 intervals each
        let mut backbone_values = vec![
            vec![1.5, 2.0], // axis 0: 2 intervals
            vec![1.2, 1.8], // axis 1: 2 intervals
        ];
        let mut tilt_values = vec![
            vec![0.3, -0.2], // axis 0
            vec![0.1, 0.4],  // axis 1
        ];
        let observation_counts = vec![
            vec![10, 15], // axis 0
            vec![12, 13], // axis 1
        ];
        let mut lambda_plus = 2.5;
        let mut lambda_minus = 1.8;

        // Compute predictions BEFORE identification for all 4 combinations
        let mut predictions_before = Vec::new();
        for i0 in 0..2 {
            for i1 in 0..2 {
                let b0: f64 = backbone_values[0][i0];
                let d0: f64 = tilt_values[0][i0];
                let b1: f64 = backbone_values[1][i1];
                let d1: f64 = tilt_values[1][i1];

                // f_plus = λ_+ * Π_j b_j * exp(d_j)
                let f_plus = lambda_plus * b0 * d0.exp() * b1 * d1.exp();
                // f_minus = λ_- * Π_j b_j * exp(-d_j)
                let f_minus = lambda_minus * b0 * (-d0).exp() * b1 * (-d1).exp();
                let pred = f_plus - f_minus;
                predictions_before.push((i0, i1, pred));
            }
        }

        // Apply identification
        l2_identify(
            &mut backbone_values,
            &mut tilt_values,
            &observation_counts,
            &mut lambda_plus,
            &mut lambda_minus,
        );

        // Compute predictions AFTER identification
        let mut predictions_after = Vec::new();
        for i0 in 0..2 {
            for i1 in 0..2 {
                let b0: f64 = backbone_values[0][i0];
                let d0: f64 = tilt_values[0][i0];
                let b1: f64 = backbone_values[1][i1];
                let d1: f64 = tilt_values[1][i1];

                let f_plus = lambda_plus * b0 * d0.exp() * b1 * d1.exp();
                let f_minus = lambda_minus * b0 * (-d0).exp() * b1 * (-d1).exp();
                let pred = f_plus - f_minus;
                predictions_after.push((i0, i1, pred));
            }
        }

        // Compare predictions
        println!("\nPrediction comparison:");
        for ((i0_b, i1_b, pred_b), (i0_a, i1_a, pred_a)) in
            predictions_before.iter().zip(predictions_after.iter())
        {
            assert_eq!(*i0_b, *i0_a);
            assert_eq!(*i1_b, *i1_a);
            let diff = (pred_b - pred_a).abs();
            println!(
                "  Interval ({}, {}): before={:.10e}, after={:.10e}, diff={:.10e}",
                i0_b, i1_b, pred_b, pred_a, diff
            );
            assert!(
                diff < 1e-10,
                "Prediction mismatch: before={}, after={}, diff={}",
                pred_b,
                pred_a,
                diff
            );
        }
    }

    #[test]
    fn test_identification_preserves_predictions_when_lambda_plus_lt_lambda_minus() {
        // Same test, but with lambdas reversed to ensure we cover that branch historically.
        // Identification must still be prediction-preserving.
        let mut backbone_values = vec![vec![1.5, 2.0], vec![1.2, 1.8]];
        let mut tilt_values = vec![vec![0.3, -0.2], vec![0.1, 0.4]];
        let observation_counts = vec![vec![10, 15], vec![12, 13]];

        // Force λ_+ < λ_- (previous "orientation canonicalization" would have flipped sign).
        let mut lambda_plus = 1.8;
        let mut lambda_minus = 2.5;

        let mut predictions_before = Vec::new();
        for i0 in 0..2 {
            for i1 in 0..2 {
                let b0: f64 = backbone_values[0][i0];
                let d0: f64 = tilt_values[0][i0];
                let b1: f64 = backbone_values[1][i1];
                let d1: f64 = tilt_values[1][i1];
                let f_plus = lambda_plus * b0 * d0.exp() * b1 * d1.exp();
                let f_minus = lambda_minus * b0 * (-d0).exp() * b1 * (-d1).exp();
                predictions_before.push(f_plus - f_minus);
            }
        }

        l2_identify(
            &mut backbone_values,
            &mut tilt_values,
            &observation_counts,
            &mut lambda_plus,
            &mut lambda_minus,
        );

        let mut predictions_after = Vec::new();
        for i0 in 0..2 {
            for i1 in 0..2 {
                let b0: f64 = backbone_values[0][i0];
                let d0: f64 = tilt_values[0][i0];
                let b1: f64 = backbone_values[1][i1];
                let d1: f64 = tilt_values[1][i1];
                let f_plus = lambda_plus * b0 * d0.exp() * b1 * d1.exp();
                let f_minus = lambda_minus * b0 * (-d0).exp() * b1 * (-d1).exp();
                predictions_after.push(f_plus - f_minus);
            }
        }

        for (before, after) in predictions_before.iter().zip(predictions_after.iter()) {
            let diff = (before - after).abs();
            assert!(
                diff < 1e-10,
                "Prediction mismatch: before={}, after={}, diff={}",
                before,
                after,
                diff
            );
        }
    }
}

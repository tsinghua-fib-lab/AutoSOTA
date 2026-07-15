//! Two-Tensor Solver Module
//!
//!
//! Solves for (u_+, u_-) that minimize the per-side penalized least-squares objective:
//!   L_S(u_+, u_-) = sum w_i (r_tilde - u_+ φ_1 - u_- φ_2)^2
//!                   + α((u_+)^2 + (u_-)^2)
//!                   + τ(u_+ - u_-)^2
//!                   + ρ|u_+ - u_-|
//!
//! Where:
//!   - φ_1 = f_+ * 1_S(i)
//!   - φ_2 = -f_- * 1_S(i)
//!   - S_{11} = sum w_i f_+^2
//!   - S_{22} = sum w_i f_-^2
//!   - S_{12} = -sum w_i f_+ f_-
//!   - t_1 = sum w_i r_tilde f_+
//!   - t_2 = -sum w_i r_tilde f_-

use std::f64;

/// Default hyperparameters for two-tensor solver
pub const DEFAULT_ALPHA: f64 = 0.1;
pub const DEFAULT_TAU: f64 = 0.01;
pub const DEFAULT_RHO: f64 = 0.0;
pub const DEFAULT_V_MIN: f64 = 0.05;
pub const DEFAULT_V_MAX: f64 = 20.0;

/// Condition number threshold for near-singular matrix detection
const COND_THRESHOLD: f64 = 1e12;

/// Solve the two-tensor 2×2 system
///
/// # Arguments
/// * `s11` - Sum of w_i * f_plus[i]^2 for side S
/// * `s22` - Sum of w_i * f_minus[i]^2 for side S
/// * `s12` - -Sum of w_i * f_plus[i] * f_minus[i] for side S (note: negative)
/// * `t1` - Sum of w_i * r_tilde[i] * f_plus[i] for side S
/// * `t2` - -Sum of w_i * r_tilde[i] * f_minus[i] for side S (note: negative)
/// * `alpha` - Ridge regularization strength (≥ 0)
/// * `tau` - L2 tilt coupling strength (≥ 0)
/// * `rho` - L1 tilt penalty strength (≥ 0)
/// * `v_min` - Minimum multiplier value (typically 0.05)
/// * `v_max` - Maximum multiplier value (typically 20.0)
///
/// # Returns
/// `(u_plus, u_minus, gain)` where:
/// - `u_plus` = v_plus - 1 (after clamping)
/// - `u_minus` = v_minus - 1 (after clamping)
/// - `gain` = objective decrease (J(0) - J(u)) for candidate scoring
///
/// # Panics
/// Never panics - returns (0.0, 0.0, 0.0) for near-singular or invalid cases
pub fn solve_two_tensor(
    s11: f64,
    s22: f64,
    s12: f64,
    t1: f64,
    t2: f64,
    alpha: f64,
    tau: f64,
    rho: f64,
    v_min: f64,
    v_max: f64,
) -> (f64, f64, f64) {
    // Build the 2×2 system matrix A
    // A = [[S11 + α + τ,  S12 - τ    ],
    //      [S12 - τ,      S22 + α + τ]]
    let a11 = s11 + alpha + tau;
    let a12 = s12 - tau;
    let a21 = a12; // Symmetric
    let a22 = s22 + alpha + tau;

    // Right-hand side vector t = [t1, t2]^T
    let t = [t1, t2];

    // Solve for u = [u_+, u_-]^T
    let (u_plus, u_minus) = if rho == 0.0 {
        // Case 1: ρ = 0 (pure quadratic)
        solve_rho_zero(a11, a12, a21, a22, t[0], t[1])
    } else {
        // Case 2: ρ > 0 (L1 on tilt difference)
        solve_rho_positive(a11, a12, a21, a22, t[0], t[1], rho)
    };

    // Clamp multipliers to [v_min, v_max]
    let v_plus = (1.0 + u_plus).clamp(v_min, v_max);
    let v_minus = (1.0 + u_minus).clamp(v_min, v_max);

    // Recompute u after clamping
    let u_plus_clamped = v_plus - 1.0;
    let u_minus_clamped = v_minus - 1.0;

    // Compute gain: J(0) - J(u) = 2 t^T u - u^T A u - ρ|u_+ - u_-|
    let gain = compute_gain(
        u_plus_clamped,
        u_minus_clamped,
        a11,
        a12,
        a22,
        t[0],
        t[1],
        rho,
    );

    (u_plus_clamped, u_minus_clamped, gain)
}

/// Solve the 2×2 system when ρ = 0 (pure quadratic case)
fn solve_rho_zero(a11: f64, a12: f64, a21: f64, a22: f64, t1: f64, t2: f64) -> (f64, f64) {
    // Compute determinant
    let det = a11 * a22 - a12 * a21;

    // Check for near-singularity
    if det.abs() < 1e-15 || !det.is_finite() {
        // Near-singular: return no-op (u = 0)
        return (0.0, 0.0);
    }

    // Check condition number
    let norm_a = (a11 * a11 + a12 * a12 + a21 * a21 + a22 * a22).sqrt();
    let cond = norm_a / det.abs();
    if cond > COND_THRESHOLD {
        // Near-singular: return no-op (u = 0)
        return (0.0, 0.0);
    }

    // Solve: A u = t
    // u_+ = (a22 * t1 - a12 * t2) / det
    // u_- = (a11 * t2 - a21 * t1) / det
    let u_plus = (a22 * t1 - a12 * t2) / det;
    let u_minus = (a11 * t2 - a21 * t1) / det;

    // Check for NaN/Inf
    if !u_plus.is_finite() || !u_minus.is_finite() {
        return (0.0, 0.0);
    }

    (u_plus, u_minus)
}

/// Solve the 2×2 system when ρ > 0 (L1 penalty case)
///
/// Uses 3-case closed-form subgradient check:
/// - (+) Solve A u = t - (ρ/2) c, accept if c^T u > 0
/// - (−) Solve A u = t + (ρ/2) c, accept if c^T u < 0
/// - (0) Else project q = A^{-1} t onto hyperplane c^T u = 0
fn solve_rho_positive(
    a11: f64,
    a12: f64,
    a21: f64,
    a22: f64,
    t1: f64,
    t2: f64,
    rho: f64,
) -> (f64, f64) {
    // c = [1, -1]^T
    let c = [1.0, -1.0];
    let rho_half = rho / 2.0;

    // Compute determinant
    let det = a11 * a22 - a12 * a21;

    // Check for near-singularity
    if det.abs() < 1e-15 || !det.is_finite() {
        return (0.0, 0.0);
    }

    // Case (+): Solve A u = t - (ρ/2) c
    let t_plus = [t1 - rho_half * c[0], t2 - rho_half * c[1]];
    let u_plus_case = solve_2x2(a11, a12, a21, a22, t_plus[0], t_plus[1], det);
    if u_plus_case.is_some() {
        let (u_p, u_m) = u_plus_case.unwrap();
        let c_dot_u = c[0] * u_p + c[1] * u_m;
        if c_dot_u > 0.0 {
            return (u_p, u_m);
        }
    }

    // Case (−): Solve A u = t + (ρ/2) c
    let t_minus = [t1 + rho_half * c[0], t2 + rho_half * c[1]];
    let u_minus_case = solve_2x2(a11, a12, a21, a22, t_minus[0], t_minus[1], det);
    if u_minus_case.is_some() {
        let (u_p, u_m) = u_minus_case.unwrap();
        let c_dot_u = c[0] * u_p + c[1] * u_m;
        if c_dot_u < 0.0 {
            return (u_p, u_m);
        }
    }

    // Case (0): Project q = A^{-1} t onto hyperplane c^T u = 0
    // u^(0) = q - r * (c^T q) / (c^T r)
    // where q = A^{-1} t, r = A^{-1} c
    let q = solve_2x2(a11, a12, a21, a22, t1, t2, det);
    let r = solve_2x2(a11, a12, a21, a22, c[0], c[1], det);

    if q.is_some() && r.is_some() {
        let (q_p, q_m) = q.unwrap();
        let (r_p, r_m) = r.unwrap();
        let c_dot_q = c[0] * q_p + c[1] * q_m;
        let c_dot_r = c[0] * r_p + c[1] * r_m;

        if c_dot_r.abs() > 1e-15 {
            let u_p = q_p - r_p * (c_dot_q / c_dot_r);
            let u_m = q_m - r_m * (c_dot_q / c_dot_r);
            if u_p.is_finite() && u_m.is_finite() {
                return (u_p, u_m);
            }
        }
    }

    // Fallback: no-op
    (0.0, 0.0)
}

/// Helper to solve 2×2 system A u = t given precomputed determinant
fn solve_2x2(
    a11: f64,
    a12: f64,
    a21: f64,
    a22: f64,
    t1: f64,
    t2: f64,
    det: f64,
) -> Option<(f64, f64)> {
    if det.abs() < 1e-15 || !det.is_finite() {
        return None;
    }

    let u_plus = (a22 * t1 - a12 * t2) / det;
    let u_minus = (a11 * t2 - a21 * t1) / det;

    if u_plus.is_finite() && u_minus.is_finite() {
        Some((u_plus, u_minus))
    } else {
        None
    }
}

/// Compute gain: J(0) - J(u) = 2 t^T u - u^T A u - ρ|u_+ - u_-|
fn compute_gain(
    u_plus: f64,
    u_minus: f64,
    a11: f64,
    a12: f64,
    a22: f64,
    t1: f64,
    t2: f64,
    rho: f64,
) -> f64 {
    // 2 t^T u = 2 * (t1 * u_plus + t2 * u_minus)
    let t_dot_u = 2.0 * (t1 * u_plus + t2 * u_minus);

    // u^T A u = u_plus^2 * a11 + 2 * u_plus * u_minus * a12 + u_minus^2 * a22
    let u_au = u_plus * u_plus * a11 + 2.0 * u_plus * u_minus * a12 + u_minus * u_minus * a22;

    // ρ|u_+ - u_-|
    let rho_term = rho * (u_plus - u_minus).abs();

    // Gain = 2 t^T u - u^T A u - ρ|u_+ - u_-|
    t_dot_u - u_au - rho_term
}

/// Convert (v_+, v_-) multipliers to (v_b, Δd) backbone/tilt updates
///
/// # Arguments
/// * `v_plus` - Multiplier for f_+ (must be > 0)
/// * `v_minus` - Multiplier for f_- (must be > 0)
///
/// # Returns
/// `(v_b, delta_d)` where:
/// - `v_b = sqrt(v_+ * v_-)` (backbone scaling factor)
/// - `delta_d = 0.5 * log(v_+ / v_-)` (tilt increment)
///
/// These satisfy:
/// - `v_b * exp(+delta_d) = v_+`
/// - `v_b * exp(-delta_d) = v_-`
pub fn convert_multipliers_to_backbone_tilt(v_plus: f64, v_minus: f64) -> (f64, f64) {
    debug_assert!(
        v_plus > 0.0 && v_minus > 0.0,
        "Multipliers must be positive"
    );

    let v_b = (v_plus * v_minus).sqrt();
    let delta_d = 0.5 * (v_plus / v_minus).ln();

    (v_b, delta_d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_rho_zero_simple() {
        // Simple case: S11 = 1, S22 = 1, S12 = 0, t1 = 1, t2 = 0
        // A = [[1 + α + τ, -τ    ],
        //      [-τ,         1 + α + τ]]
        // With α = 0.1, τ = 0.01:
        // A = [[1.11, -0.01],
        //      [-0.01, 1.11]]
        // det = 1.11^2 - 0.01^2 = 1.232 - 0.0001 = 1.2319
        // u_+ = (1.11 * 1 - (-0.01) * 0) / 1.2319 = 1.11 / 1.2319 ≈ 0.900
        // u_- = (1.11 * 0 - (-0.01) * 1) / 1.2319 = 0.01 / 1.2319 ≈ 0.008

        let (u_plus, u_minus, gain) = solve_two_tensor(
            1.0,  // s11
            1.0,  // s22
            0.0,  // s12
            1.0,  // t1
            0.0,  // t2
            0.1,  // alpha
            0.01, // tau
            0.0,  // rho
            0.05, // v_min
            20.0, // v_max
        );

        assert!(u_plus > 0.8 && u_plus < 1.0);
        assert!(u_minus.abs() < 0.1);
        assert!(gain > 0.0);
    }

    #[test]
    fn test_solve_near_singular() {
        // Near-singular case: very small determinant
        let (u_plus, u_minus, gain) = solve_two_tensor(
            1e-10, // s11
            1e-10, // s22
            1e-10, // s12 (makes det very small)
            1.0,   // t1
            1.0,   // t2
            0.0,   // alpha
            0.0,   // tau
            0.0,   // rho
            0.05,  // v_min
            20.0,  // v_max
        );

        // Should return no-op (0, 0, 0) for near-singular
        assert_eq!(u_plus, 0.0);
        assert_eq!(u_minus, 0.0);
        assert_eq!(gain, 0.0);
    }

    #[test]
    fn test_clamping() {
        // Case that would produce v_+ > v_max
        let (u_plus, _u_minus, _gain) = solve_two_tensor(
            100.0, // Large s11
            1.0, 0.0, 1000.0, // Large t1
            0.0, 0.0,  // alpha
            0.0,  // tau
            0.0,  // rho
            0.05, // v_min
            20.0, // v_max
        );

        let v_plus = 1.0 + u_plus;
        assert!(v_plus <= 20.0, "v_plus should be clamped to v_max");
        assert!(v_plus >= 0.05, "v_plus should be clamped to v_min");
    }

    #[test]
    fn test_convert_multipliers_to_backbone_tilt() {
        let v_plus = 2.0;
        let v_minus = 0.5;

        let (v_b, delta_d) = convert_multipliers_to_backbone_tilt(v_plus, v_minus);

        // v_b = sqrt(2.0 * 0.5) = sqrt(1.0) = 1.0
        assert!((v_b - 1.0).abs() < 1e-10);

        // delta_d = 0.5 * ln(2.0 / 0.5) = 0.5 * ln(4.0) = 0.5 * 1.386... ≈ 0.693
        let expected_delta_d = 0.5 * (2.0f64 / 0.5f64).ln();
        assert!((delta_d - expected_delta_d).abs() < 1e-10);

        // Verify: v_b * exp(+delta_d) = v_+
        let v_plus_reconstructed = v_b * delta_d.exp();
        assert!((v_plus_reconstructed - v_plus).abs() < 1e-10);

        // Verify: v_b * exp(-delta_d) = v_-
        let v_minus_reconstructed = v_b * (-delta_d).exp();
        assert!((v_minus_reconstructed - v_minus).abs() < 1e-10);
    }

    #[test]
    fn test_rho_positive_case() {
        // Test rho > 0 case
        let (u_plus, u_minus, gain) = solve_two_tensor(
            1.0, 1.0, 0.0, 1.0, 0.0, 0.1,  // alpha
            0.01, // tau
            0.1,  // rho > 0
            0.05, // v_min
            20.0, // v_max
        );

        // Should produce valid solution
        assert!(u_plus.is_finite());
        assert!(u_minus.is_finite());
        assert!(gain.is_finite());
    }
}

"""
Evaluation script for Paper 5288:
"Well-Posed KL-Regularized Control via Wasserstein and Kalman-Wasserstein KL Divergences"

Reproduces: WKL Feedback Gain Frobenius Norm on Double Integrator with Anisotropic Noise.
Setting: Sigma_w = diag(1e-5, 1e-5, 10, 10), gamma=0.9, lambda=1, Q=I.

Enhanced with CLI arguments for systematic parameter exploration (CODE-01 + CODE-02).
"""
import argparse
import json
import numpy as np
from scipy.linalg import solve_discrete_are


def discounted_lqr_gain(A, B, Q, R, gamma):
    """Compute discounted LQR feedback gain via DARE."""
    A_gamma = np.sqrt(gamma) * A
    R_gamma = R / gamma
    P = solve_discrete_are(A_gamma, B, Q, R_gamma)
    F = -gamma * np.linalg.solve(R + gamma * B.T @ P @ B, B.T @ P @ A)
    return P, F


def build_system_matrices(system_dim=2):
    """Build A, B matrices via Kronecker product for n-dim double integrator."""
    A1 = np.array([[1, 1], [0, 1]])
    B1 = np.array([[0], [1]])
    In = np.eye(system_dim)
    A = np.kron(In, A1)  # 2n x 2n
    B = np.kron(In, B1)  # 2n x n
    return A, B


def build_Q(args, n):
    """Build state cost matrix Q based on arguments."""
    if args.q_structure == "identity":
        return np.eye(n)
    elif args.q_structure == "diagonal":
        # diag(q_pos, q_vel, q_pos, q_vel, ...) for each spatial dimension
        diag_entries = []
        for i in range(args.system_dim):
            diag_entries.append(args.q_pos)
            diag_entries.append(args.q_vel)
        return np.diag(diag_entries)
    elif args.q_structure == "scaled_identity":
        return args.q_scale * np.eye(n)
    elif args.q_structure == "block_coupled":
        # Block-diagonal with [[q_pp, q_pv], [q_pv, q_vv]] per spatial dim
        Q_block = np.array([[args.q_pp, args.q_pv],
                            [args.q_pv, args.q_vv]])
        Q = np.kron(np.eye(args.system_dim), Q_block)
        return Q
    else:
        return np.eye(n)


def build_Sigma_w(args, n):
    """Build process noise covariance based on arguments."""
    if args.sigma_w_spec == "default":
        # Original: low noise in first 2 dims, high noise in last 2
        entries = []
        for i in range(args.system_dim):
            if i < args.sigma_w_low_count:
                entries.extend([args.sigma_w_low, args.sigma_w_low])
            else:
                entries.extend([args.sigma_w_high, args.sigma_w_high])
        return np.diag(entries[:n])
    elif args.sigma_w_spec == "uniform":
        return args.sigma_w_uniform * np.eye(n)
    elif args.sigma_w_spec == "list":
        # Comma-separated list
        vals = [float(x) for x in args.sigma_w_list.split(",")]
        if len(vals) != n:
            raise ValueError(f"sigma_w_list length {len(vals)} != n={n}")
        return np.diag(vals)
    else:
        return np.diag([1e-5, 1e-5, 10, 10] + [10] * (n - 4))


def compute_WKL(A, B, Q, gamma):
    """WKL (Wasserstein KL) gain: R_WKL = B^T B (noise-independent)."""
    R_WKL = B.T @ B
    P, F = discounted_lqr_gain(A, B, Q, R_WKL, gamma)
    spectral_radius = max(abs(np.linalg.eigvals(A + B @ F)))
    return P, F, spectral_radius


def compute_KL(A, B, Q, Sigma_w, gamma):
    """KL (free energy) gain: R_KL = B^T Sigma_w^{-1} B."""
    # CODE-02: Use solve instead of inv for numerical stability
    R_KL = B.T @ np.linalg.solve(Sigma_w, B)
    P, F = discounted_lqr_gain(A, B, Q, R_KL, gamma)
    spectral_radius = max(abs(np.linalg.eigvals(A + B @ F)))
    return P, F, spectral_radius


def compute_KWKL(A, B, Q, Sigma_w, gamma, lambda_reg):
    """KWKL (Kalman-Wasserstein KL) gain: R_KWKL = B^T (Sigma_w + lambda*I)^{-1} B."""
    R_KWKL = B.T @ np.linalg.solve(Sigma_w + lambda_reg * np.eye(Sigma_w.shape[0]), B)
    P, F = discounted_lqr_gain(A, B, Q, R_KWKL, gamma)
    spectral_radius = max(abs(np.linalg.eigvals(A + B @ F)))
    return P, F, spectral_radius


def main():
    parser = argparse.ArgumentParser(
        description="Paper 5288: WKL/KL/KWKL Feedback Gain Evaluation"
    )
    # System parameters
    parser.add_argument("--system-dim", type=int, default=2,
                        help="Number of spatial dimensions (default: 2)")
    parser.add_argument("--gamma", type=float, default=0.9,
                        help="Discount factor (default: 0.9)")
    parser.add_argument("--lambda-reg", type=float, default=1.0,
                        help="KWKL regularization lambda (default: 1.0)")

    # Q matrix
    parser.add_argument("--q-structure", type=str, default="identity",
                        choices=["identity", "diagonal", "scaled_identity", "block_coupled"],
                        help="Q matrix structure (default: identity)")
    parser.add_argument("--q-pos", type=float, default=1.0,
                        help="Position penalty for diagonal Q (default: 1.0)")
    parser.add_argument("--q-vel", type=float, default=1.0,
                        help="Velocity penalty for diagonal Q (default: 1.0)")
    parser.add_argument("--q-scale", type=float, default=1.0,
                        help="Scale factor for scaled_identity Q (default: 1.0)")
    parser.add_argument("--q-pp", type=float, default=1.0,
                        help="Position-position penalty for block Q (default: 1.0)")
    parser.add_argument("--q-pv", type=float, default=0.0,
                        help="Position-velocity coupling for block Q (default: 0.0)")
    parser.add_argument("--q-vv", type=float, default=1.0,
                        help="Velocity-velocity penalty for block Q (default: 1.0)")

    # Sigma_w (noise covariance)
    parser.add_argument("--sigma-w-spec", type=str, default="default",
                        choices=["default", "uniform", "list"],
                        help="Sigma_w specification mode (default: default)")
    parser.add_argument("--sigma-w-low", type=float, default=1e-5,
                        help="Low noise entry value (default: 1e-5)")
    parser.add_argument("--sigma-w-high", type=float, default=10.0,
                        help="High noise entry value (default: 10)")
    parser.add_argument("--sigma-w-low-count", type=int, default=1,
                        help="Number of spatial dims with low noise (default: 1)")
    parser.add_argument("--sigma-w-uniform", type=float, default=1.0,
                        help="Uniform Sigma_w value (default: 1.0)")
    parser.add_argument("--sigma-w-list", type=str, default="",
                        help="Comma-separated Sigma_w diagonal entries")

    # Method selection
    parser.add_argument("--methods", type=str, default="WKL,KL",
                        help="Comma-separated methods to evaluate: WKL,KL,KWKL (default: WKL,KL)")

    # Output format
    parser.add_argument("--json-output", action="store_true",
                        help="Output results as JSON instead of human-readable")

    args = parser.parse_args()

    A, B = build_system_matrices(args.system_dim)
    n = A.shape[0]
    Q = build_Q(args, n)
    Sigma_w = build_Sigma_w(args, n)

    results = {}

    methods = [m.strip() for m in args.methods.split(",")]

    if not args.json_output:
        print("=" * 60)
        print("Paper 5288: WKL/KL/KWKL Feedback Gain Evaluation")
        print(f"System: {args.system_dim}D double integrator ({n} states, {args.system_dim} inputs)")
        print(f"gamma={args.gamma}, lambda_reg={args.lambda_reg}")
        print(f"Q structure: {args.q_structure}")
        print(f"Sigma_w spec: {args.sigma_w_spec}")
        print("=" * 60)

    if "WKL" in methods:
        _, F_WKL, sr_WKL = compute_WKL(A, B, Q, args.gamma)
        norm_WKL = float(np.linalg.norm(F_WKL, 'fro'))
        margin_WKL = float(1.0 - sr_WKL)
        results["WKL_Frobenius_Norm"] = norm_WKL
        results["WKL_Spectral_Radius"] = float(sr_WKL)
        results["WKL_Stability_Margin"] = margin_WKL
        if not args.json_output:
            print(f"\n--- WKL (Wasserstein KL) ---")
            print(f"Gain F:\n{F_WKL}")
            print(f"RESULT: WKL_Frobenius_Norm={norm_WKL:.6f}")
            print(f"RESULT: WKL_Spectral_Radius={sr_WKL:.6f}")
            print(f"RESULT: WKL_Stability_Margin={margin_WKL:.6f}")

    if "KL" in methods:
        _, F_KL, sr_KL = compute_KL(A, B, Q, Sigma_w, args.gamma)
        norm_KL = float(np.linalg.norm(F_KL, 'fro'))
        margin_KL = float(1.0 - sr_KL)
        results["KL_Baseline_Frobenius_Norm"] = norm_KL
        results["KL_Spectral_Radius"] = float(sr_KL)
        results["KL_Stability_Margin"] = margin_KL
        if not args.json_output:
            print(f"\n--- KL (Free Energy / FR baseline) ---")
            print(f"Gain F:\n{F_KL}")
            print(f"RESULT: KL_Baseline_Frobenius_Norm={norm_KL:.6f}")
            print(f"RESULT: KL_Spectral_Radius={sr_KL:.6f}")
            print(f"RESULT: KL_Stability_Margin={margin_KL:.6f}")

    if "KWKL" in methods:
        _, F_KWKL, sr_KWKL = compute_KWKL(A, B, Q, Sigma_w, args.gamma, args.lambda_reg)
        norm_KWKL = float(np.linalg.norm(F_KWKL, 'fro'))
        margin_KWKL = float(1.0 - sr_KWKL)
        results["KWKL_Frobenius_Norm"] = norm_KWKL
        results["KWKL_Spectral_Radius"] = float(sr_KWKL)
        results["KWKL_Stability_Margin"] = margin_KWKL
        if not args.json_output:
            print(f"\n--- KWKL (Kalman-Wasserstein KL) ---")
            print(f"Gain F:\n{F_KWKL}")
            print(f"RESULT: KWKL_Frobenius_Norm={norm_KWKL:.6f}")
            print(f"RESULT: KWKL_Spectral_Radius={sr_KWKL:.6f}")
            print(f"RESULT: KWKL_Stability_Margin={margin_KWKL:.6f}")

    if "WKL_Frobenius_Norm" in results and "KL_Baseline_Frobenius_Norm" in results:
        gap = results["WKL_Frobenius_Norm"] - results["KL_Baseline_Frobenius_Norm"]
        results["WKL_KL_Gap"] = gap
        if not args.json_output:
            print(f"\nRESULT: WKL_KL_Gap={gap:.6f}")

    if args.json_output:
        print(json.dumps(results))

    return results


if __name__ == "__main__":
    main()

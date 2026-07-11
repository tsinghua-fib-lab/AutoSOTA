"""
Bounded Function Certifier WITHOUT Mean Constraint (Legacy Version)

NOTE: This is the original implementation without the mean constraint.
For the updated version with mean constraint (more theoretically precise),
see bounded_fn_certifier_with_mean.py.

This version solves 2 constraints (variance + gradient).
The updated version solves 3 constraints (variance + gradient + mean).
"""

import numpy as np
from scipy.optimize import root_scalar, root, minimize
from scipy.stats import norm
from typing import Callable, Optional
from .base import BaseCertifier

class BoundedCertifier(BaseCertifier):
    """
    Implements bounded function certification WITHOUT mean constraint (2 constraints only).
    
    For the updated theory with mean constraint, use BoundedCertifierWithMean instead.
    """
    def __init__(self, *, sigma: float, M: float, eps_y: float, confidence: float = 0.999, 
                 model_fn: Optional[Callable] = None, quadrature_points: int = 40):
        """
        Args:
            sigma: The standard deviation of the Gaussian smoothing.
            M: The absolute bound on the function's output, i.e., |f(x)| <= M.
            eps_y: The threshold for the change in expectation (T in the pseudocode).
            confidence: The confidence level for statistical estimates.
            model_fn: The model function to certify. If None, must be provided to certify_point.
            quadrature_points: Number of points for Gauss-Hermite quadrature. Default 40 is
                sufficient for smooth functions, but may need to be increased (e.g., 60-100)
                when M is tight and clipping introduces non-smooth "kinks" in φ*(t).
        """
        super().__init__(sigma=sigma)
        self.M = M
        self.eps_y = eps_y # This corresponds to T in your pseudocode
        self.confidence = confidence
        self.model_fn = model_fn
        self.quadrature_points = quadrature_points
        self.name = "Bounded Function Certificate"

    # === Part 0: 1-D Gaussian expectation helper (for T ~ N(0, σ²)) ===
    def _gauss_hermite_expectation(self, func) -> float:
        """
        Numerically approximates E[f(T)] for T ~ N(0, σ²) using Gauss–Hermite quadrature.
        
        We use the standard identity for Z ~ N(0,1):
            E[f(Z)] ≈ sum_i w_i f(√2 x_i) / √π
        and substitute T = σ Z.
        
        Note: The number of quadrature points is controlled by self.quadrature_points,
        which can be set in __init__ to handle non-smooth functions (e.g., when clipping
        introduces "kinks" in φ*(t)).
        """
        # Deferred import to avoid hard dependency at module import time
        from numpy.polynomial.hermite import hermgauss

        x, w = hermgauss(self.quadrature_points)  # nodes and weights for standard Hermite
        z = np.sqrt(2.0) * x

        # Map to T ~ N(0, σ²)
        t = self.sigma * z
        f_vals = func(t)

        # Expectation w.r.t. N(0,1), then implicitly scaled to N(0, σ²)
        # Note: func must already account for σ in its definition where needed.
        return float(np.sum(w * f_vals) / np.sqrt(np.pi))

    def u_statistic_variance_estimator_alpha_half(self, samples: np.ndarray) -> tuple:
        """U-statistic variance estimator with α/2 confidence interval for union bound."""
        n = len(samples)
        
        # Edge case: need at least 2 samples for variance
        if n < 2:
            return 0.0, 0.0, 0.0
        
        # U-statistic estimator: S² (unbiased sample variance)
        theta_hat = np.var(samples, ddof=1)  # This is S² directly
        
        # Sample estimates for asymptotic variance
        mean_val = np.mean(samples)
        fourth_moment = np.mean((samples - mean_val)**4)
        
        # Asymptotic variance: m̂₄ - (S²)²
        # Clamp for numerical safety (as suggested by collaborator)
        asymptotic_var = max(0.0, fourth_moment - theta_hat**2)
        
        # Use α/2 for union bound
        alpha_total = 1 - self.confidence
        alpha_split = alpha_total / 2.0
        z_critical = norm.ppf(1 - alpha_split)  # One-sided UCB for certification
        se = np.sqrt(asymptotic_var / n)
        
        lower_bound = theta_hat - z_critical * se
        upper_bound = theta_hat + z_critical * se
        
        return theta_hat, lower_bound, upper_bound
    
    def u_statistic_gradient_norm_estimator_alpha_half(
        self,
        f_values: np.ndarray,
        eta_samples: np.ndarray,
    ) -> tuple:
        """Correct U-statistic gradient norm estimator with α/2 confidence interval for union bound."""
        n = len(f_values)
        
        # Edge case: need at least 2 samples
        if n < 2:
            return 0.0, 0.0, 0.0
        
        # Construct W_i = (1/σ²) * ε_i * f(z + ε_i) from pre-computed samples
        W_samples = (1 / self.sigma**2) * eta_samples * f_values[:, np.newaxis]

        # Efficient U-statistic for ||G||^2 using the formula:
        # θ̂² = (1/C(n,2)) * [||∑W_i||² - ∑||W_i||²] / 2
        # This is equivalent to the pairwise dot product formula but more efficient
        
        sum_W = np.sum(W_samples, axis=0)
        sum_W_sq_norm = np.dot(sum_W, sum_W)
        sum_sq_norm_W = np.sum(np.linalg.norm(W_samples, axis=1) ** 2)
        off_diagonal_sum = 0.5 * (sum_W_sq_norm - sum_sq_norm_W)
        num_pairs = n * (n - 1) / 2
        
        if num_pairs > 0:
            theta_hat_sq = off_diagonal_sum / num_pairs
        else:
            theta_hat_sq = 0.0
        
        # The U-statistic can be negative due to numerical issues or high curvature
        # In such cases, we should return a small positive value rather than 0
        if theta_hat_sq < 0:
            # If negative, use the sample mean approach as fallback
            mu_hat = np.mean(W_samples, axis=0)
            theta_hat_sq = np.dot(mu_hat, mu_hat)
        
        grad_norm_estimate = np.sqrt(theta_hat_sq)
        
        # For confidence interval, use the asymptotic variance formula
        # This is the same as in VarianceGradientCertifier
        mu_hat = np.mean(W_samples, axis=0)
        centered_W = W_samples - mu_hat
        # More efficient covariance calculation (as suggested by collaborator)
        Sigma_hat = centered_W.T @ centered_W / (n - 1)  # same as np.cov(..., ddof=1)
        asymptotic_var = max(0.0, 4.0 * (mu_hat @ Sigma_hat @ mu_hat))  # clamp for numerical safety
        
        # Use α/2 for union bound
        alpha_total = 1 - self.confidence
        alpha_split = alpha_total / 2.0
        z_critical = norm.ppf(1 - alpha_split)  # One-sided UCB for certification
        
        # Standard error for ||G||^2
        se_theta_sq = np.sqrt(asymptotic_var / n)
        
        # Confidence interval for ||G||^2
        theta_lower_sq = max(0, theta_hat_sq - z_critical * se_theta_sq)
        theta_upper_sq = theta_hat_sq + z_critical * se_theta_sq
        
        # Convert to confidence interval for ||G||
        grad_norm_lower = np.sqrt(theta_lower_sq)
        grad_norm_upper = np.sqrt(theta_upper_sq)
        
        return grad_norm_estimate, grad_norm_lower, grad_norm_upper

    # === Part 1: Dual optimization for worst-case φ* for fixed shift α = r ===

    def _solve_dual_multipliers(
        self,
        r: float,
        C: float,
        G_norm: float,
        *,
        max_iter: int = 1000,  # Increased from 200
        tol: float = 1e-5,
        lr_lambda: float = 0.5,  # Increased from 0.2
        lr_mu: float = 0.5,  # Increased from 0.2
        lambda_init: Optional[float] = None,
        mu_init: Optional[float] = None,
        use_scipy_optimizer: bool = True,  # Try scipy optimizer as fallback
        prefer_exact_solver: bool = True,  # Prefer exact solver over iterative
    ) -> tuple:
        """
        Implements Algorithm 3 (Dual Optimization for Worst Case Function) from the paper.

        Given a fixed shift α = r, variance C, gradient norm ||G||, and bound M, this
        routine finds multipliers (λ_b, μ_b) such that the variance and gradient
        constraints are (approximately) satisfied:

            E[φ*(T)^2] = C
            E[φ*(T) T] = σ² ||G||_2

        where T ~ N(0, σ²) and

            w(t) = exp(α t / σ² - α² / (2σ²)) - 1
            h(t) = (w(t) - μ_b t) / (2 λ_b)
            φ*(t) = clip(h(t), -M, M).
        """
        
        # Better initialization: estimate from constraints
        if lambda_init is None:
            # Rough estimate: if no clipping, lambda ~ C / (some scale)
            # Use a heuristic based on C and M
            lambda_init = max(0.1, C / (2 * self.M**2)) if C > 0 else 1.0
        if mu_init is None:
            # Rough estimate: if no clipping, mu ~ G_norm * sigma^2 / (some scale)
            mu_init = 0.0  # Start at zero, let optimization find it
        
        # Predefine w(t) for this radius
        def w_of_t(t: np.ndarray) -> np.ndarray:
            return np.exp((r * t) / (self.sigma**2) - (r**2) / (2 * self.sigma**2)) - 1.0
        
        # If prefer_exact_solver, try exact solver first
        if prefer_exact_solver and use_scipy_optimizer:
            try:
                result = self._solve_dual_exact(r, C, G_norm, w_of_t, lambda_init, mu_init)
                if result is not None:
                    return result
            except Exception:
                # Fall back to iterative if exact solver fails
                pass
        
        lambda_b = max(1e-6, float(lambda_init))
        mu_b = float(mu_init)
        
        # Use adaptive learning rates with momentum-like behavior
        current_lr_lambda = lr_lambda
        current_lr_mu = lr_mu
        prev_grad_lambda = 0.0
        prev_grad_mu = 0.0
        momentum = 0.3  # Momentum coefficient
        
        # Track best solution
        best_lambda = lambda_b
        best_mu = mu_b
        best_residual = float('inf')

        for iter_num in range(max_iter):
            # Construct h(t) and φ*(t)
            def phi_star_sq(t: np.ndarray) -> np.ndarray:
                w_t = w_of_t(t)
                h_t = (w_t - mu_b * t) / (2.0 * lambda_b)
                phi = np.clip(h_t, -self.M, self.M)
                return phi**2

            def phi_star_times_t(t: np.ndarray) -> np.ndarray:
                w_t = w_of_t(t)
                h_t = (w_t - mu_b * t) / (2.0 * lambda_b)
                phi = np.clip(h_t, -self.M, self.M)
                return phi * t

            # 3. Compute constraint integrals (moments) via Gauss–Hermite
            V_val = self._gauss_hermite_expectation(phi_star_sq)
            G_val = self._gauss_hermite_expectation(phi_star_times_t)

            # 4. Gradients of the dual (constraint residuals)
            # C constraint is now an INEQUALITY: E[φ²] ≤ C
            # Constraint is satisfied if C - V_val ≥ 0
            # If satisfied with slack (C - V_val > 0), λ can be 0
            # If tight (C - V_val = 0), λ > 0
            slack_C = C - V_val  # Slack in C constraint (≥ 0 means satisfied)
            grad_lambda = slack_C  # Gradient: positive if slack, negative if violated
            grad_mu = (self.sigma**2) * G_norm - G_val  # G constraint remains equality
            
            # Track best solution
            # For inequality: only count violation (negative slack) as error
            C_error = max(0, -slack_C)  # 0 if satisfied, positive if violated
            residual_norm = np.sqrt(C_error**2 + grad_mu**2)
            if residual_norm < best_residual:
                best_residual = residual_norm
                best_lambda = lambda_b
                best_mu = mu_b

            # Check convergence
            # For inequality: constraint satisfied if slack_C ≥ 0
            # For equality: constraint satisfied if |grad_mu| < tol
            C_satisfied = slack_C >= -tol  # Allow small numerical error
            G_satisfied = abs(grad_mu) < tol
            if C_satisfied and G_satisfied:
                break

            # 5. Update multipliers with momentum and adaptive learning rate
            # NOTE: We do DESCENT (not ascent) because g(λ,μ) is CONVEX and we want to minimize it
            # (The primal is maximization, so dual function is convex and we minimize)
            lambda_b_old = lambda_b
            mu_b_old = mu_b
            
            # Momentum update
            update_lambda = current_lr_lambda * grad_lambda + momentum * prev_grad_lambda
            update_mu = current_lr_mu * grad_mu + momentum * prev_grad_mu
            
            # GRADIENT DESCENT to minimize convex g
            # For inequality: if constraint has slack (grad_lambda > 0), reduce λ
            # If constraint is violated (grad_lambda < 0), increase λ
            lambda_b = max(0.0, lambda_b - update_lambda)  # λ ≥ 0 for inequality
            mu_b = mu_b - update_mu
            
            prev_grad_lambda = update_lambda
            prev_grad_mu = update_mu
            
            # Adaptive learning rate: increase if making progress, decrease if stuck
            if iter_num > 5:
                progress_lambda = abs(lambda_b - lambda_b_old)
                progress_mu = abs(mu_b - mu_b_old)
                
                # If making good progress, increase learning rate slightly
                if progress_lambda > 1e-6 and progress_mu > 1e-6:
                    current_lr_lambda = min(1.0, current_lr_lambda * 1.01)
                    current_lr_mu = min(1.0, current_lr_mu * 1.01)
                # If stuck, decrease learning rate
                elif progress_lambda < 1e-10 or progress_mu < 1e-10:
                    current_lr_lambda *= 0.95
                    current_lr_mu *= 0.95
                    # Reset momentum if stuck
                    prev_grad_lambda = 0.0
                    prev_grad_mu = 0.0
        
        # Compute final residuals for checking
        def phi_star_sq_final(t: np.ndarray) -> np.ndarray:
            w_t = w_of_t(t)
            h_t = (w_t - best_mu * t) / (2.0 * best_lambda)
            phi = np.clip(h_t, -self.M, self.M)
            return phi**2
        
        def phi_star_times_t_final(t: np.ndarray) -> np.ndarray:
            w_t = w_of_t(t)
            h_t = (w_t - best_mu * t) / (2.0 * best_lambda)
            phi = np.clip(h_t, -self.M, self.M)
            return phi * t
        
        V_val_final = self._gauss_hermite_expectation(phi_star_sq_final)
        G_val_final = self._gauss_hermite_expectation(phi_star_times_t_final)
        # For inequality: C constraint satisfied if C - V_val ≥ 0
        C_slack_final = C - V_val_final
        G_error_final = abs((self.sigma**2) * G_norm - G_val_final)
        C_violation = max(0, -C_slack_final)  # Only count violation (negative slack)
        final_residual = np.sqrt(C_violation**2 + G_error_final**2)
        
        # If didn't converge, try more robust optimization methods
        if (C_slack_final < -tol or G_error_final > tol) and use_scipy_optimizer:
            # Try scipy.optimize.minimize with L-BFGS-B (handles bounds well)
            try:
                def objective(x):
                    """
                    Dual function g(λ,μ) for minimization.
                    Since primal is MAXIMIZATION, g is CONVEX and we minimize it directly.
                    
                    Dual function: g(λ,μ) = ∫[φ*(w - μt) - λφ*²] dt + λC + μσ²G
                    where φ* = argmax_φ L(φ, λ, μ)
                    """
                    lambda_b_opt = max(1e-8, x[0])
                    mu_b_opt = x[1]
                    
                    def phi_star_opt(t: np.ndarray) -> np.ndarray:
                        w_t = w_of_t(t)
                        h_t = (w_t - mu_b_opt * t) / (2.0 * lambda_b_opt)
                        return np.clip(h_t, -self.M, self.M)
                    
                    def integrand(t: np.ndarray) -> np.ndarray:
                        phi = phi_star_opt(t)
                        w_t = w_of_t(t)
                        return phi * (w_t - mu_b_opt * t) - lambda_b_opt * phi**2
                    
                    integral = self._gauss_hermite_expectation(integrand)
                    # g(λ,μ) = integral + λC + μσ²G (CONVEX function for max-primal)
                    dual_value_g = integral + lambda_b_opt * C + mu_b_opt * (self.sigma**2) * G_norm
                    return dual_value_g
                
                def compute_constraints_opt(x):
                    """Compute constraint values for verification."""
                    lambda_b_opt = max(1e-8, x[0])
                    mu_b_opt = x[1]
                    
                    def phi_star_sq_opt(t: np.ndarray) -> np.ndarray:
                        w_t = w_of_t(t)
                        h_t = (w_t - mu_b_opt * t) / (2.0 * lambda_b_opt)
                        phi = np.clip(h_t, -self.M, self.M)
                        return phi**2
                    
                    def phi_star_times_t_opt(t: np.ndarray) -> np.ndarray:
                        w_t = w_of_t(t)
                        h_t = (w_t - mu_b_opt * t) / (2.0 * lambda_b_opt)
                        phi = np.clip(h_t, -self.M, self.M)
                        return phi * t
                    
                    V_val_opt = self._gauss_hermite_expectation(phi_star_sq_opt)
                    G_val_opt = self._gauss_hermite_expectation(phi_star_times_t_opt)
                    
                    return V_val_opt, G_val_opt
                
                # Use best solution as initial guess
                x0 = np.array([best_lambda, best_mu])
                
                # Try L-BFGS-B with bounds
                bounds = [(1e-8, None), (None, None)]  # lambda > 0, mu unbounded
                sol = minimize(
                    objective, 
                    x0, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                if sol.success:
                    lambda_b_opt = max(1e-8, sol.x[0])
                    mu_b_opt = sol.x[1]
                    # Verify constraints are satisfied
                    V_val, G_val = compute_constraints_opt(sol.x)
                    C_slack = C - V_val
                    G_error = abs((self.sigma**2) * G_norm - G_val)
                    # C constraint satisfied if slack ≥ 0, G constraint satisfied if error < tol
                    if C_slack >= -1e-5 and G_error < 1e-5:
                        return lambda_b_opt, mu_b_opt
                    
                    # If L-BFGS-B didn't work well, try trust-region
                    sol2 = minimize(
                        objective,
                        x0,
                        method='trust-constr',
                        bounds=bounds,
                        options={'maxiter': 500, 'gtol': 1e-6}
                    )
                    
                    if sol2.success:
                        lambda_b_opt = max(1e-8, sol2.x[0])
                        mu_b_opt = sol2.x[1]
                        V_val, G_val = compute_constraints_opt(sol2.x)
                        C_slack = C - V_val
                        G_error = abs((self.sigma**2) * G_norm - G_val)
                        C_violation = max(0, -C_slack)
                        residual_norm = np.sqrt(C_violation**2 + G_error**2)
                        if residual_norm < best_residual:
                            return lambda_b_opt, mu_b_opt
                        
            except Exception as e:
                # If optimization fails, use best solution from gradient descent
                pass
        
        # Return best solution found
        return best_lambda, best_mu
    
    def _solve_dual_exact(
        self,
        r: float,
        C: float,
        G_norm: float,
        w_of_t: Callable,
        lambda_init: Optional[float] = None,
        mu_init: Optional[float] = None,
    ) -> Optional[tuple]:
        """
        Solve the dual optimization problem exactly using scipy.optimize.
        
        This method uses root finding on the constraint equations:
            E[φ*(T)^2] = C
            E[φ*(T) T] = σ² ||G||
        
        Returns:
            (lambda_b, mu_b) if successful, None otherwise
        """
        # Initialize
        if lambda_init is None:
            lambda_init = max(0.1, C / (2 * self.M**2)) if C > 0 else 1.0
        if mu_init is None:
            mu_init = 0.0
        
        x0 = np.array([lambda_init, mu_init])
        
        def compute_constraints(x):
            """Compute constraint values."""
            lambda_b_opt = max(1e-8, x[0])
            mu_b_opt = x[1]
            
            def phi_star_sq_opt(t: np.ndarray) -> np.ndarray:
                w_t = w_of_t(t)
                h_t = (w_t - mu_b_opt * t) / (2.0 * lambda_b_opt)
                phi = np.clip(h_t, -self.M, self.M)
                return phi**2
            
            def phi_star_times_t_opt(t: np.ndarray) -> np.ndarray:
                w_t = w_of_t(t)
                h_t = (w_t - mu_b_opt * t) / (2.0 * lambda_b_opt)
                phi = np.clip(h_t, -self.M, self.M)
                return phi * t
            
            V_val_opt = self._gauss_hermite_expectation(phi_star_sq_opt)
            G_val_opt = self._gauss_hermite_expectation(phi_star_times_t_opt)
            
            return V_val_opt, G_val_opt
        
        # For inequality constraint E[φ²] ≤ C, use constrained optimization
        # Constraint: V_val ≤ C (i.e., C - V_val ≥ 0)
        try:
            from scipy.optimize import NonlinearConstraint
            
            def objective(x):
                """Minimize constraint violation for equality constraint."""
                V_val, G_val = compute_constraints(x)
                # Only penalize G constraint violation (equality)
                G_error = (self.sigma**2) * G_norm - G_val
                return G_error**2
            
            # Inequality constraint: V_val ≤ C (i.e., C - V_val ≥ 0)
            def C_constraint(x):
                V_val, _ = compute_constraints(x)
                return C - V_val  # Must be ≥ 0
            
            C_ineq = NonlinearConstraint(C_constraint, lb=0.0, ub=np.inf)
            
            # Equality constraint: G_val = σ²||G||
            def G_constraint(x):
                _, G_val = compute_constraints(x)
                return G_val - (self.sigma**2) * G_norm  # Must be = 0
            
            G_eq = NonlinearConstraint(G_constraint, lb=0.0, ub=0.0)
            
            bounds = [(1e-8, None), (None, None)]  # lambda ≥ 0, mu unbounded
            sol = minimize(
                objective,
                x0,
                method='trust-constr',
                bounds=bounds,
                constraints=[C_ineq, G_eq],
                options={'maxiter': 500, 'gtol': 1e-6}
            )
            
            if sol.success:
                lambda_b = max(1e-8, sol.x[0])
                mu_b = sol.x[1]
                V_val, G_val = compute_constraints(sol.x)
                # Verify constraints
                C_slack = C - V_val
                G_error = abs((self.sigma**2) * G_norm - G_val)
                if C_slack >= -1e-5 and G_error < 1e-5:  # C satisfied, G exact
                    return lambda_b, mu_b
        except Exception:
            pass
        
        # Fallback: Try minimization with penalty for inequality violation
        try:
            def objective(x):
                """Minimize constraint violations."""
                V_val, G_val = compute_constraints(x)
                C_slack = C - V_val
                G_error = (self.sigma**2) * G_norm - G_val
                # Penalize C violation (negative slack) and G error
                penalty = max(0, -C_slack)**2 + G_error**2
                return penalty
            
            bounds = [(1e-8, None), (None, None)]  # lambda ≥ 0, mu unbounded
            sol = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if sol.success:
                lambda_b = max(1e-8, sol.x[0])
                mu_b = sol.x[1]
                V_val, G_val = compute_constraints(sol.x)
                C_slack = C - V_val
                G_error = abs((self.sigma**2) * G_norm - G_val)
                # C constraint satisfied if slack ≥ 0, G constraint satisfied if error < tol
                if C_slack >= -1e-5 and G_error < 1e-5:
                    return lambda_b, mu_b
        except Exception:
            pass
        
        return None

    # === Part 2: Worst-case harm Δ(r) for bounded case ===

    def _worst_harm_bounded(self, r: float, C: float, G_norm: float) -> float:
        """
        Implements WorstHarmBounded(r) from Algorithm 2.

        For fixed radius r, solve the dual problem for (λ_b*, μ_b*) and then compute
        the worst-case change in expectation:

            Δ(r) = E_T[φ*(T) w(T)]

        where T ~ N(0, σ²), w(t) is as above, and φ* is the clipped worst-case
        function determined by the optimal multipliers.
        """

        if r <= 0 or (C <= 0 and G_norm <= 0):
            return 0.0

        # Step 1: Solve for multipliers λ_b*, μ_b*
        lambda_b, mu_b = self._solve_dual_multipliers(r, C, G_norm)

        def w_of_t(t: np.ndarray) -> np.ndarray:
            return np.exp((r * t) / (self.sigma**2) - (r**2) / (2 * self.sigma**2)) - 1.0

        # Step 2: Compute worst-case change Δ(r)
        def phi_star_times_w(t: np.ndarray) -> np.ndarray:
            w_t = w_of_t(t)
            h_t = (w_t - mu_b * t) / (2.0 * lambda_b)
            phi = np.clip(h_t, -self.M, self.M)
            return phi * w_t

        delta = self._gauss_hermite_expectation(phi_star_times_w)

        # We care about the magnitude of the worst-case deviation
        return abs(delta)

    # === Part 3: Main certification routines ===

    def certify_point_from_estimates(self, C_ucb: float, G_ucb: float) -> float:
        """
        Full bounded-function certificate using the new dual formulation
        (Algorithms 2 and 3 in the paper).

        This uses pre-computed high-confidence upper bounds on the variance C
        and gradient norm ||G|| and returns the largest radius R such that
        WorstHarmBounded(R) ≤ eps_y.
        """
        C_ucb = float(max(0.0, C_ucb))
        G_ucb = float(max(0.0, G_ucb))

        if C_ucb == 0.0 and G_ucb == 0.0:
            return 0.0

        def worst_harm_minus_eps(r: float) -> float:
            return self._worst_harm_bounded(r, C_ucb, G_ucb) - self.eps_y

        # Bisection search over radius, as in Algorithm 2.
        r_low = 0.0
        # A conservative upper bound; the certificate will usually be much smaller.
        r_high = 5.0 * self.sigma

        # If even this large radius is safe, just return it.
        if worst_harm_minus_eps(r_high) <= 0.0:
            return r_high

        # Ensure the bracket has opposite signs
        f_low = worst_harm_minus_eps(r_low)
        f_high = worst_harm_minus_eps(r_high)

        if f_low > 0.0 and f_high > 0.0:
            # No safe radius found
            return 0.0

        try:
            sol = root_scalar(
                worst_harm_minus_eps,
                bracket=[r_low, r_high],
                method="brentq",
                xtol=1e-4,
                rtol=1e-4,
            )
            return max(0.0, float(sol.root))
        except (ValueError, RuntimeError):
            return 0.0
    
    def certify_point(
        self,
        z: np.ndarray,
        model_fn: Optional[Callable] = None,
        N_samples_stats: int = 10000,
        N_samples_mc: int = 5000,
        seed: Optional[int] = None,
    ) -> float:
        """
        Performs the full certification for a given point z.
        This is the implementation of PROCEDURE FindCertifiedRadius.
        
        Args:
            z: Input point to certify
            model_fn: Model function to certify. If None, uses self.model_fn
            N_samples_stats: Number of samples for statistical estimation
            N_samples_mc: Number of samples for Monte Carlo integration
            seed: Random seed
            
        Returns:
            Certified radius
        """
        if model_fn is None:
            if self.model_fn is None:
                raise ValueError("model_fn must be provided either as parameter or in constructor")
            model_fn = self.model_fn
            
        rng = np.random.default_rng(seed)

        # 1. Estimate statistical quantities (C, G) with high-confidence bounds
        eta_samples = rng.normal(0.0, self.sigma, size=(N_samples_stats, z.shape[-1]))
        f_values = np.array([model_fn(z + eta) for eta in eta_samples])

        # Use U-statistic estimators with α/2 confidence intervals (for union bound)
        _, _, C_ucb = self.u_statistic_variance_estimator_alpha_half(f_values)
        _, _, G_ucb = self.u_statistic_gradient_norm_estimator_alpha_half(
            f_values, eta_samples
        )

        # 2. Delegate to the bounded dual-based certificate
        return self.certify_point_from_estimates(C_ucb, G_ucb)
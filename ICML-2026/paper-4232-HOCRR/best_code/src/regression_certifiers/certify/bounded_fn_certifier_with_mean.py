"""
Bounded Function Certifier WITH Mean Constraint

This implements the updated bounded certification procedure that includes
a mean constraint in the optimization problem. This is theoretically more
precise than the version without the mean constraint.

Key differences from bounded_fn_certifier.py:
- Adds mean constraint: E[φ*(T)] = 0 (centered function, not E[φ] = E)
- Uses 3 Lagrange multipliers: λ (variance), μ (gradient), ν (mean)
- Uses centered bounds: B_up = M - E, B_lo = -M - E
- Solves 3 constraint equations instead of 2

Based on Algorithm 5 and 6 from the updated paper draft.
"""

import numpy as np
from scipy.optimize import root_scalar, root, minimize
from scipy.stats import norm
from typing import Callable, Optional
from .base import BaseCertifier


class BoundedCertifierWithMean(BaseCertifier):
    """
    Bounded function certifier with mean constraint (updated formulation).
    
    This certifier solves a more constrained optimization problem that includes
    an explicit mean constraint, providing tighter certificates for bounded functions.
    """
    
    def __init__(self, *, sigma: float, M: float, eps_y: float, confidence: float = 0.999,
                 model_fn: Optional[Callable] = None, quadrature_points: int = 60,
                 mean_target: float = 0.0,
                 r_high_init_mult: float = 5.0,
                 r_cap_mult: float = 20.0):
        """
        Args:
            sigma: The standard deviation of the Gaussian smoothing.
            M: The absolute bound on the function's output, i.e., |f(x)| <= M.
            eps_y: The threshold for the change in expectation.
            confidence: The confidence level for statistical estimates.
            model_fn: The model function to certify. If None, must be provided to certify_point.
            quadrature_points: Number of points for Gauss-Hermite quadrature (default 60).
            mean_target: Target mean value E (typically 0 for centered functions).
        """
        super().__init__(sigma=sigma)
        self.M = M
        self.eps_y = eps_y
        self.confidence = confidence
        self.model_fn = model_fn
        self.quadrature_points = quadrature_points
        self.mean_target = mean_target  # E in the formulation
        # Bracketing controls for the root solve over radius.
        # Historically we used r_high = 5*sigma to avoid numerical issues; that can hard-cap results.
        # We now expand r_high adaptively up to r_cap_mult*sigma.
        self.r_high_init_mult = float(r_high_init_mult)
        self.r_cap_mult = float(r_cap_mult)
        self.name = "Bounded Function Certificate (With Mean Constraint)"
    
    # === Part 0: 1-D Gaussian expectation helper ===
    
    def _gauss_hermite_expectation(self, func) -> float:
        """
        Numerically approximates E[f(T)] for T ~ N(0, σ²) using Gauss–Hermite quadrature.
        """
        from numpy.polynomial.hermite import hermgauss
        
        x, w = hermgauss(self.quadrature_points)
        z = np.sqrt(2.0) * x
        t = self.sigma * z
        f_vals = func(t)
        
        return float(np.sum(w * f_vals) / np.sqrt(np.pi))
    
    # === Part 1: U-statistic estimators (same as before) ===
    
    def u_statistic_variance_estimator_alpha_half(self, samples: np.ndarray) -> tuple:
        """U-statistic variance estimator with α/2 confidence interval for union bound."""
        n = len(samples)
        
        if n < 2:
            return 0.0, 0.0, 0.0
        
        theta_hat = np.var(samples, ddof=1)
        mean_val = np.mean(samples)
        fourth_moment = np.mean((samples - mean_val)**4)
        asymptotic_var = max(0.0, fourth_moment - theta_hat**2)
        
        alpha_total = 1 - self.confidence
        alpha_split = alpha_total / 3.0  # Now split 3 ways (variance, gradient, mean)
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
        """U-statistic gradient norm estimator with α/2 confidence interval for union bound."""
        n = len(f_values)
        
        if n < 2:
            return 0.0, 0.0, 0.0
        
        W_samples = (1 / self.sigma**2) * eta_samples * f_values[:, np.newaxis]
        
        sum_W = np.sum(W_samples, axis=0)
        sum_W_sq_norm = np.dot(sum_W, sum_W)
        sum_sq_norm_W = np.sum(np.linalg.norm(W_samples, axis=1) ** 2)
        off_diagonal_sum = 0.5 * (sum_W_sq_norm - sum_sq_norm_W)
        num_pairs = n * (n - 1) / 2
        
        if num_pairs > 0:
            theta_hat_sq = off_diagonal_sum / num_pairs
        else:
            theta_hat_sq = 0.0
        
        if theta_hat_sq < 0:
            mu_hat = np.mean(W_samples, axis=0)
            theta_hat_sq = np.dot(mu_hat, mu_hat)
        
        grad_norm_estimate = np.sqrt(theta_hat_sq)
        
        mu_hat = np.mean(W_samples, axis=0)
        centered_W = W_samples - mu_hat
        Sigma_hat = centered_W.T @ centered_W / (n - 1)
        asymptotic_var = max(0.0, 4.0 * (mu_hat @ Sigma_hat @ mu_hat))
        
        alpha_total = 1 - self.confidence
        alpha_split = alpha_total / 3.0  # Now split 3 ways
        z_critical = norm.ppf(1 - alpha_split)  # One-sided UCB for certification
        
        se_theta_sq = np.sqrt(asymptotic_var / n)
        theta_lower_sq = max(0, theta_hat_sq - z_critical * se_theta_sq)
        theta_upper_sq = theta_hat_sq + z_critical * se_theta_sq
        
        grad_norm_lower = np.sqrt(theta_lower_sq)
        grad_norm_upper = np.sqrt(theta_upper_sq)
        
        return grad_norm_estimate, grad_norm_lower, grad_norm_upper
    
    def u_statistic_mean_estimator_alpha_third(self, samples: np.ndarray) -> tuple:
        """
        U-statistic mean estimator with α/3 confidence interval (for 3-way union bound).
        
        Estimates E[f(z + η)] where η ~ N(0, σ²I).
        """
        n = len(samples)
        
        if n < 2:
            return 0.0, 0.0, 0.0
        
        # Sample mean is unbiased estimator
        mean_hat = np.mean(samples)
        
        # Standard error
        se = np.std(samples, ddof=1) / np.sqrt(n)
        
        # Confidence interval with α/3 for union bound
        alpha_total = 1 - self.confidence
        alpha_split = alpha_total / 3.0
        z_critical = norm.ppf(1 - alpha_split)  # One-sided UCB for certification
        
        lower_bound = mean_hat - z_critical * se
        upper_bound = mean_hat + z_critical * se
        
        return mean_hat, lower_bound, upper_bound
    
    # === Part 2: Dual optimization with 3 constraints ===
    
    def _solve_dual_multipliers_with_mean(
        self,
        r: float,
        C: float,
        G_norm: float,
        E: float,
        *,
        max_iter: int = 1000,  # Same as original
        tol: float = 1e-5,  # Same as original
        lambda_init: Optional[float] = None,
        mu_init: Optional[float] = None,
        nu_init: Optional[float] = None,
        use_scipy_optimizer: bool = True,  # Same as original - try scipy as fallback
        prefer_exact_solver: bool = True,  # Same as original - prefer exact solver (root finding)
    ) -> tuple:
        """
        Implements Algorithm 6 (Dual Optimization for Worst Case Function with Mean Constraint).
        
        Given fixed shift α = r, variance C, gradient norm ||G||, mean E, and bound M,
        finds multipliers (λ, μ, ν) satisfying three constraints:
        
            E[φ*(T)²] = C           (Variance, Eq. 52)
            E[φ*(T)·T] = σ²||G||    (Gradient, Eq. 53)
            E[φ*(T)] = 0            (Mean - centered function, Eq. 54)
        
        Note: The constraint is E[φ] = 0 (centered), not E[φ] = E.
        The parameter E is used only to shift the bounds: B_up = M - E, B_lo = -M - E.
        
        where T ~ N(0, σ²) and
        
            w(t) = exp(rt/σ² - r²/(2σ²)) - 1
            h(t) = (w(t) - μt - ν) / (2λ)
            φ*(t) = clip(h(t), B_lo, B_up)
        
        with centered bounds B_up = M - E, B_lo = -M - E.
        
        Returns:
            (lambda, mu, nu): Lagrange multipliers
        """
        # Initialize multipliers
        if lambda_init is None:
            lambda_init = max(0.1, C / (2 * self.M**2)) if C > 0 else 1.0
        if mu_init is None:
            mu_init = 0.0
        if nu_init is None:
            nu_init = 0.0
        
        # Define centered bounds
        B_up = self.M - E
        B_lo = -self.M - E
        
        # Define w(t)
        def w_of_t(t: np.ndarray) -> np.ndarray:
            return np.exp((r * t) / (self.sigma**2) - (r**2) / (2 * self.sigma**2)) - 1.0
        
        # If prefer_exact_solver, try exact solver first (same as original)
        if prefer_exact_solver and use_scipy_optimizer:
            try:
                result = self._solve_dual_exact_with_mean(r, C, G_norm, E, w_of_t, 
                                                          lambda_init, mu_init, nu_init,
                                                          B_lo, B_up)
                if result is not None:
                    return result
            except Exception:
                # Fall back to iterative if exact solver fails
                pass
        
        # Use same robust gradient descent as original bounded certifier (with momentum)
        lambda_b = max(1e-6, float(lambda_init))
        mu_b = float(mu_init)
        nu_b = float(nu_init)
        
        # Use adaptive learning rates with momentum-like behavior (same as original)
        current_lr_lambda = 0.5
        current_lr_mu = 0.5
        current_lr_nu = 0.5
        prev_grad_lambda = 0.0
        prev_grad_mu = 0.0
        prev_grad_nu = 0.0
        momentum = 0.3  # Momentum coefficient (same as original)
        
        # Track best solution
        best_lambda = lambda_b
        best_mu = mu_b
        best_nu = nu_b
        best_residual = float('inf')

        for iter_num in range(max_iter):
            # Define φ*(t) with current multipliers
            def phi_star(t: np.ndarray) -> np.ndarray:
                w_t = w_of_t(t)
                h_t = (w_t - mu_b * t - nu_b) / (2.0 * lambda_b)
                return np.clip(h_t, B_lo, B_up)
            
            def phi_star_sq(t: np.ndarray) -> np.ndarray:
                phi = phi_star(t)
                return phi**2
            
            def phi_star_times_t(t: np.ndarray) -> np.ndarray:
                phi = phi_star(t)
                return phi * t
            
            # Compute constraint integrals (moments) via Gauss–Hermite
            V_val = self._gauss_hermite_expectation(phi_star_sq)
            G_val = self._gauss_hermite_expectation(phi_star_times_t)
            E_val = self._gauss_hermite_expectation(phi_star)
            
            # Gradients of the dual (constraint residuals)
            # C constraint is now an INEQUALITY: E[φ²] ≤ C
            # Constraint is satisfied if C - V_val ≥ 0
            # If satisfied with slack (C - V_val > 0), λ can be 0
            # If tight (C - V_val = 0), λ > 0
            slack_C = C - V_val  # Slack in C constraint (≥ 0 means satisfied)
            grad_lambda = slack_C  # Gradient: positive if slack, negative if violated
            grad_mu = (self.sigma**2) * G_norm - G_val  # G constraint remains equality
            grad_nu = 0.0 - E_val  # Mean constraint: E[φ] = 0 for centered function (equality)
            
            # Track best solution
            # For inequality: only count violation (negative slack) as error
            C_error = max(0, -slack_C)  # 0 if satisfied, positive if violated
            residual_norm = np.sqrt(C_error**2 + grad_mu**2 + grad_nu**2)
            if residual_norm < best_residual:
                best_residual = residual_norm
                best_lambda = lambda_b
                best_mu = mu_b
                best_nu = nu_b

            # Check convergence
            # For inequality: constraint satisfied if slack_C ≥ 0
            # For equality: constraint satisfied if |grad| < tol
            C_satisfied = slack_C >= -tol  # Allow small numerical error
            G_satisfied = abs(grad_mu) < tol
            E_satisfied = abs(grad_nu) < tol
            if C_satisfied and G_satisfied and E_satisfied:
                break

            # Update multipliers with momentum and adaptive learning rate
            # NOTE: We do DESCENT (not ascent) because g(λ,μ,ν) is CONVEX and we want to minimize it
            # (The primal is maximization, so dual function is convex and we minimize)
            lambda_b_old = lambda_b
            mu_b_old = mu_b
            nu_b_old = nu_b
            
            # Momentum update (same pattern as original)
            update_lambda = current_lr_lambda * grad_lambda + momentum * prev_grad_lambda
            update_mu = current_lr_mu * grad_mu + momentum * prev_grad_mu
            update_nu = current_lr_nu * grad_nu + momentum * prev_grad_nu
            
            # GRADIENT DESCENT to minimize convex g
            # For inequality: if constraint has slack (grad_lambda > 0), reduce λ
            # If constraint is violated (grad_lambda < 0), increase λ
            lambda_b = max(0.0, lambda_b - update_lambda)  # λ ≥ 0 for inequality
            mu_b = mu_b - update_mu
            nu_b = nu_b - update_nu
            
            prev_grad_lambda = update_lambda
            prev_grad_mu = update_mu
            prev_grad_nu = update_nu
            
            # Adaptive learning rate: increase if making progress, decrease if stuck (same as original)
            if iter_num > 5:
                progress_lambda = abs(lambda_b - lambda_b_old)
                progress_mu = abs(mu_b - mu_b_old)
                progress_nu = abs(nu_b - nu_b_old)
                
                # If making good progress, increase learning rate slightly
                if progress_lambda > 1e-6 and progress_mu > 1e-6 and progress_nu > 1e-6:
                    current_lr_lambda = min(1.0, current_lr_lambda * 1.01)
                    current_lr_mu = min(1.0, current_lr_mu * 1.01)
                    current_lr_nu = min(1.0, current_lr_nu * 1.01)
                # If stuck, decrease learning rate
                elif progress_lambda < 1e-10 or progress_mu < 1e-10 or progress_nu < 1e-10:
                    current_lr_lambda *= 0.95
                    current_lr_mu *= 0.95
                    current_lr_nu *= 0.95
                    # Reset momentum if stuck
                    prev_grad_lambda = 0.0
                    prev_grad_mu = 0.0
                    prev_grad_nu = 0.0
        
        # Compute final residuals for checking
        def phi_star_final(t: np.ndarray) -> np.ndarray:
            w_t = w_of_t(t)
            h_t = (w_t - best_mu * t - best_nu) / (2.0 * best_lambda)
            return np.clip(h_t, B_lo, B_up)
        
        def phi_star_sq_final(t: np.ndarray) -> np.ndarray:
            phi = phi_star_final(t)
            return phi**2
        
        def phi_star_times_t_final(t: np.ndarray) -> np.ndarray:
            phi = phi_star_final(t)
            return phi * t
        
        V_val_final = self._gauss_hermite_expectation(phi_star_sq_final)
        G_val_final = self._gauss_hermite_expectation(phi_star_times_t_final)
        E_val_final = self._gauss_hermite_expectation(phi_star_final)
        # For inequality: C constraint satisfied if C - V_val ≥ 0
        C_slack_final = C - V_val_final
        G_error_final = abs((self.sigma**2) * G_norm - G_val_final)
        E_error_final = abs(0.0 - E_val_final)
        C_violation = max(0, -C_slack_final)  # Only count violation (negative slack)
        final_residual = np.sqrt(C_violation**2 + G_error_final**2 + E_error_final**2)
        
        # If didn't converge, try more robust optimization methods (same fallback as original)
        if (C_slack_final < -tol or G_error_final > tol or E_error_final > tol) and use_scipy_optimizer:
            # Try scipy.optimize.minimize with L-BFGS-B (handles bounds well)
            try:
                def objective(x):
                    """
                    Dual function g(λ,μ,ν) for minimization.
                    Since primal is MAXIMIZATION, g is CONVEX and we minimize it directly.
                    
                    Dual function: g(λ,μ,ν) = ∫[φ*(w - μt - ν) - λφ*²] dt + λC + μσ²G + ν·0
                    where φ* = argmax_φ L(φ, λ, μ, ν)
                    Note: Since constraint is E[φ] = 0 (centered), the ν term is ν·0 = 0 (no νE term).
                    """
                    lambda_b_opt = max(1e-8, x[0])
                    mu_b_opt = x[1]
                    nu_b_opt = x[2]
                    
                    def phi_star_opt(t: np.ndarray) -> np.ndarray:
                        w_t = w_of_t(t)
                        h_t = (w_t - mu_b_opt * t - nu_b_opt) / (2.0 * lambda_b_opt)
                        return np.clip(h_t, B_lo, B_up)
                    
                    def integrand(t: np.ndarray) -> np.ndarray:
                        phi = phi_star_opt(t)
                        w_t = w_of_t(t)
                        return phi * (w_t - mu_b_opt * t - nu_b_opt) - lambda_b_opt * phi**2
                    
                    integral = self._gauss_hermite_expectation(integrand)
                    # g(λ,μ,ν) = integral + λC + μσ²G + ν·0 (CONVEX function for max-primal)
                    # Note: Since constraint is E[φ] = 0 (centered), the ν term is ν·0 = 0
                    dual_value_g = (integral + lambda_b_opt * C + 
                                   mu_b_opt * (self.sigma**2) * G_norm)
                    # No + nu_b_opt * E term because constraint is E[φ] = 0, not E[φ] = E
                    return dual_value_g
                
                def compute_constraints_opt(x):
                    """Compute constraint values for verification."""
                    lambda_b_opt = max(1e-8, x[0])
                    mu_b_opt = x[1]
                    nu_b_opt = x[2]
                    
                    def phi_star_opt(t: np.ndarray) -> np.ndarray:
                        w_t = w_of_t(t)
                        h_t = (w_t - mu_b_opt * t - nu_b_opt) / (2.0 * lambda_b_opt)
                        return np.clip(h_t, B_lo, B_up)
                    
                    def phi_star_sq_opt(t: np.ndarray) -> np.ndarray:
                        phi = phi_star_opt(t)
                        return phi**2
                    
                    def phi_star_times_t_opt(t: np.ndarray) -> np.ndarray:
                        phi = phi_star_opt(t)
                        return phi * t
                    
                    V_val_opt = self._gauss_hermite_expectation(phi_star_sq_opt)
                    G_val_opt = self._gauss_hermite_expectation(phi_star_times_t_opt)
                    E_val_opt = self._gauss_hermite_expectation(phi_star_opt)
                    
                    return V_val_opt, G_val_opt, E_val_opt
                
                # Use best solution as initial guess
                x0 = np.array([best_lambda, best_mu, best_nu])
                
                # Try L-BFGS-B with bounds
                bounds = [(1e-8, None), (None, None), (None, None)]  # lambda > 0, mu and nu unbounded
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
                    nu_b_opt = sol.x[2]
                    # Verify constraints are satisfied
                    V_val, G_val, E_val = compute_constraints_opt(sol.x)
                    C_slack = C - V_val
                    G_error = abs((self.sigma**2) * G_norm - G_val)
                    E_error = abs(0.0 - E_val)
                    C_violation = max(0, -C_slack)
                    residual_norm = np.sqrt(C_violation**2 + G_error**2 + E_error**2)
                    if residual_norm < best_residual:
                        return lambda_b_opt, mu_b_opt, nu_b_opt
                
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
                    nu_b_opt = sol2.x[2]
                    V_val, G_val, E_val = compute_constraints_opt(sol2.x)
                    C_slack = C - V_val
                    G_error = abs((self.sigma**2) * G_norm - G_val)
                    E_error = abs(0.0 - E_val)
                    C_violation = max(0, -C_slack)
                    residual_norm = np.sqrt(C_violation**2 + G_error**2 + E_error**2)
                    if residual_norm < best_residual:
                        return lambda_b_opt, mu_b_opt, nu_b_opt
                        
            except Exception as e:
                # If optimization fails, use best solution from gradient descent
                pass
        
        # Return best solution found
        return best_lambda, best_mu, best_nu
    
    def _solve_dual_exact_with_mean(
        self,
        r: float,
        C: float,
        G_norm: float,
        E: float,
        w_of_t: Callable,
        lambda_init: float,
        mu_init: float,
        nu_init: float,
        B_lo: float,
        B_up: float,
    ) -> Optional[tuple]:
        """
        Solve the 3-constraint dual problem exactly using scipy.optimize.
        
        This method uses root finding on the constraint equations FOR THE CENTERED FUNCTION:
            E[φ*(T)^2] = C           (Variance)
            E[φ*(T) T] = σ² ||G||    (Gradient)
            E[φ*(T)] = 0             (Mean - centered function has zero mean!)
        
        Returns:
            (lambda, mu, nu) if successful, None otherwise
        """
        x0 = np.array([lambda_init, mu_init, nu_init])
        
        def compute_constraints(x):
            """Compute constraint values."""
            lambda_b_opt = max(1e-8, x[0])
            mu_b_opt = x[1]
            nu_b_opt = x[2]
            
            def phi_star_sq_opt(t: np.ndarray) -> np.ndarray:
                w_t = w_of_t(t)
                h_t = (w_t - mu_b_opt * t - nu_b_opt) / (2.0 * lambda_b_opt)
                phi = np.clip(h_t, B_lo, B_up)
                return phi**2
            
            def phi_star_times_t_opt(t: np.ndarray) -> np.ndarray:
                w_t = w_of_t(t)
                h_t = (w_t - mu_b_opt * t - nu_b_opt) / (2.0 * lambda_b_opt)
                phi = np.clip(h_t, B_lo, B_up)
                return phi * t
            
            def phi_star_opt(t: np.ndarray) -> np.ndarray:
                w_t = w_of_t(t)
                h_t = (w_t - mu_b_opt * t - nu_b_opt) / (2.0 * lambda_b_opt)
                return np.clip(h_t, B_lo, B_up)
            
            V_val_opt = self._gauss_hermite_expectation(phi_star_sq_opt)
            G_val_opt = self._gauss_hermite_expectation(phi_star_times_t_opt)
            E_val_opt = self._gauss_hermite_expectation(phi_star_opt)
            
            return V_val_opt, G_val_opt, E_val_opt
        
        # For inequality constraint E[φ²] ≤ C, use constrained optimization
        # Constraint: V_val ≤ C (i.e., C - V_val ≥ 0)
        try:
            from scipy.optimize import NonlinearConstraint
            
            def objective(x):
                """Minimize constraint violation for equality constraints."""
                _, G_val, E_val = compute_constraints(x)
                # Only penalize G and E constraint violations (equalities)
                G_error = (self.sigma**2) * G_norm - G_val
                E_error = 0.0 - E_val
                return G_error**2 + E_error**2
            
            # Inequality constraint: V_val ≤ C (i.e., C - V_val ≥ 0)
            def C_constraint(x):
                V_val, _, _ = compute_constraints(x)
                return C - V_val  # Must be ≥ 0
            
            C_ineq = NonlinearConstraint(C_constraint, lb=0.0, ub=np.inf)
            
            # Equality constraints: G_val = σ²||G||, E_val = 0
            def G_constraint(x):
                _, G_val, _ = compute_constraints(x)
                return G_val - (self.sigma**2) * G_norm  # Must be = 0
            
            def E_constraint(x):
                _, _, E_val = compute_constraints(x)
                return E_val - 0.0  # Must be = 0
            
            G_eq = NonlinearConstraint(G_constraint, lb=0.0, ub=0.0)
            E_eq = NonlinearConstraint(E_constraint, lb=0.0, ub=0.0)
            
            bounds = [(1e-8, None), (None, None), (None, None)]  # lambda ≥ 0, mu and nu unbounded
            sol = minimize(
                objective,
                x0,
                method='trust-constr',
                bounds=bounds,
                constraints=[C_ineq, G_eq, E_eq],
                options={'maxiter': 500, 'gtol': 1e-6}
            )
            
            if sol.success:
                lambda_b = max(1e-8, sol.x[0])
                mu_b = sol.x[1]
                nu_b = sol.x[2]
                V_val, G_val, E_val = compute_constraints(sol.x)
                # Verify constraints
                C_slack = C - V_val
                G_error = abs((self.sigma**2) * G_norm - G_val)
                E_error = abs(0.0 - E_val)
                if C_slack >= -1e-5 and G_error < 1e-5 and E_error < 1e-5:
                    return lambda_b, mu_b, nu_b
        except Exception:
            pass
        
        # Fallback: Try minimization with penalty for inequality violation
        try:
            def objective(x):
                """Minimize constraint violations."""
                V_val, G_val, E_val = compute_constraints(x)
                C_slack = C - V_val
                G_error = (self.sigma**2) * G_norm - G_val
                E_error = 0.0 - E_val
                # Penalize C violation (negative slack) and G, E errors
                penalty = max(0, -C_slack)**2 + G_error**2 + E_error**2
                return penalty
            
            bounds = [(1e-8, None), (None, None), (None, None)]  # lambda > 0, mu and nu unbounded
            sol = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}  # Same as original
            )
            
            if sol.success:
                lambda_b = max(1e-8, sol.x[0])
                mu_b = sol.x[1]
                nu_b = sol.x[2]
                V_val, G_val, E_val = compute_constraints(sol.x)
                C_slack = C - V_val
                G_error = abs((self.sigma**2) * G_norm - G_val)
                E_error = abs(0.0 - E_val)
                # C constraint satisfied if slack ≥ 0, G and E constraints satisfied if error < tol
                if C_slack >= -1e-5 and G_error < 1e-5 and E_error < 1e-5:
                    return lambda_b, mu_b, nu_b
        except Exception:
            pass
        
        return None
    
    # === Part 3: Worst-case harm computation ===
    
    def _worst_harm_bounded_with_mean(self, r: float, C: float, G_norm: float, E: float) -> float:
        """
        Compute worst-case change in expectation Δ(r) for the bounded case with mean constraint.
        
        Implements WorstHarmBounded(r) from Algorithm 5.
        
        Args:
            r: Radius
            C: Variance upper bound
            G_norm: Gradient norm upper bound
            E: Mean estimate
            
        Returns:
            Worst-case harm |Δ(r)|
        """
        if r <= 0 or (C <= 0 and G_norm <= 0):
            return 0.0
        
        # Solve for multipliers
        lambda_b, mu_b, nu_b = self._solve_dual_multipliers_with_mean(r, C, G_norm, E)
        
        # Define centered bounds
        B_up = self.M - E
        B_lo = -self.M - E
        
        # Define w(t)
        def w_of_t(t: np.ndarray) -> np.ndarray:
            return np.exp((r * t) / (self.sigma**2) - (r**2) / (2 * self.sigma**2)) - 1.0
        
        # Compute Δ(r) = E[φ*(T) · w(T)]
        def phi_star_times_w(t: np.ndarray) -> np.ndarray:
            w_t = w_of_t(t)
            h_t = (w_t - mu_b * t - nu_b) / (2.0 * lambda_b)
            phi = np.clip(h_t, B_lo, B_up)
            return phi * w_t
        
        delta = self._gauss_hermite_expectation(phi_star_times_w)
        
        return abs(delta)
    
    # === Part 4: Main certification routines ===
    
    def certify_point_from_estimates(self, C_ucb: float, G_ucb: float, E_est: Optional[float] = None) -> float:
        """
        Compute certified radius from pre-computed statistical estimates.
        
        Implements Algorithm 5 (ComputeRadius).
        
        Args:
            C_ucb: Upper confidence bound on variance
            G_ucb: Upper confidence bound on gradient norm
            E_est: Mean estimate (if None, uses self.mean_target)
            
        Returns:
            Certified radius R
        """
        C_ucb = float(max(0.0, C_ucb))
        G_ucb = float(max(0.0, G_ucb))
        
        if E_est is None:
            E_est = self.mean_target
        
        if C_ucb == 0.0 and G_ucb == 0.0:
            return 0.0
        
        def worst_harm_minus_eps(r: float) -> float:
            return self._worst_harm_bounded_with_mean(r, C_ucb, G_ucb, E_est) - self.eps_y
        
        # Root solve over radius:
        # Find r such that WorstHarmBounded(r) - eps_y = 0 with a bracketing interval.
        r_low = 0.0
        f_low = worst_harm_minus_eps(r_low)
        if f_low > 0.0:
            # Already violates at r=0
            return 0.0

        # Start with historical default 5*sigma, but expand if still safe.
        r_cap = max(0.0, self.r_cap_mult * self.sigma)
        r_high = max(0.0, self.r_high_init_mult * self.sigma)
        if r_cap > 0.0:
            r_high = min(r_high, r_cap)
        f_high = worst_harm_minus_eps(r_high)

        # Expand the bracket until we find a violating radius (f_high > 0) or hit the cap.
        while f_high <= 0.0 and r_high < r_cap:
            r_high = min(r_high * 2.0, r_cap)
            f_high = worst_harm_minus_eps(r_high)

        # If even the capped radius is safe, return the cap (lower bound on true certified radius).
        if f_high <= 0.0:
            return r_high
        
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
        seed: Optional[int] = None,
    ) -> float:
        """
        Full certification for a given point z with statistical estimation.
        
        Args:
            z: Input point to certify
            model_fn: Model function. If None, uses self.model_fn
            N_samples_stats: Number of samples for statistical estimation
            seed: Random seed
            
        Returns:
            Certified radius
        """
        if model_fn is None:
            if self.model_fn is None:
                raise ValueError("model_fn must be provided")
            model_fn = self.model_fn
        
        rng = np.random.default_rng(seed)
        
        # 1. Estimate statistical quantities with high-confidence bounds
        eta_samples = rng.normal(0.0, self.sigma, size=(N_samples_stats, z.shape[-1]))
        f_values = np.array([model_fn(z + eta) for eta in eta_samples])
        
        # Use α/3 split for union bound over 3 constraints
        _, _, C_ucb = self.u_statistic_variance_estimator_alpha_half(f_values)
        _, _, G_ucb = self.u_statistic_gradient_norm_estimator_alpha_half(f_values, eta_samples)
        E_hat, E_lower, E_upper = self.u_statistic_mean_estimator_alpha_third(f_values)
        
        # Use mean estimate (could use upper or lower bound for conservativeness)
        E_est = E_hat
        
        # 2. Compute certified radius
        return self.certify_point_from_estimates(C_ucb, G_ucb, E_est)


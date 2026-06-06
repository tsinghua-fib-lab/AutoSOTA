import os
import datetime

import jax
import jax.scipy.special as jss
import jax.numpy as jnp
import numpy as np

from inference.optimal_VI import OptimalVI


class CAVI(OptimalVI):
    def __init__(
        self,
        time,
        event,
        x,
        alpha_prior,
        beta_prior,
        rho,
        phi_MAP,
        theta_MAP,
        g,
        Z,
        num_points,
        batch_size,
        max_iter,
        workdir=None,
    ):
        OptimalVI.__init__(self)

        self.alpha_prior, self.beta_prior, self.rho = alpha_prior, beta_prior, rho
        self.g = g
        self.Z = Z
        self.phi_MAP, self.theta_MAP = phi_MAP, theta_MAP
        self.time, self.event, self.x = time, event, x
        self.N = len(self.time)
        self.size_theta = theta_MAP.shape[0]
        self.num_points, self.batch_size = num_points, batch_size
        self.max_iter = max_iter
        self.workdir = workdir

        # grid over which to evaluate the integrals
        self.time_grid = jnp.linspace(
            1e-16, jnp.max(self.time), self.num_points, dtype=jnp.float32
        )
        self.n = len(self.time_grid)

        # (t, x) pairs
        self.time_all = jnp.concatenate([self.time, jnp.tile(self.time_grid, self.N)])
        self.x_all = jnp.concatenate(
            [self.x, jnp.repeat(self.x, self.n, axis=0)], axis=0
        )

        # When time_grid <= y_i
        self.trap_mask = jnp.tile(self.time_grid, self.N) <= jnp.repeat(
            self.time, self.n, axis=0
        )
        index_last_true = jnp.sum(self.trap_mask.reshape(self.N, self.n), axis=1) - 1

        # Generic trapezoid weights
        dt = jnp.diff(self.time_grid)
        w = jnp.zeros_like(self.time_grid)
        w = w.at[1:-1].set(0.5 * (dt[1:] + dt[:-1]))
        w = w.at[0].set(0.5 * dt[0])
        w = jnp.tile(w, self.N).reshape(self.N, self.n)
        w = w.at[jnp.arange(self.N), index_last_true].set(0.5 * dt[-1])
        self.trap_w = w.reshape(self.N * self.n)

        # Evaluate Z over all (t, x) pairs
        self.Z_all = jax.vmap(self.Z, in_axes=(0, 0))(self.time_all, self.x_all)

        # Compile the functions
        self.setup()

        # Instantiate mu, Sigma for q_theta
        self.mu = theta_MAP
        self.w_B = jnp.zeros_like(self.time_all)

        # Get beta
        self.get_q_phi_beta()

        # Instantiate alpha for q_phi
        self.alpha = self.phi_MAP * self.beta
        self.digamma_alpha = jss.digamma(self.alpha)

        # Get beta for E_log_phi, m and s
        self.get_q_phi_beta()
        self.get_E_log_phi()
        self.get_m_and_s()

    def update(self):

        # Update q_omega, and E[omega]
        self.update_q_omega_c()
        self.get_E_omega()

        # Update q_Pi and integrals depending on Lambda
        self.update_q_Pi_Lambda()

        # Update q_phi and E[log phi]
        self.update_q_phi_alpha()
        self.get_E_log_phi()

        # Update q_theta and m = E[g^lin], s = sqrt(E[g^lin**2])
        self.update_q_theta_mu_and_Sigma()
        self.get_m_and_s()

    def fit(self, tol=1e-6):

        # Coordinate Ascent Variational Inference Algorithm
        convergence = False
        k = 0
        while convergence == False and k < self.max_iter:

            k += 1
            print("\nIteration:", k)

            # Keep track old params
            params_old = jnp.concatenate([self.mu, jnp.atleast_1d(self.alpha)])

            # Update
            self.update()

            # Compute convergence criteria
            numerator = np.linalg.norm(
                jnp.concatenate([self.mu, jnp.atleast_1d(self.alpha)]) - params_old
            )
            denominator = np.linalg.norm(params_old) + 1e-8
            assess_convergence = numerator / denominator
            print("Norm:", numerator)
            print("Convergence criterion:", assess_convergence)

            # Check convergence
            if assess_convergence < tol:
                print(f"Converged at iteration {k}")
                break

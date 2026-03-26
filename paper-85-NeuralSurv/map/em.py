import numpy as np
from scipy.optimize import minimize
import jax.numpy as jnp

from model.model_utils import *
from map.optimal_VI import OptimalVIParams
from map.integrals import *

jax.config.update("jax_enable_x64", True)  # Use float32 for all computations


class EMAlgorithm(OptimalVIParams, Integrals):
    def __init__(
        self,
        time,
        event,
        x,
        alpha_prior,
        beta_prior,
        rho,
        g,
        Z,
        theta_init,
        phi_init,
        num_points,
        batch_size,
        max_iter,
    ):
        OptimalVIParams.__init__(self)
        Integrals.__init__(self, num_points=num_points, batch_size=batch_size)

        self.alpha_prior, self.beta_prior = alpha_prior, beta_prior
        self.g = g
        self.rho, self.Z = rho, Z
        self.time, self.event, self.x = time, event, x
        self.max_iter = max_iter

        # Initialize parameters
        self.theta_current = theta_init
        self.phi_current = phi_init
        self.params = self.pack_params(self.theta_current, jnp.log(phi_init))

        # Keep track of the length of theta
        self.theta_dim = len(self.theta_current)

    def update_g_current(self):
        "g(t, x; theta^(l))"
        theta_current = self.theta_current
        self.g_current = lambda t, x: self.g(t, x, theta_current)

    def pack_params(self, theta, log_phi):
        return np.concatenate([theta, log_phi])

    def unpack_params(self, params):
        d = self.theta_dim
        theta = params[:d]
        phi = jnp.exp(params[d:])
        return theta, phi

    def e_step(self):
        """
        Compute expected values (E-step).
        """

        # Get integrals that do not depend on theta
        self.get_integrals_wo_theta()

        # Get g(y_i, x_i;theta^(l))
        g_current_i = self.g_current(self.time, self.x)

        @jax.jit
        def q_function(params):
            theta, phi = self.unpack_params(params)

            # Update integrals that depend on theta
            self.get_integrals_w_theta(theta)

            # g(y_i, x_i;theta)
            g_i = self.g(self.time, self.x, theta)

            # q function
            return (
                jnp.sum(
                    self.event * (g_i / 2)
                    - g_i**2 / (4 * jnp.abs(g_current_i)) * self.tanh_c_over_2
                )
                - 1 / 2 * self.integrals["L_g"]
                - 1 / 4 * self.integrals["L_g_2_E_omega_previous"]
                - 1 / 2 * jnp.sum(theta**2)
                + jnp.log(phi).squeeze()
                * (self.alpha_prior + jnp.sum(self.event) + self.integrals["L"] - 1)
                - phi.squeeze()
                * (
                    self.beta_prior
                    + jnp.sum(self.integrals["baseline_hazard_constant"])
                )
            )

        self.q_function = q_function
        self.q_grad = jax.jit(jax.grad(self.q_function))

    def m_step(self):
        """
        Maximize Q function numerically.
        """

        # print("Q value:", self.q_function(self.params))
        # print("Gradient norm:", jnp.linalg.norm(self.q_grad(self.params)))

        result = minimize(
            fun=lambda params: -self.q_function(params),  # Negate for maximization
            jac=lambda params: -self.q_grad(params),
            x0=np.array(self.params),
            method="L-BFGS-B",
            # options={
            #     "maxiter": 100,  # allow more iterations
            #     "disp": True,  # print progress
            #     "gtol": 1e-3,  # smaller gradient tolerance = more precision
            #     "eps": 1e-1,  # controls step size for finite-diff approx or scaling
            # },
        )
        return jnp.array(result.x, dtype=jnp.float32)

    def fit(self, tol=1e-6):

        # Instantiate empty list to save q-functions
        self.q_function_history = []
        rel_change_history = []

        for l in range(self.max_iter):

            print("\nIteration:", l + 1)

            # Update g(t, x ; theta^(l))
            self.update_g_current()

            # Update variational distribution parameters given theta^(l), phi^(l)
            self.update_q_omega_c()
            self.update_q_Pi_Lambda()

            # Expectation step, update q function
            self.e_step()

            # Compute Q-function
            self.q_function_history.append(self.q_function(self.params))
            print("Q value: ", self.q_function_history[-1])

            # Maximimization step
            new_params = self.m_step()

            # Check for convergence
            diff = abs(self.q_function(new_params) - self.q_function(self.params))
            rel_change_history.append(diff / abs(self.q_function(self.params)))
            print("Q function change:", rel_change_history[-1])
            if len(rel_change_history) >= 2 and all(
                change < tol for change in rel_change_history[-2:]
            ):
                print(f"Converged at iteration {l}")
                break

            # Update theta^(l) and phi^(l)
            self.params = new_params
            self.theta_current, self.phi_current = self.unpack_params(self.params)

        self.theta_MAP, self.phi_MAP = self.theta_current, self.phi_current

        return self.theta_MAP, self.phi_MAP

import datetime

import jax
import jax.scipy.special as jss
import jax.numpy as jnp
from jax import lax
from jax.scipy.sparse.linalg import cg

from inference.integrals import Integrals


# Note: cg inconsistent results between runs on GPUs: https://github.com/jax-ml/jax/issues/10255


class OptimalVIParams:

    def setup_OptimalVIParams(self):
        # Compile functions
        self.compile_G()
        self.compile_Lambda_fn()
        self.compile_c_fn()
        self.compile_Sigma_mul()
        self.compile_w_B()
        self.compile_mu_fn()

    def compile_G(self):

        @jax.jit
        def G(theta):
            return jax.vmap(lambda t, x: self.g(t, x, theta))(
                self.time_all, self.x_all
            ).squeeze()

        self.G = G

    def compile_Lambda_fn(self):
        "Lambda = [{Lambda_i(t)}_{i = 1, ..., N; t in time_grid}], shape (N n, )"

        time_integral = self.time_all[self.N :]
        Z_integral = self.Z_all[self.N :]

        @jax.jit
        def Lambda_fn(m, s, E_log_phi):
            # s: shape (N n, ), m: shape (N n, )
            return (
                time_integral ** (self.rho - 1)
                / Z_integral
                * jnp.exp(E_log_phi)
                * jax.nn.sigmoid(s)
                * jnp.exp(-(m + s) / 2)
                + 1e-6
            )  # to remove

        self.Lambda_fn = Lambda_fn

    def compile_c_fn(self):
        @jax.jit
        def c_fn(s):
            return self.event * s

        @jax.jit
        def tanh_c_fn(c):
            return jnp.tanh(c / 2)

        self.c_fn = c_fn
        self.tanh_c_fn = tanh_c_fn

    def compile_w_B(self):
        "w_B: [{delta_i E[omega_i]}_{i = 1 , ..., N} | {Lambda_i(t) tanh(s_i(t) / 1) / (2 s_i(t)) * dt}_ {t \in time_grid, i = 1, ..., N}]"

        @jax.jit
        def get_w_B(E_omega, Lambda, s_previous):
            """
            E_omega (N,)
            Lambda ( N n, )
            s (N n, )
            """

            # Part 1: (N,)
            w1_array = self.event * E_omega / 2  # shape (N,)

            # Part 2: (N n, )
            w2_array = (
                Lambda
                * jnp.tanh(s_previous / 2)
                / (2 * s_previous)
                * self.trap_w
                * self.trap_mask
                / 2
            )

            return jnp.concat([w1_array, w2_array])

        self.get_w_B = get_w_B

    def compile_Sigma_mul(self):
        """
        return v such that v = 1/2 * B^-1 A
        """
        # Pre‑build VJP closure at θ_map
        _, vjp_fun = jax.vjp(self.G, self.theta_MAP)

        def get_B_matvec(w_B):
            @jax.jit
            def B_matvec(v):
                # forward‐mode for J(θ_map)·v
                _, jvp = jax.jvp(self.G, (self.theta_MAP,), (v,))  # shape (R,)
                # weight and back‐propagate
                weighted = w_B * jvp  # shape (R,)
                (back,) = vjp_fun(weighted)  # shape (dθ,)
                return v / 2 + back

            return B_matvec

        @jax.jit
        def Sigma_mul(w_B, A):

            B_matvec = get_B_matvec(w_B)
            sol, _ = cg(B_matvec, A, maxiter=50, tol=1e-4)
            return sol / 2

        self.Sigma_mul = Sigma_mul

    def compile_mu_fn(self):

        N = self.N
        G_val = self.G(self.theta_MAP)
        ev_vals = G_val[:N]  # g(y_i, x_i)
        quad_vals = G_val[N:]  # g(t_{i,q}, x_i)
        trap_weights = self.trap_w * self.trap_mask

        @jax.jit
        def compute_mu(w_B, E_omega, Lambda, s_previous):
            "s_previous: shape (N n, ); Lambda: shape (N n, ); E_omega: shape (N, )"

            # J(t, x)^T theta_MAP for (t, x)  = (t_all, x_all)
            _, jvp_out = jax.jvp(
                self.G, (self.theta_MAP,), (self.theta_MAP,)
            )  # Shape (N n + N, )

            # delta_i * J(y_i, x_i) (1 - 2 E[omega_i](g(y_i, x_i) - J (y_i, x_i)^T theta_MAP)) / 2, for i = 1, ..., N
            w_event = 0.5 * jnp.where(
                self.event == 0,
                jnp.float32(0.0),
                (1.0 - 2.0 * E_omega * (ev_vals - jvp_out[: self.N])),
            )  # shape (N, )

            # Trapezoid-term weights
            factor = Lambda * (
                jnp.tanh(s_previous / 2) / (2.0 * s_previous)
            )  # Shape (N n , )
            w_I1 = Lambda * trap_weights  # I1, Shape (N n , )
            w_I2 = factor * quad_vals * trap_weights  # I2,
            w_I3 = factor * (jvp_out[self.N :]) * trap_weights  # I3, Shape (N n , )
            w_quad = -0.5 * w_I1 - w_I2 + w_I3  # Shape (N n , )

            # 4) Concatenate weights
            w_A = jnp.concatenate([w_event, w_quad])  # shape (N + N n,)

            # 6) Compute A = J(t_all, x_all)^T w_A
            _, vjp_fun = jax.vjp(self.G, self.theta_MAP)
            A = vjp_fun(w_A)[0]

            # 7) Solve for mu_tilde
            return self.Sigma_mul(w_B, A)

        self.compute_mu = compute_mu

    def update_q_omega_c(self):
        "tilde{c}_i"
        self.c = self.c_fn(self.s[: self.N])
        self.tanh_c_over_2 = self.tanh_c_fn(self.c)

    def update_q_Pi_Lambda(self):
        "tilde{lambda}_i(t)"

        # Capture the current value of self.E_log_phi
        E_log_phi_fixed = self.E_log_phi

        # Capture the current value of self.m and self.s
        m_fixed, s_fixed = self.m, self.s

        # Save Lambda
        self.Lambda = self.Lambda_fn(
            m_fixed[self.N :], s_fixed[self.N :], E_log_phi_fixed
        )

    def update_q_phi_alpha(self):
        "tilde{alpha}"
        Lambda_fixed = self.Lambda

        self.alpha = (
            self.alpha_prior + jnp.sum(self.event) + self.integral_L_fn(Lambda_fixed)
        )
        self.digamma_alpha = jss.digamma(self.alpha)

    def get_q_phi_beta(self):
        "tilde{beta}"
        self.beta = self.beta_prior + self.integral_baseline_hazard_constant
        self.log_beta = jnp.log(self.beta)

    def update_q_theta_mu_and_Sigma(self):
        "tilde{mu}, tilde{Sigma}"

        # Capture the current value of self.E_omega
        E_omega_fixed = self.E_omega
        Lambda_fixed = self.Lambda
        s_previous_fixed = self.s_previous[self.N :]

        # Compute Sigma_mul
        self.w_B = self.get_w_B(E_omega_fixed, Lambda_fixed, s_previous_fixed)

        # Compute mu
        self.mu = self.compute_mu(
            self.w_B, E_omega_fixed, Lambda_fixed, s_previous_fixed
        )


class OptimalVIExpectations:

    def setup_OptimalVIExpectations(self):
        self.compile_m_fn()
        self.compile_s_fn()

    def compile_m_fn(self):

        @jax.jit
        def compute_m(mu):
            # run one JVP over the whole batch:
            G_map, j_delta = jax.jvp(self.G, (self.theta_MAP,), (mu - self.theta_MAP,))
            # shape(G_map) = shape(j_delta) = (#time_all,)
            return G_map + j_delta

        self.compute_m = compute_m

    def compile_s_fn(self):

        def get_batch_eval(w_B, mu):

            @jax.jit
            def batch_eval(t, x):

                f_g = lambda theta: self.g(t, x, theta)

                # JVP for mean shift
                g_map, j_delta = jax.jvp(f_g, (self.theta_MAP,), (mu - self.theta_MAP,))

                b = jax.jacfwd(f_g)(self.theta_MAP).squeeze()

                # Quadratic term
                quad = jnp.dot(b, self.Sigma_mul(w_B, b))
                quad = jnp.where(quad < 0, 0.0, quad)
                # quad = jnp.vdot(b, Sigma_b)

                return jnp.sqrt((g_map + j_delta) ** 2 + quad) + 1e-6

            return batch_eval

        @jax.jit
        def compute_s(w_B, mu):
            batch_eval = get_batch_eval(w_B, mu)
            s = []
            for i in range(0, len(self.time_all), self.batch_size):
                t_batch = self.time_all[i : i + self.batch_size]
                x_batch = self.x_all[i : i + self.batch_size]
                s.append(
                    jax.vmap(batch_eval, in_axes=(0, 0))(t_batch, x_batch).squeeze()
                )

            return jnp.concatenate(s)

        self.compute_s = compute_s

    def get_E_log_phi(self):
        "E[log phi]"
        self.E_log_phi = self.digamma_alpha - self.log_beta

    def get_m_and_s(self):

        # Capture the current value of mu, Sigma
        mu_fixed = self.mu
        w_B_fixed = self.w_B

        # find m and s
        self.m = self.compute_m(mu_fixed)

        s = self.compute_s(w_B_fixed, mu_fixed)

        # Keep track of the previous s
        if hasattr(self, "s"):
            self.s_previous = self.s
        else:
            self.s_previous = s

        self.s = s

    def get_E_omega(self):
        "E[omega_i]"
        self.E_omega = jnp.where(
            self.event == 0, 0.0, (1 / (2 * self.c)) * self.tanh_c_over_2
        )


class OptimalVI(OptimalVIParams, OptimalVIExpectations, Integrals):
    def __init__(self):
        OptimalVIParams.__init__(self)
        OptimalVIExpectations.__init__(self)
        Integrals.__init__(self)

    def setup(self):
        self.setup_OptimalVIParams()
        self.setup_OptimalVIExpectations()
        self.setup_Integrals()

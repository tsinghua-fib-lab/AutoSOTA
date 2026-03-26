import jax
import jax.numpy as jnp


class Integrals:
    def __init__(self):
        self.integrals_train = {}  # integrals evaluated on the train set

    def setup_Integrals(self):
        self.get_integral_baseline_hazard_constant()
        self.compile_integral_L_fn()
        self.compile_integral_L_logL()
        self.compile_integral_L_logt()
        self.compile_integral_L_logZ()
        self.compile_integral_L_s_previous_E_omega_previous()
        self.compile_integral_L_lcosh_s_previous()
        self.compile_integral_L_m()
        self.compile_integral_ss_L_E_omega()

    def get_integral_baseline_hazard_constant(self):

        Z_integral = self.Z_all[self.N :]
        time_integral = self.time_all[self.N :]

        integrant = time_integral ** (self.rho - 1) / Z_integral
        self.integral_baseline_hazard_constant = jnp.sum(
            (self.trap_w * self.trap_mask * integrant),
        )

    def compile_integral_L_fn(self):

        @jax.jit
        def integral_L_fn(Lambda):
            return jnp.sum(
                (self.trap_w * self.trap_mask * Lambda),
            )

        self.integral_L_fn = integral_L_fn

    def compile_integral_L_logL(self):

        @jax.jit
        def integral_L_logL_fn(Lambda):
            return jnp.sum(
                (self.trap_w * self.trap_mask * Lambda * jnp.log(Lambda)).reshape(
                    (self.N, self.n)
                ),
            )

        self.integral_L_logL_fn = integral_L_logL_fn

    def compile_integral_L_logt(self):
        time_integral = self.time_all[self.N :]

        @jax.jit
        def integral_L_logt_fn(Lambda):
            return jnp.sum(
                (
                    self.trap_w * self.trap_mask * Lambda * jnp.log(time_integral)
                ).reshape((self.N, self.n)),
            )

        self.integral_L_logt_fn = integral_L_logt_fn

    def compile_integral_L_logZ(self):
        Z_integral = self.Z_all[self.N :]

        @jax.jit
        def integral_L_logZ_fn(Lambda):
            return jnp.sum(
                (self.trap_w * self.trap_mask * Lambda * jnp.log(Z_integral)).reshape(
                    (self.N, self.n)
                ),
            )

        self.integral_L_logZ_fn = integral_L_logZ_fn

    def compile_integral_L_s_previous_E_omega_previous(self):
        @jax.jit
        def integral_L_s_previous_E_omega_previous_fn(Lambda, s_previous):
            return jnp.sum(
                (
                    self.trap_w
                    * self.trap_mask
                    * Lambda
                    * (s_previous / 2)
                    * jnp.tanh(s_previous / 2)
                ).reshape((self.N, self.n)),
            )

        self.integral_L_s_previous_E_omega_previous_fn = (
            integral_L_s_previous_E_omega_previous_fn
        )

    def compile_integral_L_lcosh_s_previous(self):

        @jax.jit
        def integral_L_lcosh_s_previous_fn(Lambda, s_previous):
            return jnp.sum(
                (
                    self.trap_w
                    * self.trap_mask
                    * Lambda
                    * jnp.log(jnp.cosh(s_previous / 2))
                ).reshape((self.N, self.n)),
            )

        self.integral_L_lcosh_s_previous_fn = integral_L_lcosh_s_previous_fn

    def compile_integral_L_m(self):

        @jax.jit
        def integral_L_m_fn(Lambda, m):
            return jnp.sum(
                (self.trap_w * self.trap_mask * Lambda * m).reshape((self.N, self.n)),
            )

        self.integral_L_m_fn = integral_L_m_fn

    def compile_integral_ss_L_E_omega(self):

        @jax.jit
        def integral_ss_L_E_omega_fn(Lambda, s, s_previous):
            return jnp.sum(
                (
                    self.trap_w
                    * self.trap_mask
                    * s
                    * s
                    * Lambda
                    * (jnp.tanh(s_previous / 2) / (2 * s_previous))
                ).reshape((self.N, self.n)),
            )

        self.integral_ss_L_E_omega_fn = integral_ss_L_E_omega_fn

import jax
import jax.numpy as jnp


class OptimalVIParams:
    def update_q_omega_c(self):
        "breve{c}_i"
        self.c = self.event * jnp.abs(self.g_current(self.time, self.x))
        self.tanh_c_over_2 = jnp.tanh(self.c / 2)

    def update_q_Pi_Lambda(self):
        "breve{lambda}_i(t)"

        # Capture the current value
        phi_current, g_current = self.phi_current, self.g_current

        def q_Pi_Lambda(t, x):
            return (
                t ** (self.rho - 1)
                / self.Z(t, x)
                * phi_current
                * jax.nn.sigmoid(jnp.abs(g_current(t, x)))
                * jnp.exp(-(g_current(t, x) + jnp.abs(g_current(t, x))) / 2)
            )

        self.Lambda = q_Pi_Lambda

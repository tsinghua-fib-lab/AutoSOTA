import jax
import jax.numpy as jnp
from jax import custom_vjp


class Integrals:
    def __init__(self, num_points, batch_size):
        self.integrals = {}
        self.num_points = num_points
        self.batch_size = batch_size

    def get_integrals_wo_theta(self):
        "Integrals for Q function that does not depend on theta"
        self._cache_integral_fns()
        self._compute_integrals("Q_function_fixed")

    def get_integrals_w_theta(self, theta):
        "Integrals for Q function that does depend on theta"
        self._cache_integral_fns()
        self._compute_integrals("Q_function", theta)

    def _compute_integrals(self, type, theta=None):

        if type == "Q_function_fixed":
            keys = ["L", "baseline_hazard_constant"]
        elif type == "Q_function":
            keys = [
                "L_g",
                "L_g_2_E_omega_previous",
            ]

        for key in keys:
            self.integrals[key] = self._compute_integral(key, theta)

    def _compute_integral(self, key, theta):

        results = 0
        for i in range(0, len(self.time), self.batch_size):
            t_batch = self.time[i : i + self.batch_size]
            x_batch = self.x[i : i + self.batch_size]
            if theta is not None:
                results += jnp.sum(
                    jax.vmap(self.integral_fn[key], in_axes=(0, 0, None))(
                        t_batch, x_batch, theta
                    )
                )
            else:
                results += jnp.sum(
                    jax.vmap(self.integral_fn[key], in_axes=(0, 0))(t_batch, x_batch)
                )
        return results

    def _get_integral_theta_fn(self, func):

        @custom_vjp
        def integral_fn(time, x, theta):
            times = jnp.linspace(1e-16, time, self.num_points)
            return jnp.squeeze(
                jax.scipy.integrate.trapezoid(
                    jax.vmap(func, in_axes=(0, None, None))(times, x, theta),
                    times,
                    axis=0,
                )
            )

        def fwd(time, x, theta):
            result = integral_fn(time, x, theta)
            return result, (time, x, theta)

        def bwd(res, g):
            time, x, theta = res

            # Define a helper function that computes the integral for a given theta.
            def theta_fn(theta_prime):
                times = jnp.linspace(1e-16, time, self.num_points)
                return jnp.squeeze(
                    jax.scipy.integrate.trapezoid(
                        jax.vmap(func, in_axes=(0, None, None))(times, x, theta_prime),
                        times,
                        axis=0,
                    )
                )

            # Use jax.grad to obtain the full gradient with respect to theta.
            grad_theta = jax.grad(theta_fn)(theta)
            # The backward pass must return a tuple matching the shapes of (time, x, theta).
            # Here we are not computing derivatives with respect to time and x.
            return (None, None, g * grad_theta)

        integral_fn.defvjp(fwd, bwd)
        return jax.jit(integral_fn)

    def _get_integral_fn(self, func):

        @jax.jit
        def integral_trapezoidal_fn(time, x):
            times = jnp.linspace(1e-16, time, self.num_points)  # Discretize time range
            integral = jax.scipy.integrate.trapezoid(
                jax.vmap(func, in_axes=(0, None))(times, x), times, axis=0
            )  # Apply trapezoidal rule
            return integral

        return integral_trapezoidal_fn

    def _cache_integral_fns(self):
        self.integral_fn = {
            # int t^{rho-1}/Z(t,x) dt
            "baseline_hazard_constant": self._get_integral_fn(
                lambda t, x: t ** (self.rho - 1) / self.Z(t, x)
            ),
            # int Lambda(t,x) dt
            "L": self._get_integral_fn(lambda t, x: self.Lambda(t, x)),
            # int Lambda(t,x) g(t,x; theta) dt
            "L_g": self._get_integral_theta_fn(
                lambda t, x, theta: self.Lambda(t, x) * self.g(t, x, theta)
            ),
            # int Lambda(t,x) g(t,x;theta)**2 E_omega_previous dt, omega ~ PG(1, |g(t,x;theta_previous)|)
            "L_g_2_E_omega_previous": self._get_integral_theta_fn(
                lambda t, x, theta: self.Lambda(t, x)
                * self.g(t, x, theta) ** 2
                / jnp.abs(self.g_current(t, x))
                * jnp.tanh(jnp.abs(self.g_current(t, x)) / 2)
            ),
        }

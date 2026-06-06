from lifelines import WeibullFitter

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from model.model_utils import (
    get_jacobian_fn,
    get_Z_fn,
    from_params_to_theta,
    from_theta_to_params,
)


class NeuralSurvTools:

    def _find_Z(self):

        # Mean theta at prior
        theta_prior = jnp.zeros_like(from_params_to_theta(self.model_params_init))

        # Get g and J evaluated at theta prior
        J_prior = get_jacobian_fn(self.g, theta_prior)
        g_prior = lambda t, x: self.g(t, x, theta_prior)

        # Get Z
        self.Z = get_Z_fn(g_prior, J_prior, theta_prior)

    def _get_g(self):

        # Find unravel_fn function
        _, unravel_fn = ravel_pytree(self.model_params_init)

        def g(t, x, theta):
            model_params = from_theta_to_params(theta, unravel_fn)
            output = self.model.apply(
                model_params, t, x, mutable=False
            ).squeeze()  # Evaluate model
            return jnp.where(output == 0, 1e-8, output)

        self.g = g

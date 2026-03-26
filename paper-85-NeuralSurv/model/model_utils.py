import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax import traverse_util
from flax.core import freeze, unfreeze
from jax.flatten_util import ravel_pytree


def create_state(rng, model, in_shape_x, learning_rate=0.1):
    params = model.init(rng, jnp.zeros(in_shape_x))
    tx = optax.adam(learning_rate)  # Our optimizer
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def get_jacobian_fn(g, theta_fixed):
    "Get jacobian of g evaluated at theta_fixed"

    @jax.jit
    def jacobian_fn(t, x):
        return jnp.squeeze(
            jax.jacrev(lambda theta: g(t, x, theta).squeeze())(theta_fixed)
        )

    return jacobian_fn


def get_Z_fn(g, J, theta):
    return lambda t, x: jnp.maximum(
        1e-16,
        jax.nn.sigmoid(
            g(t, x)
            - jnp.sum(J(t, x) * theta)
            / jnp.sqrt(1 + jnp.pi / 8 * jnp.sum(J(t, x) ** 2))
        ),
    ).squeeze()


def from_params_to_theta(model_params):
    theta, _ = ravel_pytree(model_params)
    return theta


def from_theta_to_params(theta, unravel_fn):
    return unravel_fn(theta)

import jax
import jax.numpy as jnp

from jax.tree_util import tree_map, tree_flatten


@jax.jit
def l2_norm(tree):
    """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
    leaves, _ = tree_flatten(tree)
    return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))


@jax.jit
def clip_grads(grad_tree, max_norm):
    """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
    norm = l2_norm(grad_tree)
    normalize = lambda g: jnp.where(norm < max_norm, g, g * (max_norm / norm))
    return tree_map(normalize, grad_tree)

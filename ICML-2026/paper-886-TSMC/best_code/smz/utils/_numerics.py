import jax

from jaxtyping import PyTree


@jax.jit
def tree_compare(a: PyTree[jax.Array], b: PyTree[jax.Array]) -> jax.Array:
    bool_tree = jax.tree_util.tree_map(jax.numpy.array_equal, a, b)
    return jax.tree.reduce(jax.numpy.logical_and, bool_tree)


@jax.jit
def compute_pmf(
        true_actions: jax.Array,
        atoms: jax.Array,
        mass: jax.Array
) -> jax.Array:
    """Get the canonical PMF from bootstrapped atoms (discrete space only)

    """

    def match(a: jax.Array) -> jax.Array:
        return (jax.vmap(jax.numpy.array_equal,
                         in_axes=(0, None))(atoms, a) * mass).sum()

    return jax.vmap(match)(true_actions)

@jax.jit
def compute_values_from_branches(
        true_actions: jax.Array,
        atoms: jax.Array,
        values: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Get the canonical PMF from bootstrapped atoms (discrete space only)

    """

    def match(a: jax.Array) -> tuple[jax.Array, jax.Array]:
        valuesum = (jax.vmap(jax.numpy.array_equal,
                         in_axes=(0, None))(atoms, a) * values).sum()
        num = (jax.vmap(jax.numpy.array_equal,
                         in_axes=(0, None))(atoms, a)).sum()
        return valuesum / jax.numpy.clip(num, a_min=1), num > 0

    return jax.vmap(match)(true_actions)


@jax.jit
def compute_coverage_mask_from_branches(
        true_actions: jax.Array,
        atoms: jax.Array,
        values: jax.Array
) -> jax.Array:
    """Get the canonical PMF from bootstrapped atoms (discrete space only)

    """

    def match(a: jax.Array) -> jax.Array:
        num = (jax.vmap(jax.numpy.array_equal,
                         in_axes=(0, None))(atoms, a)).sum()
        return num > 0

    return jax.vmap(match)(true_actions)


@jax.jit
def num_unique_atoms(true_actions: jax.Array, atoms: jax.Array) -> jax.Array:
    """Returns how many of `true_actions` appear at least once in `atoms`."""
    def match(a: jax.Array) -> jax.Array:
        return jax.numpy.any(jax.vmap(jax.numpy.array_equal, in_axes=(0, None))(atoms, a))
    return jax.numpy.sum(jax.vmap(match)(true_actions))


@jax.jit
def compute_true_root_values(
        true_actions: jax.Array,
        atoms: jax.Array,
        root_values: jax.Array
) -> jax.Array:
    """Compute average root value per true action."""

    def match(a: jax.Array) -> jax.Array:
        matches = jax.vmap(jax.numpy.array_equal, in_axes=(0, None))(atoms, a)
        total = (matches * root_values).sum()
        count = matches.sum()
        return jax.numpy.where(count > 0, total / count, 0.0)  # , jax.numpy.nan

    return jax.vmap(match)(true_actions)

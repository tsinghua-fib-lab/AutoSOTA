"""
Generic jax helper functions
"""

import jax
import jax.numpy as jnp
import numpy as np


def is_array(x):
    return isinstance(x, (np.ndarray, jnp.ndarray))

def partition(pytree, filter_fn=is_array):
    "Splits a pytree into its dynamic and static leaves"
    leaves, treedef = jax.tree_util.tree_flatten(pytree)# note treedef is hashable
    dynamic_leaves = tuple(x if filter_fn(x) else None for x in leaves)# tuple makes sure hashable
    static_leaves = tuple(None if filter_fn(x) else x for x in leaves)# tuple makes sure hashable
    return dynamic_leaves, (static_leaves, treedef)

def combine(dynamic_leaves, static):
    "Recombines dynamic and static leaves of a pytree into a single tree"
    static_leaves, treedef = static
    leaves = [d if s is None else s for d, s in zip(dynamic_leaves, static_leaves)]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def str_tensor(x):
    "Fancy str a tensor"
    if is_array(x):
        return f"{x.shape}, {x.dtype}, {type(x).__name__}"
    else:
        return x

def tree_index(pytree, i):
    "Index element in tree"
    return jax.tree_util.tree_map(lambda x: x[i], pytree)

def total_size(pytree):
    "Returns total number of elements in pytree of arrays"
    p = sum([p.size for p in jax.tree_util.tree_leaves(pytree)])
    return p


if __name__ == "__main__":

    f = lambda x: x
    class C:
        pass
    pytree = {"a":[jnp.arange(10), None, "asd"], "b":(0, True, f, C, C(), [0,1], np.arange(3))}
    print(pytree)
    print()

    d, s = partition(pytree)

    print(combine(d, s))
    print()

    print(d)
    print()

    print(s)
    print(hash(s))
    print()

    print(str_tensor(jnp.arange(10)))
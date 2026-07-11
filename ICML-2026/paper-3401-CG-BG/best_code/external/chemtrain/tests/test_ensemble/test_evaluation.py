from chemtrain.ensemble import evaluation

import jax.numpy as jnp

import jax_md_mod
from jax_md import partition, space

import numpy as onp

import pytest

class TestEvaluationBatching:

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 6, 7])
    def test_batch_sizes(self, batch_size):
        sample_size = 6

        position = jnp.arange(sample_size).reshape((-1, 1, 1)) * jnp.ones((1, 1, 3))

        displacement_fn, _ = space.free()
        neighbor_fn = partition.neighbor_list(
            displacement_fn, 0.0, 0.5, disable_cell_list=True
        )
        nbrs = neighbor_fn.allocate(position[0])

        data = evaluation.SimpleState(position=position)
        compute_fn = lambda state, *args, **kwargs: state.position[0, 0]

        quantities = evaluation.quantity_map(
            data, {"idx": compute_fn}, nbrs, batch_size=batch_size)

        print(quantities["idx"])
        print(position[..., 0, 0].flatten())
        print(quantities["idx"] == position[..., 0, 0].flatten())
        assert onp.all(quantities["idx"] == position[..., 0, 0].flatten())

import pytest
import ntkunlimited as nq
import jax
import jax.numpy as jnp
import neural_tangents as nt
from ntkunlimited.nn import ParameterizationSetup
from ntkunlimited.empirical.tensor_convergence_all_layers import (
    calc_emp_ntks_all_layers,
    calc_emp_nngps_all_layers,
)


def calc_emp_ntk_last_layer(layers, all_params, x1):
    init_fn, apply_fn, _ = nt.stax.serial(*layers)
    ntk_fn = nt.empirical_ntk_fn(apply_fn, trace_axes=())
    ntk_final = jnp.diagonal(ntk_fn(x1, None, all_params), axis1=-2, axis2=-1)

    return ntk_final


@pytest.mark.parametrize("seed", [3463])
def test_calc_emp_ntks_all_layers(seed):
    key = jax.random.key(seed)
    parameterization_setup = ParameterizationSetup("standard", 2.0, 0.0, convert_to_book=False)
    layers = nq.nn.create_network(3, 10, "Relu", parameterization_setup)
    layers = jax.tree.map(jax.tree_util.Partial, layers)
    x1 = jax.random.normal(key, (2, 10))

    key, subkey = jax.random.split(key)
    ntks, params = calc_emp_ntks_all_layers(
        layers, subkey, x1, None, None, False, parameterization_setup
    )

    ntk_final = calc_emp_ntk_last_layer(layers, params, x1)

    assert jnp.allclose(ntks[-1], ntk_final)


@pytest.mark.parametrize("seed", [3463])
def test_calc_emp_ntk_first_layer(seed):
    key = jax.random.key(seed)
    parameterization_setup = ParameterizationSetup("standard", 2.0, None, convert_to_book=False)
    layers = nq.nn.create_network(
        2, 10, "Identity", parameterization_setup
    )
    layers = jax.tree.map(jax.tree_util.Partial, layers)
    x1 = jax.random.normal(key, (2, 10))

    key, subkey = jax.random.split(key)
    ntks, params = calc_emp_ntks_all_layers(
        layers, subkey, x1, None, None, False, parameterization_setup
    )
    ntk_first = jnp.einsum("ai,bi->ab", x1, x1)

    assert jnp.allclose(ntks[1][..., 0], ntk_first)


def calc_emp_nngp_last_layer(layers, all_params, x1):
    init_fn, apply_fn, _ = nt.stax.serial(*layers)
    nngp_fn = nt.empirical_nngp_fn(apply_fn, trace_axes=())
    nngp_final = jnp.diagonal(nngp_fn(x1, None, all_params), axis1=-2, axis2=-1)

    return nngp_final


@pytest.mark.parametrize("seed", [3463])
def test_calc_emp_nngp_all_layers(seed):
    key = jax.random.key(seed)
    parameterization_setup = ParameterizationSetup("standard", 2.0, 0.0)
    layers = nq.nn.create_network(3, 10, "Relu", parameterization_setup)
    layers = jax.tree.map(jax.tree_util.Partial, layers)
    x1 = jax.random.normal(key, (2, 10))

    key, subkey = jax.random.split(key)
    nngps, params = calc_emp_nngps_all_layers(
        layers, subkey, x1, None, None, False
    )

    nngp_final = calc_emp_nngp_last_layer(layers, params, x1)

    assert jnp.allclose(nngps[-1], nngp_final, atol=1e-6)

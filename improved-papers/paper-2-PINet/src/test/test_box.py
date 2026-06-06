"""Tests for the box constraint."""

from itertools import product

import jax
import jax.numpy as jnp
import pytest

from pinet import BoxConstraint, BoxConstraintSpecification, ProjectionInstance

DIM = 100


@pytest.mark.parametrize(
    "lb, ub, x, y",
    [
        ([0, 0], [1, 1], [2, 2], [1, 1]),
        ([0, 0], [1, 2], [2, 2], [1, 2]),
        ([0, 0], [1, 1], [0.5, 0.5], [0.5, 0.5]),
        ([0, 0], [1, 1], [-0.5, -0.5], [0, 0]),
        ([-0.5, 0], [0.5, 1], [-0.5, 1.5], [-0.5, 1]),
        ([-0.5, 0], [0.5, 1], [-1.5, 1.5], [-0.5, 1]),
    ],
)
def test_box(lb, ub, x, y):
    lb = jnp.array(lb).reshape((1, 2, 1))
    ub = jnp.array(ub).reshape((1, 2, 1))
    x = jnp.array(x).reshape((1, 2, 1))
    y = jnp.array(y).reshape((1, 2, 1))

    box_constraint = BoxConstraint(BoxConstraintSpecification(lb=lb, ub=ub))
    z = box_constraint.project(ProjectionInstance(x=x)).x

    assert jnp.allclose(
        y, z
    ), f"""
        Projection of {x} onto:
            lb: {lb}
            ub: {ub}
        should be {y}, instead of {z}.
    """


def test_mask():
    box_constraint = BoxConstraint(
        BoxConstraintSpecification(
            lb=jnp.array([1]).reshape((1, 1, 1)),
            ub=jnp.array([2]).reshape((1, 1, 1)),
            mask=jnp.array([1, 0], dtype=jnp.bool_),
        )
    )
    x = jnp.array([[-2, 2]]).reshape((1, 2, 1))
    y = box_constraint.project(ProjectionInstance(x=x)).x

    assert y[0, 0, 0] == 1, "The first element should be clipped to 1."
    assert y[0, 1, 0] == 2, "The second element should not be clipped."


@pytest.mark.parametrize(
    "n_batch_l, n_batch_u, n_batch_x, seed",
    product([1, 10], [1, 10], [1, 10], [24, 42]),
)
def test_box_parametrized(n_batch_l, n_batch_u, n_batch_x, seed):
    if n_batch_l > n_batch_x or n_batch_u > n_batch_x:
        return
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key, num=4)
    mask = jax.random.bernoulli(key[0], shape=(DIM)).astype(jnp.bool_)
    active_entries = mask.sum().item()

    lb = jax.random.uniform(
        key[1], shape=(n_batch_l, active_entries, 1), minval=0, maxval=1
    )
    ub = jax.random.uniform(
        key[2], shape=(n_batch_u, active_entries, 1), minval=lb.max(axis=0), maxval=1
    )
    x = jax.random.uniform(key[3], shape=(n_batch_x, DIM, 1), minval=-2, maxval=2)

    # static bounds passed at construction
    box_constraint_static = BoxConstraint(
        BoxConstraintSpecification(
            lb=lb,
            ub=ub,
            mask=mask,
        )
    )
    z_static = box_constraint_static.project(ProjectionInstance(x=x)).x

    # Compute projection with for loop (baseline)
    y = x.copy()
    for ii in range(n_batch_x):
        y = y.at[ii, mask, :].set(
            jnp.clip(
                x[ii, mask, :],
                min=lb[min(ii, n_batch_l - 1), :, :],
                max=ub[min(ii, n_batch_u - 1), :, :],
            )
        )

    assert jnp.allclose(y, z_static), "Projection (static) should match the for loop."

    # variable bounds from ProjectionInstance
    base_lb = jnp.zeros((1, active_entries, 1))
    base_ub = jnp.ones((1, active_entries, 1))
    box_constraint_variable = BoxConstraint(
        BoxConstraintSpecification(lb=base_lb, ub=base_ub)
    )

    proj = ProjectionInstance(
        x=x,
        box=BoxConstraintSpecification(lb=lb, ub=ub, mask=mask),
    )
    z_variable = box_constraint_variable.project(proj).x

    # Should match the same baseline and the static version
    assert jnp.allclose(
        y, z_variable
    ), "Projection (variable) should match the for loop."
    assert jnp.allclose(
        z_static, z_variable
    ), "Static and variable bound paths should be equivalent."


@pytest.mark.parametrize(
    "lb, ub, x, expected",
    [
        # Case 1: per-batch variable bounds (each batch has its own lb/ub)
        (
            jnp.array([[0.0, 0.0], [-1.0, 0.5], [0.2, -2.0]]).reshape((3, 2, 1)),
            jnp.array([[1.0, 2.0], [0.5, 0.5], [0.2, -1.0]]).reshape((3, 2, 1)),
            jnp.array([[2.0, -1.0], [0.7, 0.2], [0.0, -3.0]]).reshape((3, 2, 1)),
            jnp.array([[1.0, 0.0], [0.5, 0.5], [0.2, -2.0]]).reshape((3, 2, 1)),
        ),
        # Case 2: single set of bounds (batch=1) broadcast across multiple x batches
        (
            jnp.array([[0.0, 0.0]]).reshape((1, 2, 1)),
            jnp.array([[1.0, 1.0]]).reshape((1, 2, 1)),
            jnp.array([[2.0, -1.0], [0.3, 0.7], [-0.2, 4.0]]).reshape((3, 2, 1)),
            jnp.array([[1.0, 0.0], [0.3, 0.7], [0.0, 1.0]]).reshape((3, 2, 1)),
        ),
        # Case 3: variable lower bounds only;
        # upper bounds come from the constraint template
        (
            jnp.array([[-1.0, -1.0], [0.5, 0.0]]).reshape((2, 2, 1)),
            None,
            jnp.array([[-2.0, 2.0], [0.3, -2.0]]).reshape((2, 2, 1)),
            jnp.array([[-1.0, 1.0], [0.5, 0.0]]).reshape((2, 2, 1)),
        ),
    ],
    ids=["per-batch-bounds", "broadcast-bounds", "lb-only"],
)
def test_box_variable_bounds_from_projection_instance_param(lb, ub, x, expected):
    # Template defines dimension; its values get overridden by ProjectionInstance.box
    base_lb = jnp.zeros((1, 2, 1))
    base_ub = jnp.ones((1, 2, 1))
    box_constraint = BoxConstraint(BoxConstraintSpecification(lb=base_lb, ub=base_ub))

    proj = ProjectionInstance(
        x=x,
        box=BoxConstraintSpecification(lb=lb, ub=ub),
    )
    z = box_constraint.project(proj).x

    assert jnp.allclose(
        z, expected
    ), f"""
        Variable-bounds projection failed.
        x: {x} lb: {lb} ub: {ub}
        expected: {expected}
        got: {z}
    """


@pytest.mark.parametrize(
    "base_lb, base_ub, var_lb, var_ub, x, expected",
    [
        (
            jnp.zeros((1, 2, 1)),
            None,
            jnp.array([[-1.0, -1.0], [0.5, 0.0]]).reshape((2, 2, 1)),
            None,
            jnp.array([[-2.0, 2.0], [0.3, -2.0]]).reshape((2, 2, 1)),
            jnp.array([[-1.0, 2.0], [0.5, 0.0]]).reshape((2, 2, 1)),
        ),
        (
            None,
            jnp.ones((1, 2, 1)),
            None,
            jnp.array([[1.0, 1.0], [0.4, 0.6]]).reshape((2, 2, 1)),
            jnp.array([[2.0, -1.0], [0.3, 0.7]]).reshape((2, 2, 1)),
            jnp.array([[1.0, -1.0], [0.3, 0.6]]).reshape((2, 2, 1)),
        ),
        (
            jnp.zeros((1, 3, 1)),
            jnp.ones((1, 3, 1)),
            jnp.array([[0.0, -1.0, 0.2]]).reshape((1, 3, 1)),
            jnp.array([[1.0, 0.0, 0.5]]).reshape((1, 3, 1)),
            jnp.array([[2.0, -2.0, 0.3], [-0.5, 0.5, 1.0]]).reshape((2, 3, 1)),
            jnp.array([[1.0, -1.0, 0.3], [0.0, 0.0, 0.5]]).reshape((2, 3, 1)),
        ),
    ],
    ids=["var-lb-only-no-ub", "var-ub-only-no-lb", "no-mask-variable-bounds"],
)
def test_box_variable_bounds_edge_cases(base_lb, base_ub, var_lb, var_ub, x, expected):
    box_constraint = BoxConstraint(BoxConstraintSpecification(lb=base_lb, ub=base_ub))

    proj = ProjectionInstance(
        x=x,
        box=BoxConstraintSpecification(lb=var_lb, ub=var_ub),
    )
    z = box_constraint.project(proj).x

    assert jnp.allclose(
        z, expected
    ), f"""
        Edge-case variable-bounds projection failed.
        base_lb: {base_lb}, base_ub: {base_ub}
        var_lb: {var_lb}, var_ub: {var_ub}
        x: {x}
        expected: {expected}
        got: {z}
    """

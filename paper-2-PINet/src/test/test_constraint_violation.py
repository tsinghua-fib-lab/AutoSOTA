"""Test constraint violation functionality."""

from itertools import product

import cvxpy as cp
import jax
import jax.numpy as jnp
import pytest

from pinet import (
    AffineInequalityConstraint,
    BoxConstraint,
    BoxConstraintSpecification,
    EqualityConstraint,
    EqualityConstraintsSpecification,
    Project,
    ProjectionInstance,
)

jax.config.update("jax_enable_x64", True)


def test_box_cv():
    lower_bound = jnp.array([-1.0, -2.0]).reshape(1, -1, 1)
    upper_bound = jnp.array([1.0, 2.0]).reshape(1, -1, 1)
    mask = jnp.array([True, False, True])
    box_constraint = BoxConstraint(
        BoxConstraintSpecification(lb=lower_bound, ub=upper_bound, mask=mask)
    )

    x = jnp.concatenate(
        [
            jnp.array([1.5, 10.0, 1.0]).reshape(1, -1, 1),
            jnp.array([-2.0, 10.0, 1.0]).reshape(1, -1, 1),
            jnp.array([1.0, 10.0, 3.5]).reshape(1, -1, 1),
            jnp.array([1.0, 10.0, -4.0]).reshape(1, -1, 1),
        ],
        axis=0,
    )
    gt_cv = jnp.array([0.5, 1.0, 1.5, 2.0]).reshape(-1, 1, 1)
    box_cv = box_constraint.cv(ProjectionInstance(x=x))
    assert jnp.allclose(box_cv, gt_cv), f"Expected {gt_cv}, but got {box_cv}."

    # Check that the projection has zero cv
    x_proj = box_constraint.project(ProjectionInstance(x=x))
    box_cv_proj = box_constraint.cv(x_proj)
    assert jnp.allclose(box_cv_proj, 0.0), f"Expected 0.0, but got {box_cv_proj}."


def test_equality_cv():
    A = jnp.array([1.0, 2.0, 0]).reshape(1, 1, 3)
    b = jnp.array([3.0]).reshape(1, 1, 1)
    eq_constraint = EqualityConstraint(A, b, method="pinv")

    x = jnp.concatenate(
        [
            jnp.array([1.0, 2.0, 5.0]).reshape(1, -1, 1),
            jnp.array([3.0, 2.0, 2.0]).reshape(1, -1, 1),
            jnp.array([1.0, -1.0, 3.0]).reshape(1, -1, 1),
            jnp.array([-5.0, 1.0, 4.0]).reshape(1, -1, 1),
        ],
        axis=0,
    )

    gt_cv = jnp.array([2.0, 4.0, 4.0, 6.0]).reshape(-1, 1, 1)
    eq_cv = eq_constraint.cv(ProjectionInstance(x=x))
    assert jnp.allclose(eq_cv, gt_cv), f"Expected {gt_cv}, but got {eq_cv}."

    # Check that the projection has zero cv
    x_proj = eq_constraint.project(ProjectionInstance(x=x))
    eq_cv_proj = eq_constraint.cv(x_proj)
    assert jnp.allclose(eq_cv_proj, 0.0), f"Expected 0.0, but got {eq_cv_proj}."

    # Do a random test
    m = 10
    n = 20
    batch_size = 10
    key = jax.random.PRNGKey(42)
    key = jax.random.split(key, num=3)
    A = jax.random.normal(key[0], shape=(1, m, n))
    b = jax.random.normal(key[1], shape=(1, m, 1))
    x = jax.random.normal(key[2], shape=(batch_size, n, 1))
    eq_constraint = EqualityConstraint(A, b, method="pinv")

    # Ground truth
    gt_cv = jnp.max(jnp.abs(A @ x - b), axis=1, keepdims=True)
    eq_cv = eq_constraint.cv(ProjectionInstance(x=x))
    assert jnp.allclose(eq_cv, gt_cv), f"Expected {gt_cv}, but got {eq_cv}."

    # Check that the projection has zero cv
    x_proj = eq_constraint.project(ProjectionInstance(x=x))
    eq_cv_proj = eq_constraint.cv(x_proj)
    assert jnp.allclose(eq_cv_proj, 0.0), f"Expected 0.0, but got {eq_cv_proj}."


def test_inequality_cv():
    C = jnp.array([1.0, 2.0, 0]).reshape(1, 1, 3)
    lb = jnp.array([0.0]).reshape(1, 1, 1)
    ub = jnp.array([2.0]).reshape(1, 1, 1)
    ineq_constraint = AffineInequalityConstraint(C, lb, ub)

    x = jnp.concatenate(
        [
            jnp.array([1.0, 2.0, 5.0]).reshape(1, -1, 1),
            jnp.array([3.0, 2.0, 2.0]).reshape(1, -1, 1),
            jnp.array([1.0, -1.0, 3.0]).reshape(1, -1, 1),
            jnp.array([-5.0, 1.0, 4.0]).reshape(1, -1, 1),
        ],
        axis=0,
    )
    yraw = ProjectionInstance(x=x)

    gt_cv = jnp.array([3.0, 5.0, 1.0, 3.0]).reshape(-1, 1, 1)
    ineq_cv = ineq_constraint.cv(yraw)
    assert jnp.allclose(ineq_cv, gt_cv), f"Expected {gt_cv}, but got {ineq_cv}."

    # Check that the projection has zero cv
    # The inequality constraint does not implement project.
    projection_layer = Project(ineq_constraint=ineq_constraint)
    x_proj = projection_layer.call(
        yraw=yraw,
        n_iter=100,
    )[0]
    ineq_cv_proj = ineq_constraint.cv(x_proj)
    assert jnp.allclose(ineq_cv_proj, 0.0), f"Expected 0.0, but got {ineq_cv_proj}."
    # Do a random test
    m = 10
    n = 20
    batch_size = 10
    key = jax.random.PRNGKey(42)
    key = jax.random.split(key, num=4)
    C = jax.random.normal(key[0], shape=(1, m, n))
    lb = jax.random.normal(key[1], shape=(1, m, 1))
    ub = lb + jnp.abs(jax.random.normal(key[2], shape=(1, m, 1)))
    x = jax.random.normal(key[3], shape=(batch_size, n, 1))
    yraw = ProjectionInstance(x=x)
    ineq_constraint = AffineInequalityConstraint(C, lb, ub)

    # Ground truth
    gt_cv = jnp.max(
        jnp.maximum(jnp.maximum(C @ x - ub, 0), jnp.maximum(lb - C @ x, 0)),
        axis=1,
        keepdims=True,
    )
    ineq_cv = ineq_constraint.cv(yraw)
    assert jnp.allclose(ineq_cv, gt_cv), f"Expected {gt_cv}, but got {ineq_cv}."

    # Check that the projection has zero cv
    projection_layer = Project(ineq_constraint=ineq_constraint)
    x_proj = projection_layer.call(
        yraw=yraw,
        n_iter=100,
    )[0]
    ineq_cv_proj = ineq_constraint.cv(x_proj)
    assert jnp.allclose(ineq_cv_proj, 0.0), f"Expected 0.0, but got {ineq_cv_proj}."


def test_inequality_box_cv():
    # Define inequality constraint
    C = jnp.array([0.5, 1.0]).reshape(1, 1, 2)
    lb = jnp.array([-jnp.inf]).reshape(1, 1, 1)
    ub = jnp.array([1.0]).reshape(1, 1, 1)
    ineq_constraint = AffineInequalityConstraint(C, lb, ub)

    # Define box constraints
    lower_bound = jnp.array([0.0, 0.0]).reshape(1, -1, 1)
    upper_bound = jnp.array([jnp.inf, jnp.inf]).reshape(1, -1, 1)
    box_constraint = BoxConstraint(
        BoxConstraintSpecification(lb=lower_bound, ub=upper_bound)
    )

    # Define projection layer
    projection_layer = Project(
        ineq_constraint=ineq_constraint, box_constraint=box_constraint
    )

    x = jnp.concatenate(
        [
            jnp.array([0.5, -2.0]).reshape(1, -1, 1),
            jnp.array([-1.5, 0.5]).reshape(1, -1, 1),
            jnp.array([2.5, 1.0]).reshape(1, -1, 1),
        ],
        axis=0,
    )
    yraw = ProjectionInstance(x=x)
    gt_cv = jnp.array([2.0, 1.5, 0.5 * 2.5 + 1.0 * 1.0 - 1.0]).reshape(-1, 1, 1)
    cv = projection_layer.cv(y=yraw)
    assert jnp.allclose(cv, gt_cv), f"Expected {gt_cv}, but got {cv}."

    # Check that the projection has zero cv
    x_proj = projection_layer.call(
        yraw=yraw,
        n_iter=100,
    )[0]
    cv_proj = projection_layer.cv(x_proj)
    assert jnp.allclose(cv_proj, 0.0), f"Expected 0.0, but got {cv_proj}."


@pytest.mark.parametrize("method", ["pinv", None])
@pytest.mark.parametrize("seed, batch_size", product([24, 42], [1, 5]))
def test_equality_inequality_box_cv(method, seed, batch_size):
    dim = 100
    n_eq = 50
    n_ineq = 40
    n_box = 15
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key, num=5)
    # Generate equality constraints LHS
    A = jax.random.normal(key[0], shape=(1, n_eq, dim))
    # Generate inequality constraints LHS
    C = jax.random.normal(key[1], shape=(1, n_ineq, dim))
    # Randomly generate mask for box constraints
    indices = jnp.concatenate([jnp.ones(n_box), jnp.zeros(dim - n_box)])
    mask = jax.random.permutation(key[2], indices).astype(bool)
    # Compute RHS by solving feasibility problem
    xfeas = cp.Variable(dim * batch_size)
    bfeas = cp.Variable(n_eq * batch_size)
    lfeas = cp.Variable(n_ineq)
    ufeas = cp.Variable(n_ineq)
    lboxfeas = cp.Variable(n_box)
    uboxfeas = cp.Variable(n_box)
    constraints = [
        -1 <= lfeas,
        lfeas <= 1,
        -1 <= ufeas,
        ufeas <= 1,
        -1 <= lboxfeas,
        lboxfeas <= 1,
        -1 <= uboxfeas,
        uboxfeas <= 1,
    ]
    for ii in range(batch_size):
        constraints += [
            A[0, :, :] @ xfeas[ii * dim : (ii + 1) * dim]
            == bfeas[ii * n_eq : (ii + 1) * n_eq],
            lfeas <= C[0, :, :] @ xfeas[ii * dim : (ii + 1) * dim],
            C[0, :, :] @ xfeas[ii * dim : (ii + 1) * dim] <= ufeas,
            lboxfeas <= xfeas[ii * dim : (ii + 1) * dim][mask],
            xfeas[ii * dim : (ii + 1) * dim][mask] <= uboxfeas,
            -2 <= xfeas,
            xfeas <= 2,
            ii <= bfeas[ii * n_eq : (ii + 1) * n_eq],
        ]
    objective = cp.Minimize(cp.sum(xfeas))
    problem = cp.Problem(objective=objective, constraints=constraints)
    problem.solve(verbose=False)
    # Extract RHS parameters
    b = jnp.tile(jnp.array(bfeas.value).reshape((batch_size, n_eq, 1)), (1, 1, 1))
    lb = jnp.tile(jnp.array(lfeas.value).reshape((1, n_ineq, 1)), (1, 1, 1))
    ub = jnp.tile(jnp.array(ufeas.value).reshape((1, n_ineq, 1)), (1, 1, 1))
    lbox = jnp.tile(jnp.array(lboxfeas.value).reshape((1, n_box, 1)), (1, 1, 1))
    ubox = jnp.tile(jnp.array(uboxfeas.value).reshape((1, n_box, 1)), (1, 1, 1))
    # Point to be projected
    x = jax.random.uniform(key[3], shape=(batch_size, dim), minval=-2, maxval=2)
    # Define the projection layer
    eq_constraint = EqualityConstraint(A=A, b=b, method=method, var_b=True)
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)
    box_constraint = BoxConstraint(
        BoxConstraintSpecification(lb=lbox, ub=ubox, mask=mask)
    )
    projection_layer = Project(
        eq_constraint=eq_constraint,
        ineq_constraint=ineq_constraint,
        box_constraint=box_constraint,
    )
    # Compute constraint violation for each instance
    x_reshaped = x.reshape(batch_size, dim, 1)
    eq_cv = jnp.linalg.norm(A @ x_reshaped - b, axis=1, ord=jnp.inf, keepdims=True)
    ineq_cv = jnp.max(
        jnp.maximum(
            jnp.maximum(C @ x_reshaped - ub, 0), jnp.maximum(lb - C @ x_reshaped, 0)
        ),
        axis=1,
        keepdims=True,
    )
    box_cv = jnp.max(
        jnp.maximum(
            jnp.maximum(x_reshaped[:, mask, :] - ubox, 0),
            jnp.maximum(lbox - x_reshaped[:, mask, :], 0),
        ),
        axis=1,
        keepdims=True,
    )
    gt_cv = jnp.maximum(jnp.maximum(eq_cv, ineq_cv), box_cv)
    cv = projection_layer.cv(
        ProjectionInstance(x=x_reshaped, eq=EqualityConstraintsSpecification(b=b))
    )
    assert jnp.allclose(cv, gt_cv), f"Expected {gt_cv}, but got {cv}."

    # Check that the projection has zero cv
    for ii in range(batch_size):
        inp = ProjectionInstance(
            x=x[..., None], eq=EqualityConstraintsSpecification(b=b[ii : ii + 1])
        )
        x_proj = projection_layer.call(yraw=inp, n_iter=1000)[0]
        cv_proj = projection_layer.cv(
            ProjectionInstance(
                x=x_proj.x[ii : ii + 1],
                eq=EqualityConstraintsSpecification(b=b[ii : ii + 1]),
            )
        )
        assert jnp.allclose(
            cv_proj, 0.0, rtol=1e-4, atol=1e-4
        ), f"Expected 0.0, but got {cv_proj}."

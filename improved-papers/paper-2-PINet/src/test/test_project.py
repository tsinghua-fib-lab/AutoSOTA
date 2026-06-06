"""Tests for the Project class."""

from itertools import product

import cvxpy as cp
import jax
import jax.numpy as jnp
import pytest

from pinet import (
    AffineInequalityConstraint,
    EqualityConstraint,
    EqualityConstraintsSpecification,
    Project,
    ProjectionInstance,
)

jax.config.update("jax_enable_x64", True)

SEEDS = [24, 42]
BATCH_SIZE = [1, 5]


# TODO: Add another test where varA, varB are false.
@pytest.mark.parametrize("seed, batch_size", product(SEEDS, BATCH_SIZE))
def test_project_eq_ineq_varA_varb(seed, batch_size):
    dim = 100
    n_eq = 40
    n_ineq = 50
    method = "pinv"
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key, 10)
    # Generate equality constraint LHS
    A = jax.random.normal(key[0], (batch_size, n_eq, dim))
    # Generate equality constraint RHS
    b = A @ jax.random.normal(key[1], (batch_size, dim, 1))
    # Generate random point
    xinfeas = jax.random.normal(key[2], (batch_size, dim))
    # Compute projection with cvxpy
    yqp = jnp.zeros(shape=(batch_size, dim))
    for ii in range(batch_size):
        yprojcv = cp.Variable(dim)
        constraints = [A[ii, :, :] @ yprojcv == b[ii, :, 0]]
        objective = cp.Minimize(cp.sum_squares(yprojcv - xinfeas[ii, :]))
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)
        yqp = yqp.at[ii, :].set(jnp.array(yprojcv.value).reshape(dim))

    # Compute projection with Project
    eq_constraint = EqualityConstraint(A, b, method="pinv", var_b=True)
    projection_layer = Project(eq_constraint=eq_constraint)
    yprojiter = projection_layer.call(
        yraw=ProjectionInstance(
            x=xinfeas[..., None], eq=EqualityConstraintsSpecification(b=b)
        )
    )[0].x
    assert jnp.allclose(yprojiter[..., 0], yqp)

    # Generate new RHS
    b_new = A @ jax.random.normal(key[3], (batch_size, dim, 1))
    yprojiter = projection_layer.call(
        yraw=ProjectionInstance(
            x=xinfeas[..., None], eq=EqualityConstraintsSpecification(b=b_new)
        )
    )[0].x
    # New cvxpy problem
    yqp = jnp.zeros(shape=(batch_size, dim))
    for ii in range(batch_size):
        yprojcv = cp.Variable(dim)
        constraints_b_new = [A[ii, :, :] @ yprojcv == b_new[ii, :, 0]]
        objective_b_new = cp.Minimize(cp.sum_squares(yprojcv - xinfeas[ii, :]))
        problem_b_new = cp.Problem(objective_b_new, constraints_b_new)
        problem_b_new.solve(verbose=False)
        yqp = yqp.at[ii, :].set(jnp.array(yprojcv.value).reshape(dim))

    assert jnp.allclose(yprojiter[..., 0], yqp)
    # %%
    # Generate inequality constraints LHS
    C = jax.random.normal(key[4], shape=(batch_size, n_ineq, dim))
    b = jnp.zeros(shape=(batch_size, n_eq, 1))
    lb = jnp.zeros(shape=(batch_size, n_ineq, 1))
    ub = jnp.zeros(shape=(batch_size, n_ineq, 1))
    # Compute RHS by solving feasibility problem
    for ii in range(batch_size):
        xfeas = cp.Variable(dim)
        bfeas = cp.Variable(n_eq)
        lfeas = cp.Variable(n_ineq)
        ufeas = cp.Variable(n_ineq)
        constraints = [
            A[ii, :, :] @ xfeas == bfeas,
            lfeas <= C[ii, :, :] @ xfeas,
            C[ii, :, :] @ xfeas <= ufeas,
            -1 <= xfeas,
            xfeas <= 1,
        ]
        objective = cp.Minimize(jnp.ones(shape=(dim)) @ xfeas)
        problem = cp.Problem(objective=objective, constraints=constraints)
        problem.solve()

        # Extract RHS parameters
        b = b.at[ii, :, :].set(jnp.array(bfeas.value).reshape((n_eq, 1)))
        lb = lb.at[ii, :, :].set(jnp.array(lfeas.value).reshape((n_ineq, 1)))
        ub = ub.at[ii, :, :].set(jnp.array(ufeas.value).reshape((n_ineq, 1)))

    # Check projection layer without var_b
    eq_constraint = EqualityConstraint(A=A, b=b, method=method, var_b=False)
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)

    projection_layer_novarb = Project(
        eq_constraint=eq_constraint, ineq_constraint=ineq_constraint
    )

    xprojiter_novarb = projection_layer_novarb.call(
        yraw=ProjectionInstance(x=xinfeas[..., None]),
        n_iter=500,
    )[0].x
    # Check projection layer with var_b
    eq_constraint = EqualityConstraint(A=A, b=b, method=method, var_b=True)
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)

    projection_layer = Project(
        eq_constraint=eq_constraint, ineq_constraint=ineq_constraint
    )
    inp_varb = ProjectionInstance(
        x=xinfeas[..., None], eq=EqualityConstraintsSpecification(b=b)
    )
    xprojiter = projection_layer.call(yraw=inp_varb, n_iter=500)[0].x

    # Compute projections with QP
    yqp = jnp.zeros(shape=(batch_size, dim))
    for ii in range(batch_size):
        yproj = cp.Variable(dim)
        constraints = [
            A[ii, :, :] @ yproj == b[ii, :, 0],
            lb[ii, :, 0] <= C[ii, :, :] @ yproj,
            C[ii, :, :] @ yproj <= ub[ii, :, 0],
        ]
        objective = cp.Minimize(cp.sum_squares(yproj - xinfeas[ii, :]))
        problem_qp = cp.Problem(objective=objective, constraints=constraints)
        problem_qp.solve()
        yqp = yqp.at[ii, :].set(jnp.array(yproj.value).reshape(dim))

    assert jnp.allclose(xprojiter[..., 0], yqp, atol=1e-3, rtol=1e-3)
    assert jnp.allclose(xprojiter_novarb[..., 0], yqp, atol=1e-3, rtol=1e-3)
    # Test call and check method
    sigma = 1.0
    omega = 1.7
    tol = 1e-5
    check_every = 20
    max_iter = 500
    reduction = "max"
    for reduction in ["max", "mean", 0.5, 0.9]:
        check = projection_layer.call_and_check(
            sigma=sigma,
            omega=omega,
            check_every=check_every,
            tol=tol,
            max_iter=max_iter,
            reduction=reduction,
        )
        check_novarb = projection_layer_novarb.call_and_check(
            sigma=sigma,
            omega=omega,
            check_every=check_every,
            tol=tol,
            max_iter=max_iter,
            reduction=reduction,
        )
        _, flag, _ = check(
            ProjectionInstance(
                x=xinfeas[..., None], eq=EqualityConstraintsSpecification(b=b)
            )
        )
        _, flag_novarb, _ = check_novarb(ProjectionInstance(x=xinfeas[..., None]))

        assert flag
        assert flag_novarb

    # %%
    b_new = b + jax.random.normal(key[5], shape=(batch_size, n_eq, 1))
    yqp = jnp.zeros(shape=(batch_size, dim))
    for ii in range(batch_size):
        yproj = cp.Variable(dim)
        constraints = [
            A[ii, :, :] @ yproj == b_new[ii, :, 0],
            lb[ii, :, 0] <= C[ii, :, :] @ yproj,
            C[ii, :, :] @ yproj <= ub[ii, :, 0],
        ]
        objective = cp.Minimize(cp.sum_squares(yproj - xinfeas[ii, :]))
        problem_qp = cp.Problem(objective=objective, constraints=constraints)
        problem_qp.solve(verbose=False)
        yqp = yqp.at[ii, :].set(jnp.array(yproj.value).reshape(dim))

    inp_varb_new = inp_varb.update(eq=inp_varb.eq.update(b=b_new))
    xprojiter = projection_layer.call(yraw=inp_varb_new, n_iter=500)[0].x
    assert jnp.allclose(xprojiter[..., 0], yqp, atol=1e-3, rtol=1e-3)
    # %%
    # Generate new LHS and RHS
    A_new = jax.random.normal(key[6], (batch_size, n_eq, dim))
    b_new = A_new @ jax.random.normal(key[7], (batch_size, dim, 1))
    eq_constraint = EqualityConstraint(A=A_new, b=b_new, method=method, var_A=True)
    projection_layer = Project(eq_constraint=eq_constraint)
    inp = ProjectionInstance(
        x=xinfeas[..., None], eq=EqualityConstraintsSpecification(A=A_new, b=b_new)
    )
    xprojiter = projection_layer.call(yraw=inp)[0].x
    # New cvxpy problem
    yqp = jnp.zeros(shape=(batch_size, dim))
    for ii in range(batch_size):
        yprojcv = cp.Variable(dim)
        constraints_new = [A_new[ii, :, :] @ yprojcv == b_new[ii, :, 0]]
        objective_new = cp.Minimize(cp.sum_squares(yprojcv - xinfeas[ii, :]))
        problem_new = cp.Problem(objective_new, constraints_new)
        problem_new.solve(verbose=False)
        yqp = yqp.at[ii, :].set(jnp.array(yprojcv.value).reshape(dim))

    assert jnp.allclose(xprojiter[..., 0], yqp)
    # %% Solve projection with both equality and inequality
    yqp = jnp.zeros(shape=(batch_size, dim))
    for ii in range(batch_size):
        yproj = cp.Variable(dim)
        constraints = [
            A_new[ii, :, :] @ yproj == b_new[ii, :, 0],
            lb[ii, :, 0] <= C[ii, :, :] @ yproj,
            C[ii, :, :] @ yproj <= ub[ii, :, 0],
        ]
        objective = cp.Minimize(cp.sum_squares(yproj - xinfeas[ii, :]))
        problem_qp = cp.Problem(objective=objective, constraints=constraints)
        problem_qp.solve(verbose=False)
        yqp = yqp.at[ii, :].set(jnp.array(yproj.value).reshape(dim))

    eq_constraint = EqualityConstraint(A=A, b=b, method=method, var_b=True, var_A=True)
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)

    projection_layer = Project(
        eq_constraint=eq_constraint, ineq_constraint=ineq_constraint
    )
    inp = ProjectionInstance(
        x=xinfeas[..., None], eq=EqualityConstraintsSpecification(b=b_new, A=A_new)
    )
    xprojiter = projection_layer.call(yraw=inp, n_iter=500)[0].x

    assert jnp.allclose(xprojiter.reshape(yqp.shape), yqp, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("bad_reduction", ["median", 1.5, 0.0, -0.2, 2])
def test_call_and_check_invalid_reduction_raises(bad_reduction):
    # Minimal feasible setup: A x = b with b constructed from a random x0
    dim, n_eq, batch = 5, 2, 1
    key = jax.random.PRNGKey(0)
    kA, kx0, kx = jax.random.split(key, 3)

    A = jax.random.normal(kA, (batch, n_eq, dim))
    x0 = jax.random.normal(kx0, (batch, dim, 1))
    b = A @ x0

    eq = EqualityConstraint(A=A, b=b, method="pinv", var_b=False)
    layer = Project(eq_constraint=eq)

    xinfeas = jax.random.normal(kx, (batch, dim, 1))
    project_and_check = layer.call_and_check(
        reduction=bad_reduction, check_every=1, max_iter=1
    )

    with pytest.raises(ValueError, match="Invalid reduction method"):
        project_and_check(ProjectionInstance(x=xinfeas))

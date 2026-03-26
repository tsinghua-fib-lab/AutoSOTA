"""Tests for the parser and affine inequality constraints."""

from itertools import product

import cvxpy as cp
import jax
import jax.numpy as jnp
import pytest

from pinet import (
    AffineInequalityConstraint,
    BoxConstraint,
    BoxConstraintSpecification,
    ConstraintParser,
    EqualityConstraint,
    ProjectionInstance,
    build_iteration_step,
)

jax.config.update("jax_enable_x64", True)

VALID_METHODS = ["pinv"]
SEEDS = [24, 42]
BATCH_SIZE = [1, 10]


@pytest.mark.parametrize(
    "method, seed, batch_size", product(VALID_METHODS, SEEDS, BATCH_SIZE)
)
def test_simple_2d(method, seed, batch_size):
    # We consider a simple 2D polytope:
    # { x | x_1 = 0, 0<= x_1 + x_2 <= 1 }
    dim = 2
    n_ineq = 1
    key = jax.random.PRNGKey(seed)
    # Equality constraint: A @ x = b
    A = jnp.array([[[1, 0]]])
    b = jnp.zeros(shape=(1, 1, 1))
    eq_constraint = EqualityConstraint(A=A, b=b, method=method)
    # Inequality constraint: l <= C @ x <= u
    C = jnp.array([[[1, 1]]])
    lb = jnp.zeros(shape=(1, 1, 1))
    ub = jnp.ones(shape=(1, 1, 1))
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)

    # Parse constraints
    parser = ConstraintParser(
        eq_constraint=eq_constraint, ineq_constraint=ineq_constraint
    )
    (lifted_eq, lifted_box, _) = parser.parse()

    # Point to be projected
    x = jax.random.uniform(key, shape=(batch_size, dim, 1), minval=-2, maxval=2)

    # Compute the projection in closed form
    yclosed = jnp.concatenate(
        (
            jnp.zeros(shape=(batch_size, 1, 1)),
            jnp.clip(x[:, 1, :], lb, ub).reshape(batch_size, 1, 1),
        ),
        axis=1,
    )

    # Compute the projection with QP
    for ii in range(batch_size):
        ycp = cp.Variable(dim)
        constraints = [
            A[0, :, :] @ ycp == b[0, :, 0],
            lb[0, :, 0] <= C[0, :, :] @ ycp,
            C[0, :, :] @ ycp <= ub[0, :, 0],
        ]
        objective = cp.Minimize(cp.sum_squares(ycp - x[ii, :, 0]))
        problem_exact = cp.Problem(objective=objective, constraints=constraints)
        problem_exact.solve()
        # Extract true projection
        y = jnp.reshape(jnp.array(ycp.value), shape=(1, 2, 1))
        assert jnp.allclose(y, yclosed[ii, :])

    # Compute the projection with QP, but in lifted form
    # Last n_ineq variables corresponding to inequality lifting
    for ii in range(batch_size):
        yliftedcp = cp.Variable(dim + n_ineq)
        constraints_lifted = [
            lifted_eq.A[0, :, :] @ yliftedcp == lifted_eq.b[0, :, 0],
            lifted_box.lb[0, :, 0] <= yliftedcp[lifted_box.mask],
            yliftedcp[lifted_box.mask] <= lifted_box.ub[0, :, 0],
        ]
        objective_lifted = cp.Minimize(cp.sum_squares(yliftedcp[:dim] - x[ii, :, 0]))
        problem_lifted = cp.Problem(
            objective=objective_lifted, constraints=constraints_lifted
        )
        problem_lifted.solve()
        # Extract lifted projection
        ylifted = jnp.expand_dims(jnp.array(yliftedcp.value[:dim]), axis=1)
        assert jnp.allclose(ylifted, yclosed[ii, :])

    # Compute the projection with iterative
    n_iter = 200
    (iteration_step, final_step) = build_iteration_step(lifted_eq, lifted_box, dim)
    sk = ProjectionInstance(x=jnp.zeros(shape=(batch_size, dim + n_ineq, 1)))
    for ii in range(n_iter):
        sk = iteration_step(sk=sk, yraw=ProjectionInstance(x=x), sigma=0.1, omega=1.0)

    yiterated = final_step(sk).x[:, :dim, :]

    assert jnp.allclose(yclosed, yiterated, rtol=1e-6, atol=1e-6)


VALID_METHODS = ["pinv"]
SEEDS = [24, 42]
# Note that here batch_size only affects number of projected points
# The same constraints hold throughout the batch
BATCH_SIZE = [1, 10]


@pytest.mark.parametrize(
    "method, seed, batch_size", product(VALID_METHODS, SEEDS, BATCH_SIZE)
)
def test_general_eq_ineq(method, seed, batch_size):
    dim = 100
    n_eq = 50
    n_ineq = 40
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key, num=3)
    # Generate equality constraints LHS
    A = jax.random.normal(key[0], shape=(1, n_eq, dim))
    # Generate inequality constraints LHS
    C = jax.random.normal(key[1], shape=(1, n_ineq, dim))
    # Compute RHS by solving feasibility problem
    xfeas = cp.Variable(dim)
    bfeas = cp.Variable(n_eq)
    lfeas = cp.Variable(n_ineq)
    ufeas = cp.Variable(n_ineq)
    constraints = [
        A[0, :, :] @ xfeas == bfeas,
        lfeas <= C[0, :, :] @ xfeas,
        C[0, :, :] @ xfeas <= ufeas,
        -1 <= xfeas,
        xfeas <= 1,
    ]
    objective = cp.Minimize(jnp.ones(shape=(dim)) @ xfeas)
    problem = cp.Problem(objective=objective, constraints=constraints)
    problem.solve()
    # Extract RHS parameters
    b = jnp.tile(jnp.array(bfeas.value).reshape((1, n_eq, 1)), (1, 1, 1))
    lb = jnp.tile(jnp.array(lfeas.value).reshape((1, n_ineq, 1)), (1, 1, 1))
    ub = jnp.tile(jnp.array(ufeas.value).reshape((1, n_ineq, 1)), (1, 1, 1))

    eq_constraint = EqualityConstraint(A=A, b=b, method=method)
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)

    # Parse constraints
    parser = ConstraintParser(
        eq_constraint=eq_constraint, ineq_constraint=ineq_constraint
    )
    (lifted_eq, lifted_box, _) = parser.parse(method=method)
    # Point to be projected
    x = jax.random.uniform(key[2], shape=(batch_size, dim, 1), minval=-2, maxval=2)

    # Compute the projection by solving QP
    yqp = jnp.zeros(shape=(batch_size, dim, 1))
    for ii in range(batch_size):
        yproj = cp.Variable(dim)
        constraints = [
            A[0, :, :] @ yproj == b[0, :, 0],
            lb[0, :, 0] <= C[0, :, :] @ yproj,
            C[0, :, :] @ yproj <= ub[0, :, 0],
        ]
        objective = cp.Minimize(cp.sum_squares(yproj - x[ii, :, 0]))
        problem_qp = cp.Problem(objective=objective, constraints=constraints)
        problem_qp.solve()
        yqp = yqp.at[ii, :, :].set(jnp.array(yproj.value).reshape((dim, 1)))

    # Compute the projection with QP, but in lifted form
    ylifted = jnp.zeros(shape=(batch_size, dim, 1))
    for ii in range(batch_size):
        yliftedproj = cp.Variable(dim + n_ineq)
        constraints_lifted = [
            lifted_eq.A[0, :, :] @ yliftedproj == lifted_eq.b[0, :, 0],
            lifted_box.lb[0, :, 0] <= yliftedproj[lifted_box.mask],
            yliftedproj[lifted_box.mask] <= lifted_box.ub[0, :, 0],
        ]
        objective_lifted = cp.Minimize(cp.sum_squares(yliftedproj[:dim] - x[ii, :, 0]))
        problem_lifted = cp.Problem(
            objective=objective_lifted, constraints=constraints_lifted
        )
        problem_lifted.solve()
        ylifted = ylifted.at[ii, :, :].set(
            jnp.array(yliftedproj.value[:dim]).reshape((dim, 1))
        )

    assert jnp.allclose(yqp, ylifted, rtol=1e-6, atol=1e-6)

    # Compute the projection with iterative
    n_iter = 500
    (iteration_step, final_step) = build_iteration_step(lifted_eq, lifted_box, dim)
    iteration_step = jax.jit(iteration_step)
    xk = ProjectionInstance(x=jnp.zeros(shape=(batch_size, dim + n_ineq, 1)))
    for ii in range(n_iter):
        xk = iteration_step(xk, ProjectionInstance(x=x), sigma=1.0, omega=1.0)

    yiterated = final_step(xk).x[:, :dim, :]

    assert jnp.allclose(yqp, yiterated, rtol=1e-3, atol=1e-3)


VALID_METHODS = ["pinv"]
SEEDS = [42]
BATCH_SIZE_VAR = [1, 2]


@pytest.mark.parametrize(
    (
        "method, seed, batch_size_A, batch_size_C, "
        "batch_size_b, batch_size_lb, batch_size_ub, "
        "batch_size_box_lower, batch_size_box_upper, "
        "batch_size_x"
    ),
    product(
        VALID_METHODS,
        SEEDS,
        BATCH_SIZE_VAR,
        BATCH_SIZE_VAR,
        BATCH_SIZE_VAR,
        BATCH_SIZE_VAR,
        BATCH_SIZE_VAR,
        BATCH_SIZE_VAR,
        BATCH_SIZE_VAR,
        BATCH_SIZE_VAR,
    ),
)
def test_general_eq_ineq_box(
    method,
    seed,
    batch_size_A,
    batch_size_C,
    batch_size_b,
    batch_size_lb,
    batch_size_ub,
    batch_size_box_lower,
    batch_size_box_upper,
    batch_size_x,
):
    """This test considers the set:
    A @ x == b,
    l <= C @ x <= u
    lbox <= x[mask] <= ubox
    """
    if batch_size_x < max(
        batch_size_A,
        batch_size_C,
        batch_size_b,
        batch_size_lb,
        batch_size_ub,
        batch_size_box_lower,
        batch_size_box_upper,
    ):
        return

    dim = 20
    n_eq = 10
    n_ineq = 30
    n_box = 5
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key, num=4)
    # Generate equality constraints LHS
    A = jax.random.normal(key[0], shape=(batch_size_A, n_eq, dim))
    # Generate inequality constraints LHS
    C = jax.random.normal(key[1], shape=(batch_size_C, n_ineq, dim))
    # Randomly generate mask for box constraints
    indices = jnp.concatenate([jnp.ones(n_box), jnp.zeros(dim - n_box)])
    mask = jax.random.permutation(key[2], indices).astype(bool)
    # Initialize parameters per batch
    b = jnp.zeros(shape=(batch_size_b, n_eq, 1))
    lb = jnp.zeros(shape=(batch_size_lb, n_ineq, 1))
    ub = jnp.ones(shape=(batch_size_ub, n_ineq, 1))
    box_lower = jnp.zeros(shape=(batch_size_box_lower, n_box, 1))
    box_upper = jnp.zeros(shape=(batch_size_box_upper, n_box, 1))

    # Compute vector parameters of polytope and feasible point
    xfeas = cp.Variable(batch_size_x * dim)
    bfeas = cp.Variable(batch_size_b * n_eq)
    lfeas = cp.Variable(batch_size_lb * n_ineq)
    ufeas = cp.Variable(batch_size_ub * n_ineq)
    lbox = cp.Variable(batch_size_box_lower * n_box)
    ubox = cp.Variable(batch_size_box_upper * n_box)
    constraints = []
    for ii in range(batch_size_x):
        # Define indices for the current batch
        Aidx = min(ii, batch_size_A - 1)
        Cidx = min(ii, batch_size_C - 1)
        bfeasidx = min(ii, batch_size_b - 1)
        lfeasidx = min(ii, batch_size_lb - 1)
        ufeasidx = min(ii, batch_size_ub - 1)
        lboxidx = min(ii, batch_size_box_lower - 1)
        uboxidx = min(ii, batch_size_box_upper - 1)
        # Add constraints
        constraints += [
            A[Aidx, :, :] @ xfeas[ii * dim : (ii + 1) * dim]
            == bfeas[bfeasidx * n_eq : (bfeasidx + 1) * n_eq],
            lfeas[lfeasidx * n_ineq : (lfeasidx + 1) * n_ineq]
            <= C[Cidx, :, :] @ xfeas[ii * dim : (ii + 1) * dim],
            C[Cidx, :, :] @ xfeas[ii * dim : (ii + 1) * dim]
            <= ufeas[ufeasidx * n_ineq : (ufeasidx + 1) * n_ineq],
            -1 <= lbox[lboxidx * n_box : (lboxidx + 1) * n_box],
            ubox[uboxidx * n_box : (uboxidx + 1) * n_box] <= 1,
            lbox[lboxidx * n_box : (lboxidx + 1) * n_box]
            <= xfeas[ii * dim : (ii + 1) * dim][mask],
            xfeas[ii * dim : (ii + 1) * dim][mask]
            <= ubox[uboxidx * n_box : (uboxidx + 1) * n_box],
            xfeas[ii * dim : (ii + 1) * dim] <= 2,
            -2 <= xfeas[ii * dim : (ii + 1) * dim],
        ]
    objective = cp.Minimize(jnp.ones(shape=(dim * batch_size_x)) @ xfeas)
    problem = cp.Problem(objective=objective, constraints=constraints)
    problem.solve(verbose=False)

    # Extract RHS parameters
    b = jnp.array(bfeas.value).reshape((batch_size_b, n_eq, 1))
    lb = jnp.array(lfeas.value).reshape((batch_size_lb, n_ineq, 1))
    ub = jnp.array(ufeas.value).reshape((batch_size_ub, n_ineq, 1))
    box_lower = jnp.array(lbox.value).reshape((batch_size_box_lower, n_box, 1))
    box_upper = jnp.array(ubox.value).reshape((batch_size_box_upper, n_box, 1))

    eq_constraint = EqualityConstraint(A=A, b=b, method=method)
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)
    box_constraint = BoxConstraint(
        BoxConstraintSpecification(lb=box_lower, ub=box_upper, mask=mask)
    )

    # Parse constraints
    parser = ConstraintParser(
        eq_constraint=eq_constraint,
        ineq_constraint=ineq_constraint,
        box_constraint=box_constraint,
    )
    (lifted_eq, lifted_box, _) = parser.parse(method=method)

    # Point to be projected
    x = jax.random.uniform(key[3], shape=(batch_size_x, dim, 1), minval=-3, maxval=3)

    # Compute the projection by solving QP
    yqp = jnp.zeros(shape=(batch_size_x, dim, 1))
    for ii in range(batch_size_x):
        # Define indices for batch
        Aidx = min(ii, batch_size_A - 1)
        Cidx = min(ii, batch_size_C - 1)
        bfeasidx = min(ii, batch_size_b - 1)
        lfeasidx = min(ii, batch_size_lb - 1)
        ufeasidx = min(ii, batch_size_ub - 1)
        lboxidx = min(ii, batch_size_box_lower - 1)
        uboxidx = min(ii, batch_size_box_upper - 1)
        yproj = cp.Variable(dim)
        constraints = [
            A[Aidx, :, :] @ yproj == b[bfeasidx, :, 0],
            lb[lfeasidx, :, 0] <= C[Cidx, :, :] @ yproj,
            C[Cidx, :, :] @ yproj <= ub[ufeasidx, :, 0],
            box_lower[lboxidx, :, 0] <= yproj[mask],
            yproj[mask] <= box_upper[uboxidx, :, 0],
        ]
        objective = cp.Minimize(cp.sum_squares(yproj - x[ii, :, 0]))
        problem_qp = cp.Problem(objective=objective, constraints=constraints)
        problem_qp.solve(solver=cp.OSQP, eps_abs=1e-7, eps_rel=1e-7, verbose=False)
        yqp = yqp.at[ii, :, :].set(jnp.array(yproj.value).reshape((dim, 1)))

    # Compute the projection with QP, but in lifted form
    ylifted = jnp.zeros(shape=(batch_size_x, dim, 1))
    for ii in range(batch_size_x):
        # Define indices for batch
        # Should be careful here, because of the lifting
        ACidx = min(ii, max(batch_size_A - 1, batch_size_C - 1))
        bfeasidx = min(ii, batch_size_b - 1)
        loweridx = min(ii, max(batch_size_lb - 1, batch_size_box_lower - 1))
        upperidx = min(ii, max(batch_size_ub - 1, batch_size_box_upper - 1))
        yliftedproj = cp.Variable(dim + n_ineq)
        constraints_lifted = [
            lifted_eq.A[ACidx, :, :] @ yliftedproj == lifted_eq.b[bfeasidx, :, 0],
            lifted_box.lb[loweridx, :, 0] <= yliftedproj[lifted_box.mask],
            yliftedproj[lifted_box.mask] <= lifted_box.ub[upperidx, :, 0],
        ]
        objective_lifted = cp.Minimize(cp.sum_squares(yliftedproj[:dim] - x[ii, :, 0]))
        problem_lifted = cp.Problem(
            objective=objective_lifted, constraints=constraints_lifted
        )
        problem_lifted.solve(solver=cp.OSQP, eps_abs=1e-7, eps_rel=1e-7, verbose=False)
        ylifted = ylifted.at[ii, :, :].set(
            jnp.array(yliftedproj.value[:dim]).reshape((dim, 1))
        )

    assert jnp.allclose(yqp, ylifted, rtol=1e-5, atol=1e-5)

    # Compute with iterative using lifting of:
    # Equality + Inequality + Box
    n_iter = 5000
    (iteration_step, final_step) = build_iteration_step(lifted_eq, lifted_box, dim)
    iteration_step = jax.jit(iteration_step)
    xk = ProjectionInstance(x=jnp.zeros(shape=(batch_size_x, dim + n_ineq, 1)))
    for ii in range(n_iter):
        xk = iteration_step(xk, ProjectionInstance(x=x), sigma=1.0, omega=1.0)

    yiterated = final_step(xk).x[:, :dim, :]

    assert jnp.allclose(yqp, yiterated, rtol=1e-3, atol=1e-3)
    # Compute with iterative using lifting of:
    # Equality + Inequality
    # Write box constraints as affine inequality constraints
    Caug = jnp.concatenate(
        (C, jnp.tile(jnp.eye(dim)[mask, :].reshape(1, n_box, dim), (C.shape[0], 1, 1))),
        axis=1,
    )
    # Adapt lower and upper bounds accordingly
    # Maximum batch size of lower and upper bound
    mblb = max(lb.shape[0], box_lower.shape[0])
    mbub = max(ub.shape[0], box_upper.shape[0])
    lbaug = jnp.concatenate(
        (
            jnp.tile(lb, (mblb // lb.shape[0], 1, 1)),
            jnp.tile(box_lower, (mblb // box_lower.shape[0], 1, 1)),
        ),
        axis=1,
    )
    ubaug = jnp.concatenate(
        (
            jnp.tile(ub, (mbub // ub.shape[0], 1, 1)),
            jnp.tile(box_upper, (mbub // box_upper.shape[0], 1, 1)),
        ),
        axis=1,
    )
    n_ineq_aug = n_ineq + n_box
    ineq_constraint_aug = AffineInequalityConstraint(C=Caug, lb=lbaug, ub=ubaug)

    parser_aug = ConstraintParser(
        eq_constraint=eq_constraint, ineq_constraint=ineq_constraint_aug
    )

    (lifted_eq, lifted_box, _) = parser_aug.parse()

    n_iter = 5000
    (iteration_step, final_step) = build_iteration_step(lifted_eq, lifted_box, dim)
    iteration_step = jax.jit(iteration_step)
    xk = ProjectionInstance(x=jnp.zeros(shape=(batch_size_x, dim + n_ineq_aug, 1)))
    for ii in range(n_iter):
        xk = iteration_step(xk, ProjectionInstance(x=x), sigma=1.0, omega=1.0)

    yiterated = final_step(xk).x[:, :dim, :]

    assert jnp.allclose(yqp, yiterated, rtol=1e-3, atol=1e-3)


SEEDS = [24, 42]
BATCH_SIZE = [1, 10]


@pytest.mark.parametrize("seed, batch_size", product(SEEDS, BATCH_SIZE))
def test_simple_no_equality(seed, batch_size):
    # We consider a simple 2D polytope:
    # { x | 0 <= x_1 + x_2 <= 1 }
    dim = 2
    n_ineq = 1
    key = jax.random.PRNGKey(seed)
    # Inequality constraint: l <= C @ x <= u
    C = jnp.array([[[1, 1]]])
    lb = jnp.zeros(shape=(1, 1, 1))
    ub = jnp.ones(shape=(1, 1, 1))
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)

    # Parse constraints
    parser = ConstraintParser(eq_constraint=None, ineq_constraint=ineq_constraint)
    (lifted_eq, lifted_box, _) = parser.parse()

    # Point to be projected
    x = jax.random.uniform(key, shape=(batch_size, dim, 1), minval=-2, maxval=2)

    # Compute the projection with iterative
    (lifted_eq, lifted_box, _) = parser.parse()

    n_iter = 500
    (iteration_step, final_step) = build_iteration_step(lifted_eq, lifted_box, dim)
    xk = ProjectionInstance(x=jnp.zeros(shape=(batch_size, dim + n_ineq, 1)))
    for ii in range(n_iter):
        xk = iteration_step(xk, ProjectionInstance(x=x), sigma=0.1, omega=1.0)

    yiterated = final_step(xk).x[:, :dim, :]

    # Compute the projection with QP
    for ii in range(batch_size):
        ycp = cp.Variable(dim)
        constraints = [
            lb[0, :, 0] <= C[0, :, :] @ ycp,
            C[0, :, :] @ ycp <= ub[0, :, 0],
        ]
        objective = cp.Minimize(cp.sum_squares(ycp - x[ii, :, 0]))
        problem_exact = cp.Problem(objective=objective, constraints=constraints)
        problem_exact.solve()
        # Extract true projection
        y_qp = jnp.reshape(jnp.array(ycp.value), shape=(1, 2, 1))

        # Compute the projection with QP, but in lifted form
        # Last n_ineq variables corresponding to inequality lifting
        yliftedcp = cp.Variable(dim + n_ineq)
        constraints_lifted = [
            lifted_eq.A[0, :, :] @ yliftedcp == lifted_eq.b[0, :, 0],
            lifted_box.lb[0, :, 0] <= yliftedcp[lifted_box.mask],
            yliftedcp[lifted_box.mask] <= lifted_box.ub[0, :, 0],
        ]
        objective_lifted = cp.Minimize(cp.sum_squares(yliftedcp[:dim] - x[ii, :, 0]))
        problem_lifted = cp.Problem(
            objective=objective_lifted, constraints=constraints_lifted
        )
        problem_lifted.solve()
        # Extract lifted projection
        ylifted = jnp.expand_dims(jnp.array(yliftedcp.value[:dim]), axis=1)

        # Check the projections match
        assert jnp.allclose(ylifted, y_qp[0, :, :])
        assert jnp.allclose(y_qp[0, :, :], yiterated[ii, :, :], rtol=1e-6, atol=1e-6)


def test_affine_inequality_project_cannot_be_called_directly():
    """Test that the project method cannot be called directly."""
    C = jnp.array([[[1, 1]]])
    lb = jnp.zeros(shape=(1, 1, 1))
    ub = jnp.ones(shape=(1, 1, 1))
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)

    with pytest.raises(
        NotImplementedError,
        match="The 'project' method is not implemented and should not be called.",
    ):
        ineq_constraint.project(ProjectionInstance(x=jnp.zeros((1, 2, 1))))


def test_constraint_parser_no_ineq_no_box_returns_eq_as_is():
    dim, n_eq = 3, 2
    A = jnp.arange(n_eq * dim, dtype=jnp.float64).reshape(1, n_eq, dim)
    b = jnp.zeros((1, n_eq, 1))
    eq = EqualityConstraint(A=A, b=b, method="pinv")

    parser = ConstraintParser(
        eq_constraint=eq, ineq_constraint=None, box_constraint=None
    )
    eq_out, box_out, _ = parser.parse(method="pinv")

    # Still the same exact object (no lifting performed)
    assert eq_out is eq
    assert box_out is None
    assert eq_out.A is A
    assert eq_out.b is b


def test_constraint_parser_no_ineq_with_box_returns_inputs():
    dim, n_eq = 4, 1
    A = jnp.ones((1, n_eq, dim))
    b = jnp.zeros((1, n_eq, 1))
    eq = EqualityConstraint(A=A, b=b, method="pinv")

    mask = jnp.array([True, False, True, False])
    n_box = int(mask.sum())
    lb = jnp.array([[[-1.0], [0.0]]]).reshape(1, n_box, 1)
    ub = jnp.array([[[1.0], [2.0]]]).reshape(1, n_box, 1)
    box = BoxConstraint(BoxConstraintSpecification(lb=lb, ub=ub, mask=mask))

    parser = ConstraintParser(
        eq_constraint=eq, ineq_constraint=None, box_constraint=box
    )
    eq_out, box_out, _ = parser.parse(method="pinv")

    # Still the same exact objects (no lifting performed)
    assert eq_out is eq
    assert box_out is box

    # Sanity: mask/bounds unchanged
    assert jnp.array_equal(box_out.mask, mask)
    assert jnp.array_equal(box_out.lb, lb)
    assert jnp.array_equal(box_out.ub, ub)

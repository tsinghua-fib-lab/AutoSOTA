"""Test projection layer and vjp, with equilibration."""

from itertools import product

import cvxpy as cp
import jax
import pytest
from jax import numpy as jnp

from pinet import (
    AffineInequalityConstraint,
    BoxConstraint,
    BoxConstraintSpecification,
    EqualityConstraint,
    EqualityConstraintsSpecification,
    EquilibrationParams,
    Project,
    ProjectionInstance,
    ruiz_equilibration,
)

jax.config.update("jax_enable_x64", True)

SEEDS = [24, 42]
BATCH_SIZE = [1, 5]


@pytest.mark.parametrize("seed, batch_size", product(SEEDS, BATCH_SIZE))
def test_general_eq_ineq(seed, batch_size):
    method = "pinv"
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
        ]
    objective = cp.Minimize(jnp.ones(shape=(dim * batch_size)) @ xfeas)
    problem = cp.Problem(objective=objective, constraints=constraints)
    problem.solve(verbose=False)
    # Extract RHS parameters
    b = jnp.tile(jnp.array(bfeas.value).reshape((batch_size, n_eq, 1)), (1, 1, 1))
    lb = jnp.tile(jnp.array(lfeas.value).reshape((1, n_ineq, 1)), (1, 1, 1))
    ub = jnp.tile(jnp.array(ufeas.value).reshape((1, n_ineq, 1)), (1, 1, 1))
    lbox = jnp.tile(jnp.array(lboxfeas.value).reshape((1, n_box, 1)), (1, 1, 1))
    ubox = jnp.tile(jnp.array(uboxfeas.value).reshape((1, n_box, 1)), (1, 1, 1))
    # Define projection layer ingredients
    for var_b in [False, True]:
        eq_constraint = EqualityConstraint(A=A, b=b, method=method, var_b=var_b)
        ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)
        box_constraint = BoxConstraint(
            BoxConstraintSpecification(lb=lbox, ub=ubox, mask=mask)
        )

        # Hyperparameters
        sigma = 0.1
        omega = 1.7
        sigma_equil = 0.01

        # Projection layer with unrolling no equilibration
        pl_unroll = Project(
            eq_constraint=eq_constraint,
            ineq_constraint=ineq_constraint,
            box_constraint=box_constraint,
            unroll=True,
        )

        # Projection layer with unrolling plus equilibration
        equilibration_params = EquilibrationParams(
            max_iter=25,
            tol=1e-3,
            ord=2.0,
            col_scaling=False,
            update_mode="Gauss",
            safeguard=False,
        )
        pl_unroll_equil = Project(
            eq_constraint=eq_constraint,
            ineq_constraint=ineq_constraint,
            box_constraint=box_constraint,
            unroll=True,
            equilibration_params=equilibration_params,
        )

        # Projection layer with implicit differentiation
        pl_impl_equil = Project(
            eq_constraint=eq_constraint,
            ineq_constraint=ineq_constraint,
            box_constraint=box_constraint,
            unroll=False,
            equilibration_params=equilibration_params,
        )
        # Point to be projected
        x = jax.random.uniform(key[3], shape=(batch_size, dim), minval=-2, maxval=2)

        # Compute the projection by solving QP
        yqp = jnp.zeros(shape=(batch_size, dim))
        for ii in range(batch_size):
            yproj = cp.Variable(dim)
            constraints = [
                A[0, :, :] @ yproj == b[ii, :, 0],
                lb[0, :, 0] <= C[0, :, :] @ yproj,
                C[0, :, :] @ yproj <= ub[0, :, 0],
                lbox[0, :, 0] <= yproj[mask],
                yproj[mask] <= ubox[0, :, 0],
            ]
            objective = cp.Minimize(cp.sum_squares(yproj - x[ii, :]))
            problem_qp = cp.Problem(objective=objective, constraints=constraints)
            problem_qp.solve()
            yqp = yqp.at[ii, :].set(jnp.array(yproj.value).reshape((dim)))

        # Check that the projection is computed correctly
        n_iter = 1000
        inp = ProjectionInstance(
            x=x[..., None], eq=EqualityConstraintsSpecification(b=b) if var_b else None
        )
        y_unroll = pl_unroll.call(yraw=inp, n_iter=n_iter, sigma=sigma, omega=omega)[0]
        y_impl = pl_unroll_equil.call(
            yraw=inp,
            n_iter=n_iter,
            sigma=sigma_equil,
            omega=omega,
        )[0]
        y_impl_equil = pl_impl_equil.call(
            yraw=inp,
            n_iter=n_iter,
            sigma=sigma_equil,
            omega=omega,
        )[0]
        assert jnp.allclose(y_unroll.x[..., 0], yqp, atol=1e-4, rtol=1e-4)
        assert jnp.allclose(y_impl.x[..., 0], yqp, atol=1e-4, rtol=1e-4)
        assert jnp.allclose(y_impl_equil.x[..., 0], yqp, atol=1e-4, rtol=1e-4)

        # Check that the VJP is computed correctly
        # Compare with loop unrolling
        # Simple "loss" function as inner product
        n_iter = 1000
        vec = jnp.array(jax.random.normal(key[4], shape=(dim, batch_size)))

        def loss(x, v, mode, n_iter_bwd, fpi):
            inp = ProjectionInstance(
                x=x[..., None],
                eq=EqualityConstraintsSpecification(b=b) if var_b else None,
            )
            if mode == "unroll":
                return (
                    pl_unroll.call(
                        yraw=inp,
                        n_iter=n_iter,
                        sigma=sigma,
                        omega=omega,
                    )[
                        0
                    ].x[..., 0]
                    @ v
                ).mean()
            elif mode == "unroll_equil":
                return (
                    pl_unroll_equil.call(
                        yraw=inp,
                        n_iter=n_iter,
                        sigma=sigma_equil,
                        omega=omega,
                    )[0].x[..., 0]
                    @ v
                ).mean()
            elif mode == "impl_equil":
                return (
                    pl_impl_equil.call(
                        yraw=inp,
                        n_iter=n_iter,
                        sigma=sigma_equil,
                        omega=omega,
                        n_iter_bwd=n_iter_bwd,
                        fpi=fpi,
                    )[0].x[..., 0]
                    @ v
                ).mean()

        grad_unroll = jax.grad(loss, argnums=0)(
            x, vec, mode="unroll", n_iter_bwd=-1, fpi=True
        )
        grad_unroll_equil = jax.grad(loss, argnums=0)(
            x, vec, mode="unroll_equil", n_iter_bwd=-1, fpi=True
        )
        grad_impl_equil = jax.grad(loss, argnums=0)(
            x, vec, mode="impl_equil", n_iter_bwd=100, fpi=False
        )

        assert jnp.allclose(grad_unroll, grad_unroll_equil, atol=1e-4, rtol=1e-4)
        assert jnp.allclose(grad_unroll, grad_impl_equil, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("update_mode", ["Gauss", "Jacobi"])
def test_row_scaling_balances_rows(update_mode):
    A = jnp.array(
        [
            [100.0, 0.0, 0.0],
            [0.1, 1.0, 0.0],
            [0.0, 0.0, 20.0],
            [0.0, 5.0, 0.0],
        ]
    )
    params = EquilibrationParams(
        max_iter=50,
        tol=1e-6,
        ord=2.0,
        col_scaling=False,
        update_mode=update_mode,
        safeguard=False,
    )

    def _row_ratio(A, ord_val=2.0):
        rn = jnp.linalg.norm(A, axis=1, ord=ord_val)
        return (rn.max() / rn.min()).item()

    ratio_before = _row_ratio(A, 2.0)
    scaled, d_r, d_c = ruiz_equilibration(A, params)
    ratio_after = _row_ratio(scaled, 2.0)

    assert ratio_after < ratio_before
    assert ratio_after < 1.16
    # scaled = diag(d_r) A diag(d_c)
    recon = (A * d_r[:, None]) * d_c[None, :]
    assert jnp.allclose(scaled, recon, atol=1e-12)


@pytest.mark.parametrize("update_mode", ["Gauss", "Jacobi"])
@pytest.mark.parametrize("col_scaling", [False, True])
def test_one_step_via_max_iter_equals_high_tol(update_mode, col_scaling):
    A = jnp.array([[1.0, 2.0, -1.0], [0.0, -4.0, 5.0]])
    # Force exactly one step in two different ways
    p_one = EquilibrationParams(
        max_iter=1,
        tol=0.0,
        ord=2.0,
        col_scaling=col_scaling,
        update_mode=update_mode,
        safeguard=False,
    )
    p_tol = EquilibrationParams(
        max_iter=50,
        tol=1e9,
        ord=2.0,
        col_scaling=col_scaling,
        update_mode=update_mode,
        safeguard=False,
    )
    s1, dr1, dc1 = ruiz_equilibration(A, p_one)
    s2, dr2, dc2 = ruiz_equilibration(A, p_tol)

    assert jnp.allclose(s1, s2, atol=1e-12, rtol=0.0)
    assert jnp.allclose(dr1, dr2, atol=1e-12, rtol=0.0)
    assert jnp.allclose(dc1, dc2, atol=1e-12, rtol=0.0)


def test_safeguard_when_condition_worsens_triggers_identity_scalings():
    # This matrix yields a worse condition number after one step
    A = jnp.array(
        [
            [-1.47236611, -0.33950648, -0.81108737],
            [0.93786103, 0.49052747, 1.40301434],
        ]
    )
    p1 = EquilibrationParams(
        max_iter=1,
        tol=0.0,
        ord=2.0,
        col_scaling=False,
        update_mode="Gauss",
        safeguard=False,
    )
    p2 = EquilibrationParams(
        max_iter=1,
        tol=0.0,
        ord=2.0,
        col_scaling=False,
        update_mode="Gauss",
        safeguard=True,
    )

    cond_before = jnp.linalg.cond(A).item()
    s_no, dr_no, dc_no = ruiz_equilibration(A, p1)
    cond_after_no = jnp.linalg.cond(s_no).item()

    # Sanity: this is the branch where safeguard should matter
    assert cond_after_no > cond_before

    s_yes, dr_yes, dc_yes = ruiz_equilibration(A, p2)
    cond_after_yes = jnp.linalg.cond(s_yes).item()

    # Guard must not return something worse than original
    assert cond_after_yes <= cond_before + 1e-12
    # Guarded scalings should be identity
    assert jnp.allclose(dr_yes, jnp.ones(A.shape[0]))
    assert jnp.allclose(dc_yes, jnp.ones(A.shape[1]))

"""Test implementations for the (solved) proposal distributions.

For the regularized optimization problem, conftest gives us:
1. Functions to maximize.
2. Priors to regularize to.

The test-cases here check if the solution to the constrained program:
1. Normalize to 1.
2. All bins are positively valued.
3. Satisfy the epsilon hard-constraint if specified.
4. Recheck 1-3 for bounded rescaling of the values.
5. Robustness to input arguments.

Note that the choices for hyperparameters in the fixtures also imply
that these are values that we found to work. Anything outside that does
not guarantee stability outside the scope of these tests.
"""
from __future__ import annotations
from typing import Type, Callable, Any
from functools import partial

import pytest

import jax
import jax.numpy as jnp

from smz.planning import proposals

from utils import to_pmf, min_max


def _check_positivity(
    q_star: jax.Array,
    param: jax.Array,
    param_name: str,
    prop_name: str,
    f_name: str,
    pi_name: str,
):
    signs = jnp.sign(q_star + 1e-16).sum(axis=-1)  # shape: len(inv_beta), num
    mask = jnp.isclose(signs, q_star.shape[-1])

    assert mask.all(), (
        f"Non-Positive PMF found for: {prop_name} on values for "
        f"{param_name}: {param.at[jnp.where(~mask)[0]].get()}. "
        f"Values: {q_star.at[jnp.where(~mask)[0]].get()}. "
        f"Prior: {pi_name} - Function: {f_name}."
    )


def _check_normalizes(
    q_star: jax.Array,
    param: jax.Array,
    param_name: str,
    prop_name: str,
    f_name: str,
    pi_name: str,
    pytestconfig,
):
    zs = q_star.sum(axis=-1)  # shape: len(inv_beta), num

    tolerance = pytestconfig.getoption("norm_tol")
    mask = jnp.isclose(zs, 1.0, atol=tolerance)

    assert mask.all(), (
        f"Normalization failed for: {prop_name} on values for "
        f"{param_name}: {param.at[jnp.where(~mask)[0]].get()}. "
        f"Normalization error: {(zs - 1.0).at[jnp.where(~mask)[0]].get()}. "
        f"Prior: {pi_name} - Function: {f_name}."
    )


def _check_trust_region(
    divs: jax.Array,
    epsilon: jax.Array,
    prop_name: str,
    f_name: str,
    pi_name: str,
    pytestconfig,
):
    tr_tolerance = pytestconfig.getoption("trust_region_tol")

    if pytestconfig.getoption("trust_region_strictness") == "interior":
        mask = divs <= (epsilon + tr_tolerance)  # shape: len(epsilon), num

        illegals = (divs - epsilon).at[jnp.where(~mask)[0]].get()
        assert mask.all(), (
            f"Trust-Region not satisfied for: {prop_name} on values "
            f"for epsilon: {epsilon.at[jnp.where(~mask)[0]].get()}. "
            f"Trust-Region exceeded by: {illegals}. "
            f"Prior: {pi_name} - Function: {f_name}."
        )
    else:
        mask = jnp.abs(divs - epsilon) <= tr_tolerance

        illegals = (divs - epsilon).at[jnp.where(~mask)[0]].get()
        assert mask.all(), (
            f"Not on Vertex error! Trust-Region for: {prop_name} on values "
            f"for epsilon: {epsilon.at[jnp.where(~mask)[0]].get()}. "
            f"Trust-Region exceeded by: {illegals}. "
            f"Prior: {pi_name} - Function: {f_name}."
        )


def _check_monotonicity(
        divs: jax.Array,
        epsilon: jax.Array,
        prop_name: str, f_name: str, pi_name: str
):
    mask = jnp.append(-1.0, divs) <= jnp.append(divs, jnp.inf)

    illegals = (divs - epsilon).at[jnp.where(~mask)[0]].get()
    assert mask.all(), (
        f"Non-Monotonicity error! Encountered decreasing value for the "
        f"Trust-Region for: {prop_name} on values "
        f"for epsilon: {epsilon.at[jnp.where(~mask)[0]].get()}. "
        f"Trust-Region exceeded by: {illegals}. "
        f"Prior: {pi_name} - Function: {f_name}."
    )


def test_input_ambiguity():
    """Check if all proposal distributions correctly handle input validation.

    An error should be raised if epsilon Vs. inv_beta is ambiguous.
    """  # TODO; API robustness.
    pass


@pytest.mark.parametrize(
    "prior",
    ["gaussian", "uniform", "mixture"],
    indirect=True,
    ids=lambda v: f"prior={v}",
)
@pytest.mark.parametrize(
    "function",
    ["sine", "uniform", "uniform_but_one", "step"],
    indirect=True,
    ids=lambda v: f"function={v}",
)
@pytest.mark.parametrize(
    "proposal",
    [
        # Numerically normalizable
        (
            proposals.JensenShannon,
            {"num_init": 8, "bounds": (-3.0, 10.0), "recursive_steps": 5},
        ),
        (
            proposals.Jeffrey,
            {"num_init": 5, "bounds": (-5.0, 10.0), "recursive_steps": 5},
        ),
        (
            proposals.SquaredHellinger,
            {"num_init": 5, "bounds": (-20.0, 10.0), "recursive_steps": 5},
        ),
        (
            proposals.ExPropKullbackLeibler,
            {"num_init": 5, "bounds": (-20.0, 10.0), "recursive_steps": 5},
        ),
        # Analytically normalizable
        (proposals.VariationalKullbackLeibler, {}),
        (proposals.TotalVariationL2, {}),
        (proposals.Greedy, {"do_epsilon_greedy": True}),
        (proposals.Greedy, {"do_epsilon_greedy": False}),
        (proposals.Uniform, {}),
    ],
    ids=lambda v: f"proposal={v[0].__name__}",
)
@pytest.mark.parametrize(
    "start, stop, num", [(-1, 1, 3), (0, 10, 100), (1, 5, 50), (-10, 10, 200)]
)
@pytest.mark.parametrize(
    "inv_beta_bounds", [{"start": -20, "stop": 10}],
)
def test_normalization(
    prior: tuple[str, Callable[[jax.Array], jax.Array]],
    function: tuple[str, Callable[[jax.Array], jax.Array]],
    proposal: tuple[Type[proposals.PolicyObjective], dict[str, Any]],
    start: int | float,
    stop: int | float,
    num: int,
    inv_beta_bounds: dict[str, jax.typing.ArrayLike],
    pytestconfig,
):
    """Tests for a regularized policy-objective whether the solution sums to 1
    """
    pi_name, pi_f = prior
    f_name, f = function

    # Compute input, outputs, and prior.
    xs = jnp.linspace(start, stop, num)
    ys = min_max(f, shift=-1.0)(xs)
    pi = to_pmf(pi_f)(xs)

    # Instantiate the objective.
    proposal_type, init_kwargs = proposal
    proposal_obj = proposal_type(**init_kwargs)  # type: ignore

    # Specify range of lagrange-multipliers for the divergence-constraint.
    resolution = pytestconfig.getoption("multiplier_resolution")
    inv_beta = jnp.logspace(**inv_beta_bounds, num=resolution, base=jnp.e)

    # Compute solutions to the regularized policy optimization problem.
    q_star = jax.vmap(partial(proposal_obj, q=ys, pi=pi))(inv_beta=inv_beta)

    _check_positivity(
        q_star, inv_beta, "inv_beta", proposal_type.__name__, f_name, pi_name
    )

    _check_normalizes(
        q_star,
        inv_beta,
        "inv_beta",
        proposal_type.__name__,
        f_name,
        pi_name,
        pytestconfig,
    )


@pytest.mark.parametrize(
    "prior",
    ["gaussian", "uniform", "mixture"],
    indirect=True,
    ids=lambda v: f"prior={v}",
)
@pytest.mark.parametrize(
    "function",
    ["sine", "uniform", "uniform_but_one", "step"],
    indirect=True,
    ids=lambda v: f"function={v}",
)
@pytest.mark.parametrize(
    "proposal",
    [
        # Numerically solvable trust-region
        (
            proposals.JensenShannon,
            {"num_init": 16, "num_init_tr": 16,
             "recursive_steps": 8, "recursive_steps_tr": 10,
             "bounds": (-3.0, 10.0),
             },
        ),
        (
            proposals.Jeffrey,
            {"num_init": 8, "num_init_tr": 16,
             "recursive_steps": 8, "recursive_steps_tr": 10,
             "bounds": (-5.0, 10.0)
             },
        ),
        (
            proposals.VariationalKullbackLeibler,
            {"num_init": 10, "bounds": (-20.0, 10.0), "recursive_steps": 5},
        ),
        (
            proposals.TotalVariationL2,
            {"num_init": 10, "bounds": (-20.0, 10.0), "recursive_steps": 10},
        ),
        # Analytically solvable trust-region
        (
            proposals.SquaredHellinger,
            {"num_init": 10, "bounds": (-20.0, 10.0), "recursive_steps": 5},
        ),
        (
            proposals.ExPropKullbackLeibler,
            {"num_init": 10, "bounds": (-20.0, 10.0), "recursive_steps": 5},
        ),
    ],
    ids=lambda v: f"proposal={v[0].__name__}",
)
@pytest.mark.parametrize(
    "start, stop, num", [(-1, 1, 3), (0, 10, 100), (1, 5, 50), (-10, 10, 200)]
)
@pytest.mark.parametrize(
    "epsilon_config",
    [
        {"start": 0, "stop": 1, "multiplier": 1.5},
    ],
)
def test_constrained(
    prior: tuple[str, Callable[[jax.Array], jax.Array]],
    function: tuple[str, Callable[[jax.Array], jax.Array]],
    proposal: tuple[Type[proposals.PolicyObjective], dict[str, Any]],
    start: int | float,
    stop: int | float,
    num: int,
    epsilon_config: dict[str, jax.typing.ArrayLike],
    pytestconfig,
):
    """Tests the hard-constrained policy-objective for a valid solution.

    Check:
     1) does q_star normalize to 1.
     2) does q_star satisfy its trust-region constraint?

    We have to re-test for proper normalization as done in `test_normalization`
    since the hard-constraint can induce different optimization dynamics.
    For example, some objectives need to numerically solve for two
    multipliers (the trust-region multiplier and the normalizer) if a
    hard-constraint is imposed, and some objectives can analytically determine
    the optimal trust-region multiplier.

    Note that each proposal objective (each divergence) has its own specified
    range for the epsilon parameter. Some divergences are bounded and behave
    significantly differently for different scales of epsilon. Ultimately,
    for all objectives we have that the solutions are sandwiched between
    the Greedy and the Prior policies.
    """
    pi_name, pi_f = prior
    f_name, f = function

    # Compute input, outputs, and prior.
    xs = jnp.linspace(start, stop, num)
    ys = min_max(f, shift=-1.0)(xs)
    pi = to_pmf(pi_f)(xs)

    # Instantiate the objective.
    proposal_type, init_kwargs = proposal
    proposal_obj = proposal_type(**init_kwargs)  # type: ignore
    proposal_obj._greedy_jitter = 0.0

    # Specify range for the trust-region constraints.
    resolution = pytestconfig.getoption("multiplier_resolution")
    factors = jnp.linspace(
        epsilon_config["start"], epsilon_config["stop"], num=resolution
    )
    ub = proposal_obj.trust_region_upperbound(ys, pi)
    epsilon = factors * ub * epsilon_config["multiplier"]

    # Compute solutions to the regularized policy optimization problem.
    q_star = jax.vmap(partial(proposal_obj, q=ys, pi=pi))(epsilon=epsilon)

    _check_positivity(
        q_star, epsilon, "epsilon", proposal_type.__name__, f_name, pi_name
    )

    _check_normalizes(
        q_star,
        epsilon,
        "epsilon",
        proposal_type.__name__,
        f_name,
        pi_name,
        pytestconfig,
    )

    # Check if trust-region is satisfied.
    divs = jax.vmap(proposal_obj.divergence, in_axes=(0, None))(q_star, pi)
    divs = jnp.clip(divs, 0, jnp.clip(ub - proposal_obj._epsilon_rtol, 0))

    _check_trust_region(
        divs, epsilon, proposal_type.__name__, f_name, pi_name, pytestconfig
    )

    _check_monotonicity(divs, epsilon, proposal_type.__name__, f_name, pi_name)

import numpy as np
import math
import pytest

from likelihood.preference_likelihood_threeway import (
    bt_threeway_hier,
    log_expert_likelihood_bt_threeway,
)


@pytest.mark.parametrize("W,i,j,y", [
    # Strong asymmetric i->j
    (np.array([[0, 1.0], [0, 0]]), 0, 1, 0),
    # Strong asymmetric j->i
    (np.array([[0, 0], [1.0, 0]]), 0, 1, 1),
    # Weak asymmetric edges -> mostly none
    (np.array([[0, 0.01], [0, 0]]), 0, 1, 2),
    # Symmetric strong edges -> ambiguous direction
    (np.array([[0, 1.0], [1.0, 0]]), 0, 1, 0),
    (np.array([[0, 1.0], [1.0, 0]]), 0, 1, 1),
    # Symmetric weak edges -> none
    (np.array([[0, 0.01], [0.01, 0]]), 0, 1, 2),
])
def test_log_likelihood_matches_forward(W, i, j, y):
    """
    Check consistency: log p(y|W) must equal log(bt_threeway_hier(W)[y]).
    """
    probs = bt_threeway_hier(W, i, j, tau=0.1, beta_edge=5.0, beta_dir=5.0)
    logp = log_expert_likelihood_bt_threeway(
        y, W, i, j,
        tau=0.1, beta_edge=5.0, beta_dir=5.0
    )

    # Probability must be > 0
    assert probs[y] > 0.0, "Probability must be strictly positive."

    # Log-likelihood must match log(prob)
    assert math.isclose(
        logp, math.log(probs[y]), rel_tol=1e-12, abs_tol=1e-12
    )


def test_random_weights_consistency():
    """
    Test the identity log(p(y)) == log(bt_threeway_hier(...)[y]) on random Ws.
    """
    rng = np.random.default_rng(123)

    for _ in range(20):
        W = rng.normal(0, 0.2, size=(3, 3))
        np.fill_diagonal(W, 0)

        i, j = 0, 1
        probs = bt_threeway_hier(W, i, j)
        assert np.all(probs > 0), "All probabilities must be positive."

        for y in (0, 1, 2):
            logp = log_expert_likelihood_bt_threeway(y, W, i, j)
            assert math.isclose(logp, math.log(probs[y]), rel_tol=1e-12)


def test_invalid_label():
    W = np.array([[0,1],[0,0]])
    with pytest.raises(ValueError):
        log_expert_likelihood_bt_threeway(99, W, 0, 1)
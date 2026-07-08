import numpy as np
from likelihood.preference_likelihood_threeway import bt_threeway_hier  # adjust import to your code


# ----------------------------------------------------------------------
# HELPER
# ----------------------------------------------------------------------

def approx(x, y, tol=1e-3):
    return abs(x - y) < tol


# ----------------------------------------------------------------------
# TESTS
# ----------------------------------------------------------------------

def test_no_edge():
    """
    Both weights zero -> absolute evidence a << 0 -> p_edge ~ 0 -> none dominates.
    """
    W = np.array([[0.0, 0.0],
                  [0.0, 0.0]])
    p = bt_threeway_hier(W, 0, 1, beta_edge=5, beta_dir=5, tau=0.1)
    assert p[2] > 0.99  # none
    assert p[0] < 1e-3  # i->j
    assert p[1] < 1e-3  # j->i


def test_weak_but_asymmetric_edge_behaves_like_none():
    """
    |W| = 0.01, tau=0.1 -> eff=0.1 -> s ~ -2.3.
    a = -2.3 -> p_edge ~ sigma(-2.3*beta_edge) small.
    None should dominate even though asymmetry is large.
    """
    W = np.array([[0.0, 0.01],
                  [0.0, 0.0]])
    p = bt_threeway_hier(W, 0, 1, beta_edge=3, beta_dir=5, tau=0.1)

    assert p[2] > 0.95  # none
    assert p[0] < 0.05  # i->j should NOT dominate
    assert p[1] < 0.01  # j->i impossible


def test_strong_i_to_j():
    """
    W[i,j] = 1.0 -> eff=10 -> s=log(10)=2.3 -> strong positive.
    p_edge ~ 1. p_ij|edge ~ 1. So p_ij ~ 1.
    """
    W = np.array([[0.0, 1.0],
                  [0.0, 0.0]])
    p = bt_threeway_hier(W, 0, 1, beta_edge=3, beta_dir=5, tau=0.1)

    assert p[0] > 0.99  # i->j
    assert p[1] < 1e-3
    assert p[2] < 1e-3


def test_strong_j_to_i():
    """
    Mirror of strong_i_to_j.
    """
    W = np.array([[0.0, 0.0],
                  [1.0, 0.0]])
    p = bt_threeway_hier(W, 0, 1, beta_edge=3, beta_dir=5, tau=0.1)

    assert p[1] > 0.99  # j->i
    assert p[0] < 1e-3
    assert p[2] < 1e-3


def test_strong_but_symmetric_edges():
    """
    Both W=1.0 -> both s=2.3.
    a=2.3 -> p_edge ~ 1.
    d=0 -> p_ij|edge = 0.5.
    So p_ij~0.5, p_ji~0.5, none~0.
    """
    W = np.array([[0.0, 1.0],
                  [1.0, 0.0]])
    p = bt_threeway_hier(W, 0, 1, beta_edge=3, beta_dir=5, tau=0.1)

    assert p[2] < 1e-3  # none almost zero
    assert approx(p[0], 0.5, tol=0.05)
    assert approx(p[1], 0.5, tol=0.05)


def test_symmetric_weak_edges_behave_like_none():
    """
    W=0.01 both ways -> eff=0.1 -> s~ -2.3.
    a = -2.3 -> p_edge tiny -> none dominates.
    """
    W = np.array([[0.0, 0.01],
                  [0.01, 0.0]])
    p = bt_threeway_hier(W, 0, 1, beta_edge=3, beta_dir=5, tau=0.1)

    assert p[2] > 0.95  # none
    assert p[0] < 0.05
    assert p[1] < 0.05


def test_mixed_strength_edges():
    """
    W[i,j]=1.0 strong; W[j,i]=0.05 weak.
    Should choose i->j and disfavor none strongly.
    """
    W = np.array([[0.0, 1.0],
                  [0.05, 0.0]])
    p = bt_threeway_hier(W, 0, 1, beta_edge=3, beta_dir=5, tau=0.1)

    assert p[0] > 0.95  # i->j
    assert p[1] < 0.05
    assert p[2] < 0.05  # none suppressed

import numpy as np


def _binary_search_F_single(
    F_lb: float,
    F_ub: float,
    G_cur: float,
    copula,
    target: float,
    eps: float = 0.00001,
) -> float:
    F_cur = (F_lb + F_ub) / 2.0
    if F_ub - F_lb < eps:
        return F_cur
    u = np.array([[F_cur, G_cur]])
    temp = 1.0 - F_cur - G_cur + copula.cdf(u)
    if temp > target:
        F_lb = F_cur
    else:
        F_ub = F_cur
    return _binary_search_F_single(F_lb, F_ub, G_cur, copula, target)


def _binary_search_G_single(
    F_cur: float,
    G_lb: float,
    G_ub: float,
    copula,
    target: float,
    eps: float = 0.00001,
) -> float:
    G_cur = (G_lb + G_ub) / 2
    if G_ub - G_lb < eps:
        return G_cur
    u = np.array([[F_cur, G_cur]])
    temp = 1.0 - F_cur - G_cur + copula.cdf(u)
    if temp > target:
        G_lb = G_cur
    else:
        G_ub = G_cur
    return _binary_search_G_single(F_cur, G_lb, G_cur, copula, target)


def estimate(
    observed_times: np.ndarray,
    uncensored: np.ndarray,
    copula,
    weights: np.ndarray | None = None,
):
    """
    Copula-Graphic estimator.
    This method receives any copula.

    Parameters
    ----------
    observed_times : ndarray (float)
        One-dimensional ndarray containing observed times.

    uncensored : ndarray (bool)
        One-dimensional ndarray containing censored (False)
        or uncensored (True).

    copula : object
        Copula function.

    weights : ndarray (float) or None
        One-dimensional ndarray containing weights.
        If None, all weights are set to 1.

    Returns
    -------
    times : ndarray (float)
        One-dimensional ndarray containing time points.

    values : ndarray (float)
        One-dimensional ndarray containing survival rates.
    """

    if weights is None:
        weights = np.ones_like(observed_times)

    # sort values
    l_zip = list(zip(observed_times, uncensored.astype(bool), weights, strict=True))
    l_sorted = sorted(l_zip, key=lambda y: (y[0], ~y[1]))
    z, e, w = zip(*l_sorted, strict=True)
    observed_times = np.array(z)
    uncensored = np.array(e)
    weights = np.array(w)

    # compute
    f = 0.0
    g = 0.0
    times = []
    survival_rates = []
    total_weight = np.sum(weights)
    cum_weight = 0.0
    for i in range(len(observed_times)):
        cum_weight += weights[i]
        target = 1.0 - cum_weight / total_weight
        if uncensored[i]:
            f = _binary_search_F_single(f, 1.0, g, copula, target)
            times.append(observed_times[i])
            survival_rates.append(1.0 - f)
        else:
            g = _binary_search_G_single(f, g, 1.0, copula, target)
    return np.array(times), np.array(survival_rates)

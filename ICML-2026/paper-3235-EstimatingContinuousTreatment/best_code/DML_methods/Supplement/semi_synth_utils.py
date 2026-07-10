import numpy as np

def gen_semi_y(mu_hat: np.ndarray, g: np.ndarray, rng: np.random.Generator):
    """
    Keep X, T unchanged; generate semi-synthetic Y:
      Y_syn = mu_hat + e * g, with e in {-1, +1}.
    """
    n = len(mu_hat)
    e = rng.choice([-1.0, 1.0], size=n)
    Y_syn = mu_hat + e * g
    return Y_syn


def mise_against(grid_t, est_beta, h_star_vals):
    """
    MISE = average over grid of (beta_hat(t) - h*(t))^2
    """
    est_beta = np.asarray(est_beta)
    h_star_vals = np.asarray(h_star_vals)
    return float(np.mean((est_beta - h_star_vals) ** 2))


def summarize_mise(mise_list):
    arr = np.asarray(mise_list, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    se = std / np.sqrt(len(arr)) if len(arr) > 0 else 0.0
    return mean, std, se
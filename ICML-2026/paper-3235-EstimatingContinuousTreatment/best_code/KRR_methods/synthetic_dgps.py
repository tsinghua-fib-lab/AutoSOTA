import numpy as np

# ---------- Global dimension (fixed) ----------
D_X = 10


def sigmoid(z):
    """Numerically stable logistic sigmoid."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def beta_params_from_x(x, c=4.0, kappa=30.0):
    """
    Mean-parameterized Beta assignment:
      s(x) = sum_i x_i
      r(x) = sigmoid(c * s(x))
      U | X ~ Beta(alpha(x), beta(x)) with:
        alpha(x) = kappa * r(x)
        beta(x)  = kappa * (1 - r(x))

    Larger c and/or kappa -> stronger dependence of T on X (more confounding).
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    s = np.sum(x, axis=-1)
    r = sigmoid(c * s)

    alpha = kappa * r
    beta = kappa * (1.0 - r)

    # Numerical safety: ensure strictly positive parameters
    eps = 1e-8
    alpha = np.maximum(alpha, eps)
    beta = np.maximum(beta, eps)
    return alpha, beta


def f_star(x, t):
    """
    Structural regression function:
      f*(x,t) = sin(t) + (sum_i x_i) / D_X

    This nuisance term is aligned with s(x)=sum_i x_i, and since T depends on s(x),
    T-only regression will typically become biased for h*(t)=E_X[f*(X,t)].
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if isinstance(t, (int, float)):
        t = np.full(x.shape[0], t)

    sum_x = np.sum(np.sin(x), axis=1)
    nuisance = sum_x / D_X
    return (np.sin(t) + 4 * nuisance * np.sin(t) + 4 * nuisance)


def sample_t_given_x_batch(X, c=4.0, kappa=30.0):
    """
    X: (n, D_X)
    1) U_i ~ Beta(alpha_i, beta_i) with alpha_i=kappa*r_i, beta_i=kappa*(1-r_i)
    2) T_i = -pi + 2*pi*U_i  (maps U in [0,1] to T in [-pi, pi])
    """
    alpha, beta = beta_params_from_x(X, c=c, kappa=kappa)
    U = np.random.beta(alpha, beta)
    T = 2.0 * np.pi * U - np.pi
    return T


def generate_unified_data(n_samples, noise_std, seed=None):
    """
    X ~ Uniform([-1,1]^D_X)
    T | X via mean-parameterized Beta with r(x)=sigmoid(c*sum x_i)
    Y = f_star(X, T) + Normal(0, noise_std^2)

    Signature and output format unchanged:
      returns (X, T, Y)
    """
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    X = np.random.uniform(-1.0, 1.0, size=(n_samples, D_X))

    # Confounding strength knobs (edit if desired)
    T = sample_t_given_x_batch(X, c=2.0, kappa=20.0)

    fvals = f_star(X, T)
    Y = fvals + float(noise_std) * np.random.randn(n_samples)

    if seed is not None:
        np.random.set_state(rng_state)

    return X, T, Y


def split_data(X_full, T_full, Y_full):
    """Same split format as before."""
    mid = X_full.shape[0] // 2
    return (
        X_full[:mid],
        T_full[:mid],
        Y_full[:mid],
    ), (
        X_full[mid:],
        T_full[mid:],
        Y_full[mid:],
    )


def approximate_h_star(t_array, n_mc_samples=40000, seed=123):
    """
    Monte Carlo approximation of h*(t) = E_X[f*(X,t)] for X ~ Unif([-1,1]^D_X).

    Since E[sum_i X_i] = 0 by symmetry, the nuisance term averages out and:
      h*(t) = sin(t).
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)

    X_mc = np.random.uniform(-1.0, 1.0, size=(n_mc_samples, D_X))
    vals = np.array([f_star(X_mc, t_val).mean() for t_val in t_array])

    np.random.set_state(rng_state)
    return vals

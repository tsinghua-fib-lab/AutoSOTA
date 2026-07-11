import jax.numpy as jnp
from jax import grad, jit
from jax.example_libraries import optimizers
from jax import lax
import numpy as np
from .utils import estimate_triangular_matrices_Ti


# Residuals: shape (m, n)
def residuals(b, x, y):
    return y - (x * b[:, None])


# Empirical residual covariance: shape (m, m)
def residual_covariance(b, x, y):
    r = residuals(b, x, y)  # (m, n)
    return (r @ r.T) / r.shape[1]


# Profile log-likelihood (up to constant): log det S_e(b)
def profile_loss_b(b, x, y, eps=1e-5):
    S_e = residual_covariance(b, x, y)
    S_e = S_e + eps * jnp.eye(S_e.shape[0])  # useful when b tends to 0
    sign, logdet = jnp.linalg.slogdet(S_e)
    return logdet  # drop the sign: S_e should be pos-def anyway


# Optimizer (Adam)
@jit
def optimize_b(x, y, b_init, steps=1000, lr=1e-2):
    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(b_init)

    @jit
    def step(i, opt_state):
        b = get_params(opt_state)
        g = grad(profile_loss_b)(b, x, y)
        return opt_update(i, g, opt_state)

    opt_state = lax.fori_loop(0, steps, step, opt_state)

    return get_params(opt_state)


def profile_log_likelihood(b, x, y):
    S_x = (x @ x.T) / x.shape[1]
    S_e = residual_covariance(b, x, y)
    return -(jnp.linalg.slogdet(S_x)[1] + jnp.linalg.slogdet(S_e)[1])


def compute_ratio_for_two_variables(
    x, y, steps=1000, lr=1e-2, method_for_b="LS_regression"
):
    m, _ = x.shape

    if method_for_b == "MLE":
        b_init = jnp.zeros(m)
        b_hat = optimize_b(x, y, b_init, steps=steps, lr=lr)
        b_hat_reverse = optimize_b(y, x, b_init, steps=steps, lr=lr)
    elif method_for_b == "LS_regression":
        b_hat = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in range(m)])
        b_hat_reverse = b_hat
    
    l1 = profile_log_likelihood(b_hat, x, y)
    l2 = profile_log_likelihood(b_hat_reverse, y, x)
    
    return float(l1 - l2), b_hat, b_hat_reverse


def find_parent_variable(
    X, steps=1000, lr=1e-2, method_for_b="LS_regression", standardize=True
):
    m, p_current, _ = X.shape
    Xc = X - X.mean(axis=-1, keepdims=True)
    if standardize:
        Xz = Xc / Xc.std(axis=-1, keepdims=True)
    else:
        Xz = Xc
    
    # Initialize score matrix and coefficients
    scores = np.zeros((p_current, p_current))
    B = np.zeros((m, p_current, p_current))
    
    # Compute log-ratios for each pair of variables
    indices = [(i, j) for i in range(p_current) for j in range(p_current) if i < j]
    for (i, j) in indices:
        ratio, b_hat, b_hat_reverse = compute_ratio_for_two_variables(
            Xz[:, i], Xz[:, j], steps=steps, lr=lr, method_for_b=method_for_b)
        scores[i, j] = ratio
        scores[j, i] = -ratio
        B[:, i, j] = b_hat
        B[:, j, i] = b_hat_reverse
    
    # Penalize negative scores and find parent variable
    parent_id = np.argmin(np.sum(np.minimum(0, scores) ** 2, axis=1))
    
    # Remove the effect of the parent variable
    B_parent = B[:, parent_id][:, :, np.newaxis]  # shape (m, p_current, 1)
    X_parent = Xz[:, parent_id][:, np.newaxis, :]  # shape (m, 1, n)
    r_all_on_parent = Xz - B_parent * X_parent
    r_all_on_parent = np.delete(
        r_all_on_parent, parent_id, axis=1)  # shape (m, p_current-1, n)
    
    return parent_id, r_all_on_parent


def estimate_causal_order(
    X, steps=1000, lr=1e-2, method_for_b="LS_regression", standardize=True
):
    p = X.shape[1]
    X_current = X.copy()
    
    order = []
    remaining_indices = np.arange(p)
    while len(order) < p - 1:
        parent, X_current = find_parent_variable(
            X_current, steps=steps, lr=lr, method_for_b=method_for_b, 
            standardize=standardize)
        order.append(remaining_indices[parent])
        remaining_indices = np.delete(remaining_indices, parent)
    order.append(remaining_indices[0])

    return order


def pairwise_limvam(
    X, steps=1000, lr=1e-2, method_for_b="LS_regression", standardize=True
):
    """
    Assume the model xi = Bi xi + ei, where Bi are DAG matrices that share the 
    same causal ordering, and the disturbances ei are correlated across views.
    PairwiseLiMVAM identifies the entire causal ordering, and then estimates causal 
    weights using one-step Feasible Generalized Least Squares.
    
    Parameters
    ----------
    X: ndarray of shape (m, p, n)
        Training data, where ``m`` is the number of views, ``p`` is the number 
        of components, and ``n`` is the number of samples.
    
    steps: int, optional (default=1000)
        Number of steps of the Adam optimizer; this parameter is only used
        if method_for_b is 'MLE'.
        
    lr: float, optional (default=1e-2)
        Learning rate of the Adam optimizer; this parameter is only used
        if method_for_b is 'MLE'.
    
    method_for_b: string, optional (default='LS_regression')
        The method used to estimate regression coefficients.
        It can be either 'LS_regression' or 'MLE'.
    
    standardize: bool, optional (default=True)
        If True, observations are standardized to unit variance.
        If False, observations are only centered to zero mean.
    
    Returns
    -------
    B: DAG matrices (ndarray of shape (m, p, p))
    T: Strictly lower triangular matrices (ndarray of shape (m, p, p))
    P: Permutation matrix that contains the ordering (ndarray of shape (p, p))
    """
    # estimate the causal ordering using the likelihood-based criterion
    order = estimate_causal_order(
        X, steps=steps, lr=lr, method_for_b=method_for_b, standardize=standardize)
    
    # estimate causal weights with one-step Feasible Generalized Least Squares
    X_ordered = X[:, order]
    T = estimate_triangular_matrices_Ti(X_ordered)
    
    # reconstruct adjacency matrices
    P = np.eye(X.shape[1])[order]
    B = P.T @ T @ P

    return B, T, P

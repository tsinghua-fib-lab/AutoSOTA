import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

EPS = jnp.finfo(jnp.float32).eps

"""
Adjacency utils
"""
def fill_upper_diag(matrix, k=0, value=-jnp.inf):
    mask = jnp.triu(jnp.ones_like(matrix), k)  # Upper triangular mask (excluding diagonal)
    return jnp.where(mask, value, matrix)


def get_adjacency_mask(permutation):
    dim = permutation.shape[-1]  # Dimension of the input
    inverse_permutation = jnp.argsort(permutation)  # Need the inverse permutation to order the lower-triangular matrix
    row_permutation = inverse_permutation
    col_permutation = inverse_permutation

    # Prepare mask matrix
    M = jnp.ones((dim, dim), dtype=jnp.bool) 
    M_ = fill_upper_diag(M, value=False)
    permuted_M = M_[row_permutation, :][:, col_permutation]  # Shape: (child: dim, parent: dim)
    return permuted_M  # Shape: (dim, dim)


"""
MLP forward utils
"""
def masked_linear(x, weight, bias, mask):
    """
    Custom linear layer with a connection mask.

    Args:
        x: Input tensor, shape (batch_size, in_dim)
        weight: Weight matrix, shape (in_dim, out_dim, hdim)
        bias: Bias vector, shape (out_dim, hdim)
        mask: Binary matrix (in_dim, out_dim) defining allowed connections

    Returns:
        Output tensor, shape (batch_size, out_dim, hdim)
    """
    masked_weight = weight * mask[..., None]
    in_dim, out_dim, h_dim = masked_weight.shape
    return jnp.dot(x, masked_weight.reshape(in_dim, -1)).reshape(-1, out_dim, h_dim) + bias


def contract_independent(h, weight, bias):
    """
    Contract independent hdim-dimensional latent variables

    Args:
        h: Input tensor, shape (batch_size, in_dim, hdim_in)
        weight: Weight matrix with hdim_in parameters per output dimension, shape (in_dim, hdim_in, hdim_out)
        bias: Bias vector, shape (in_dim, hdim_out)

    Returns:
        Output tensor, shape (batch_size, in_dim, hdim_out)
    """
    return jnp.einsum('bij,ijo->bio', h, weight) + bias[None, ...]
    
# Indicate that density is a static argument for the batched_forward function
@partial(jax.jit, static_argnames='density')
def batched_forward(params, x, mask, density='gaussian'):
    """
    Forward pass with masked linear layers.
    Args:
        params: Dictionary containing model parameters including:
            - 'weight_premask': Pre-masked weight matrix, shape (in_dim, out_dim, hdim_in)
            - 'bias_premask': Pre-masked bias vector, shape (out_dim, hdim_in)
            - 'layer_weights': List of weight matrices for each layer, shape (out_dim, hdim_in, hdim_out)
            - 'layer_biases': List of bias vectors for each layer, shape (out_dim, hdim_out)
        x: Input tensor, shape (batch_size, in_dim)
        mask: Binary mask tensor, shape (in_dim, out_dim)

    Returns:
        h_mean: Mean of the output distribution, shape (batch_size, out_dim)
        h_scale: Scale of the output distribution, shape (batch_size, out_dim)
        log_prob: Log-likelihood of the data, shape (batch_size, out_dim)
    """
    loglikelihood_fn = None
    x_input = None
    if density == 'gaussian':
        x_input = x
        loglikelihood_fn = normal_loglikelihood
    elif density == 'zinb':
        x_input = jnp.log1p(x)
        loglikelihood_fn = zinb_loglikelihood
    
    h = masked_linear(x_input, params['weight_premask'], params['bias_premask'], mask)  # Shape: (batch_size, in_dim, hdim_in)
    for w, b in zip(params['layer_weights'], params['layer_biases']):
        h = jax.nn.swish(h)  # Apply activation function
        h = contract_independent(h, w, b)  # Shape: (batch_size, out_dim, hdim_out)

    return loglikelihood_fn(x, h)


def normal_loglikelihood(x, h):
    # Assumes last hdim_out is 2 (parameters for mean and scale)
    mean, h_scale = jnp.split(h, 2, axis=-1)
    mean = mean[..., 0]  # Take the first hdim_out dimension as mean
    h_scale = h_scale[..., 0]  # Take the second hdim_out dimension as scale
    scale = jax.nn.softplus(h_scale) + 0.1 # Ensure scale is positive

    # Calculate Normal log-likelihood    
    log_prob = jax.scipy.stats.norm.logpdf(x, mean, scale)

    dist_params = {
        'mean': mean,
        'scale': scale
    }

    return dist_params, log_prob


def zinb_loglikelihood(x, h, eps = 1e-8):
    # Assumes last hdim_out is 3 (parameters mean, theta, pi)
    mu, theta, pi = jnp.split(h, 3, axis=-1)
    mu = jax.nn.softplus(mu[..., 0]) + 1e-8
    theta = jax.nn.softplus(theta[..., 0]) + 0.1
    pi = pi[..., 0]
    
    # Calculate ZINB log-likelihood
    softplus_pi = jax.nn.softplus(-pi)
    log_theta_eps = jnp.log(theta + eps)
    log_theta_mu_eps = jnp.log(theta + mu + eps)
    log_mu_eps = jnp.log(mu + eps)
    
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)
    log_prob_zero = jax.nn.softplus(pi_theta_log) - softplus_pi
        
    zero_case = jnp.where(x == 0, log_prob_zero, 0.0)
    
    log_prob_nb = (
        -softplus_pi
        + pi_theta_log
        + x * (log_mu_eps - log_theta_mu_eps)
        + jax.scipy.special.gammaln(x + theta + eps)
        - jax.scipy.special.gammaln(theta + eps)
        - jax.scipy.special.gammaln(x + 1)
    )
    non_zero_case = jnp.where(x > 0, log_prob_nb, 0.0)
    log_prob = zero_case + non_zero_case
    
    dist_params = {
        'mu': mu,
        'theta': theta,
        'pi': pi
    }
        
    return dist_params, log_prob

"""
Calculate interventional log-likelihoods
"""
def calculate_interventional_loglikelihoods(params, objective_args_val, density):
    bernoulli = (params['bernoulli_logits'] > 0.0).astype(jnp.float16)
        
    permutation = jnp.argsort(params['logits'], axis=-1)[::-1]
    permuted_M = get_adjacency_mask(permutation)  # Shape: (dim, dim)
    permuted_Mt = (permuted_M * bernoulli).T

    X_val, mask_val, val_regimes = objective_args_val
    dist_params, log_probs = batched_forward(params, X_val, permuted_Mt, density)

    # Log-likelihoods
    scores = -jnp.sum(log_probs*mask_val, axis=1)/jnp.sum(mask_val, axis=1)
    unique_regimes = jnp.unique(val_regimes)
    mask_regimes = val_regimes[None, :] == unique_regimes[:, None]
    ill = jnp.sum(scores*mask_regimes, axis=1) / jnp.sum(mask_regimes, axis=1)
    
    # Interventional MAEs
    maes = None
    if density == 'gaussian':
        # Calculate MAE for Gaussian distribution
        maes = jnp.abs(X_val - dist_params['mean']).mean(axis=1)
        umaes = jnp.sum(maes*mask_regimes, axis=1) / jnp.sum(mask_regimes, axis=1)

    # Log-likelihoods for each sample
    scores = jnp.sum(log_probs*mask_val, axis=1)/jnp.sum(mask_val, axis=1)

    ill = []
    umaes = []
    unique_regimes = set(val_regimes)
    for el in unique_regimes:
        regimes_el = jnp.argwhere(val_regimes == el)
        ill.append(np.array(scores[regimes_el].mean()))
        if density == 'gaussian':
            umaes.append(np.array(maes[regimes_el].mean()))

    ill = jnp.array(ill)

    return {
        'unique_regimes': unique_regimes,
        'unique_regimes_ills': ill,
        'unique_regimes_maes': umaes,
        'sample_regimes': val_regimes,
        'sample_lls': np.array(scores),
        'sample_maes': np.array(maes),
    }

"""
Objective utils
"""
def sum_excluding_index(X, exclude_idx, axis=1):
    n, d = X.shape
    
    # Create a mask of shape (n, d), where True indicates elements to keep
    col_indices = jnp.arange(d).reshape(1, -1)  # shape (1, d)
    
    # Broadcast exclude_idx to (n, d) and compare
    mask = col_indices != exclude_idx[:, None]  # shape (n, d)
    
    # Apply mask and sum
    return jnp.sum(X * mask, axis=axis)


def objective_with_interventions(permutation, bernoulli, params, objective_args, forward_fn, aggregate='sum'):
    # Permutation: sample of Plackett-Luce model, shape (dim,)
    x, regime_masks, regime_idxs = objective_args  # x: Input data, shape (n_samples, dim), int_regimes: Interventional regimes, shape (n_samples,)
    # Get adjacency mask from the permutation
    permuted_M = get_adjacency_mask(permutation)  # Shape: (dim, dim)
    permuted_Mt = (permuted_M * bernoulli).T  # Apply Bernoulli mask on full DAG
    
    # Compute log-likelihoods for the observed and interventional samples
    _, log_prob = forward_fn(params, x, permuted_Mt) # batched_forward(params, x, permuted_Mt)
    score = jnp.sum(log_prob*regime_masks, axis=1)
    if aggregate == 'mean':
        # This matches the DCDFG implementation of the score
        score /= jnp.sum(regime_masks, axis=1)
    score = score.mean(axis=0)
    
    return -score


def objective_value_and_grad_fn(density):
    # Example usage:
    # gaussian_objective_value_and_grad_fn = objective_value_and_grad_fn('gaussian')
    # zinb_objective_value_and_grad_fn = objective_value_and_grad_fn('zinb')
    forward_fn = partial(batched_forward, density=density)
    objective_fn = partial(objective_with_interventions, forward_fn=forward_fn)
    return jax.value_and_grad(objective_fn, argnums=2, has_aux=False)


@partial(jax.jit)
def expected_adjacency_regularizer(params, lambd):
    """
    Returns the expected amount of edges given a space of DAGs defined by params.
    :lambd: regularisation term
    Supports per-edge intervention weights via params['intervention_weights'].
    """
    bernoulli_probs = jax.nn.sigmoid(params['bernoulli_logits'])
    logits_w = jnp.exp(params['logits'])
    n = logits_w.shape[0]

    prob_edge_direction = logits_w[None, :] / (logits_w[:, None] + logits_w[None, :]) # each column corresponds to a different parent
    mask = 1.0 - jnp.eye(n, dtype=prob_edge_direction.dtype)
    N = n * (n - 1) / 2 # graph-wise
    # N = n - 1 # node-wise
    
    # Per-edge intervention weights: penalize edges into frequently-intervened nodes more
    edge_weights = params.get('intervention_weights', jnp.ones((n, n)))
    reg = jnp.sum(bernoulli_probs * prob_edge_direction * mask * edge_weights) / N
    # reg += jnp.sum(jnp.abs(params['weight_premask']) * bernoulli_probs*prob_edge_direction * mask) + jnp.sum(jnp.abs(params['bias_premask'])) # TODO: remove?? posar expected en comptes de sumar directament?
    return lambd*reg


@partial(jax.jit)
def objective_analytic_gaussian(params, objective_args, subset_nodes, aggregate='sum'):
    """
    Computes the expected log-likelihood score under a probabilistic DAG model defined by params 
    on the data given by objective args (x, regime_masks).
    """
    
    x, regime_masks, _ = objective_args  # x: Input data, shape (n_samples, dim), int_regimes: Interventional regimes, shape (n_samples,)
    _, n = x.shape
    
    # Term 2: log of standard deviation
    stds_2 = jax.nn.softplus(params['stds_2'])
    stds_term = jnp.log(2*jnp.pi*stds_2)
    
    # Term 3: Expected under DAG probability space [(x - μ)^2/σ^2]
    logits_w = jnp.exp(params['logits'])

    # Probability determined by B matrix
    prob_bernoulli = jax.nn.sigmoid(params['bernoulli_logits'])
    
    # theta_j / (theta_j + theta_i) = probability j -> i
    prob_edge_direction = logits_w[None, :] / (logits_w[:, None] + logits_w[None, :]) # each column corresponds to a different parent
    mask = 1.0 - jnp.eye(n, dtype=prob_edge_direction.dtype)
    prob_edge = (prob_edge_direction * prob_bernoulli * mask)
    expected_weight_edges = prob_edge * params['weight_premask']
            
    # Code 2
    # contributions[k,i,j] = expected contribution of node j into node i in batch sample k
    term_bias = (x - params['bias_premask'] - jnp.einsum('bi,ji->bj', x, expected_weight_edges))**2
    term_var = jnp.einsum('ki,ji,ji,ji->kj', x**2, prob_edge, 1 - prob_edge, params['weight_premask']**2) 
    
    term_covar = jnp.zeros_like(term_var)
    logits_subset = logits_w[subset_nodes]
    # first dimension is j; second dimension is i; third dimension is k
    matrix = logits_subset[:, None, None] / ((logits_subset[:, None, None] + logits_subset[None, :, None]) + logits_subset[None, None, :]) 
    matrix = jnp.triu(matrix, k=1) # enforce k > i
    expected_weight_edges_subset = expected_weight_edges[subset_nodes, :][:, subset_nodes]
    x_subset = x[:, subset_nodes]
    partial_sum = 2*jnp.einsum('bi,ji,bk,jk,jik->bj', x_subset, expected_weight_edges_subset, x_subset, expected_weight_edges_subset, matrix)
    term_covar = term_covar.at[:, subset_nodes].set(partial_sum)
    dag_term = (term_bias + term_var + term_covar)/stds_2

    log_prob = -(1/2)*(stds_term + dag_term)
    score = jnp.sum(log_prob*regime_masks, axis=1)
    if aggregate == 'mean':
        # This matches the DCDFG implementation of the score
        score /= jnp.sum(regime_masks, axis=1)
    score = score.mean(axis=0)
    return -score



# @partial(jax.jit)
def covariances(params, x, regime_masks, j):
    logits_w = jnp.exp(params['logits'])
    prob_bernoulli = jax.nn.sigmoid(params['bernoulli_logits'])
    n = len(logits_w)
    
    prob_edge_direction = logits_w[None, :] / (logits_w[:, None] + logits_w[None, :])
    mask = 1.0 - jnp.eye(n, dtype=prob_edge_direction.dtype)
    prob_edge = prob_edge_direction * prob_bernoulli * mask
    expected_weight_edges = prob_edge * params['weight_premask']
    
    matrix = 1 + logits_w[j] / (logits_w[j] + logits_w[:, None] + logits_w[None, :])
    matrix = jnp.triu(matrix, k=1)
    x_expected_weight_edges_j =  x*expected_weight_edges[j]
    partial_sum = 2*jnp.einsum('i,k,ik', x_expected_weight_edges_j, x_expected_weight_edges_j, matrix)
    
    return partial_sum*regime_masks[j]

@partial(jax.jit)
def batched_forward_analytic(params, x, mask):
    """
    Given a particular DAG defined by mask, computes the log-likelihood
    of observations x and the distribution parameters of each datapoint (gaussian likelihood).
    """
    masked_weights = (params['weight_premask'].T)*mask
    mean = jnp.dot(x, masked_weights) + params['bias_premask']
    scale = jax.nn.softplus(params['stds_2'])**(1/2)
    log_prob = jax.scipy.stats.norm.logpdf(x, mean, scale)
    dist_params = {
        'mean': mean,
        'scale': scale
    }
    return dist_params, log_prob


def calculate_interventional_loglikelihoods_analytic(params, objective_args_val):
    """
    Computes the mean interventional log-likelihood and MAE across regimes of the data
    in objective_args_val. Density = gaussian.
    """
    # Compute DAG mask
    bernoulli = (params['bernoulli_logits'] > 0.0).astype(jnp.float16)
    permutation = jnp.argsort(params['logits'], axis=-1, descending=True)
    permuted_M = get_adjacency_mask(permutation)  # Shape: (dim, dim)
    permuted_Mt = (permuted_M * bernoulli).T

    # Data
    X_val, mask_val, val_regimes = objective_args_val
    dist_params, log_probs = batched_forward_analytic(params, X_val, permuted_Mt)

    # Log-likelihoods
    scores = -jnp.sum(log_probs*mask_val, axis=1)/jnp.sum(mask_val, axis=1)
    unique_regimes = jnp.unique(val_regimes)
    mask_regimes = val_regimes[None, :] == unique_regimes[:, None]
    ill = jnp.sum(scores*mask_regimes, axis=1) / jnp.sum(mask_regimes, axis=1)
    
    # MAEs
    maes = jnp.abs(X_val - dist_params['mean']).mean(axis=1)
    umaes = jnp.sum(maes*mask_regimes, axis=1) / jnp.sum(mask_regimes, axis=1)

    return {
        'unique_regimes': unique_regimes,
        'unique_regimes_ills': ill,
        'unique_regimes_maes': umaes,
        'sample_regimes': val_regimes,
        'sample_lls': scores,
        'sample_maes': maes,
    }
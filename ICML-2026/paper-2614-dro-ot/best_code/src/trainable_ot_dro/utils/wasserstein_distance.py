import numpy as np
import ot


def wasserstein_distance_L(dist_1, dist_2, L, wasserstein_type=1):
  """
  Computes the wasserstein distance between two discrete distributions and
  also returns the optimal transport plan.
  The wasserstein distance between two distributions is the minimum cost to
  transport probability mass from the first distribution to the second
  and can be calculated using a linear program.

  ### Parameters
  - dist_1: tuple
    - Tuple containing the support and probabilities of the first distribution
    - dist_1 = (supp_1, probs_1)
    - supp_1: np.array of shape (n_points_1, n_dim)
      - The support of the first distribution
    - probs_1: np.array of shape (n_points_1,)
      - The probabilities of the first distribution
  - dist_2: tuple
    - Tuple containing the support and probabilities of the second distribution
    - dist_2 = (supp_2, probs_2)
    - supp_2: np.array of shape (n_points_2, n_dim)
      - The support of the second distribution
    - probs_2: np.array of shape (n_points_2,)
      - The probabilities of the second distribution
  - L: np.array of shape (n_dim, n_dim), lower triangular matrix
    - The cost matrix for the transportation cost: ||L^T(x - y)||_2 where
      x is a point in supp_1 and y is a point in supp_2
  - wasserstein_type: int
    - The type of wasserstein distance to compute. Currently only supports
      wasserstein_type=1 and wasserstein_type=2

  ### Returns
  - wasserstein_dist: float
    - The wasserstein distance between the two distributions
  - transport_plan: np.array
    - The optimal transport plan between the two distributions.
  - d: np.array
    - The differences between all pairs of points in the supports

  """
  assert wasserstein_type == 1 or wasserstein_type == 2, \
    "Invalid wasserstein_type. Only 1 and 2 are supported."
  assert len(L.shape) == 2
  assert L.shape[0] == L.shape[1]
  assert len(dist_1) == 2
  assert len(dist_2) == 2
  supp_1, probs_1 = dist_1
  supp_2, probs_2 = dist_2

  assert supp_1.shape[1] == supp_2.shape[1]
  assert supp_1.shape[0] == probs_1.shape[0]
  assert supp_2.shape[0] == probs_2.shape[0]
  assert len(supp_1.shape) == 2
  assert len(supp_2.shape) == 2
  assert len(probs_1.shape) == 1
  assert len(probs_2.shape) == 1
  assert L.shape[0] == supp_1.shape[1]
  assert np.isclose(np.sum(probs_1), 1)
  assert np.isclose(np.sum(probs_2), 1)
  assert np.all(probs_1 >= 0)
  assert np.all(probs_2 >= 0)

  # Let's get support sizes
  N = supp_1.shape[0]
  M = supp_2.shape[0]

  # Let's get the dimension of the supports
  dim_supp_space = supp_1.shape[1]

  cost_mat = np.zeros((N, M))
  # d contains all differences d_ij = supp_1[i] - supp_2[j]
  d = np.zeros((N * M, dim_supp_space))
  for i in range(N):
    for j in range(M):
      idx = i * M + j
      d[idx] = supp_1[i] - supp_2[j]
      if wasserstein_type == 1:
        cost_mat[i, j] = np.linalg.norm(L.T @ (supp_1[i] - supp_2[j]))
      elif wasserstein_type == 2:
        cost_mat[i, j] = np.linalg.norm(L.T @ (supp_1[i] - supp_2[j]))**2

  opt_transport_plan = ot.emd(probs_1, probs_2, cost_mat)
  optimization_opt_value = np.sum(opt_transport_plan * cost_mat)

  if wasserstein_type == 1:
    wasserstein_dist = optimization_opt_value
  elif wasserstein_type == 2:
    wasserstein_dist = np.sqrt(optimization_opt_value)

  return wasserstein_dist, opt_transport_plan, d


def wasserstein_gradient_L(dist_1, dist_2, L, wasserstein_type=1):
  """
  Compute the gradient of the wasserstein distance between two discrete
  distributions with respect to the cost matrix L.
  Also returns the wasserstein distance.

  Parameters:
  - dist_1: tuple
    - Tuple containing the support and probabilities of the first distribution
    - dist_1 = (supp_1, probs_1)
    - supp_1: np.array of shape (n_points_1, n_dim)
      - The support of the first distribution
    - probs_1: np.array of shape (n_points_1,)
      - The probabilities of the first distribution
  - dist_2: tuple
    - Tuple containing the support and probabilities of the second distribution
    - dist_2 = (supp_2, probs_2)
    - supp_2: np.array of shape (n_points_2, n_dim)
      - The support of the second distribution
    - probs_2: np.array of shape (n_points_2,)
      - The probabilities of the second distribution
  - L: np.array of shape (n_dim, n_dim), lower triangular matrix
    - The cost matrix for the transportation cost: ||L^T(x - y)||_2 where
      x is a point in supp_1 and y is a point in supp_2
  - wasserstein_type: int (1 or 2)
    - The type of wasserstein distance to compute. Currently only supports
      wasserstein_type=1 and wasserstein_type=2

  Returns:
  - wasserstein_dist: float
    - The wasserstein distance between the two distributions
  - grad: (n_dim, n_dim) array
    - The gradient of the wasserstein distance with respect to L
  """

  wasserstein_dist, \
    opt_transport_plan_mat, \
      differences_supp_L = wasserstein_distance_L(dist_1,
                                                  dist_2,
                                                  L,
                                                  wasserstein_type)

  if wasserstein_type == 1:
    grad = vectorized_derivative_wrt_L(L, differences_supp_L,
                                       opt_transport_plan_mat.flatten(),
                                       wasserstein_dist,
                                       wasserstein_type)
  elif wasserstein_type == 2:
    grad = vectorized_derivative_wrt_L(L, differences_supp_L,
                                       opt_transport_plan_mat.flatten(),
                                       wasserstein_dist**2,
                                       wasserstein_type)

  return wasserstein_dist, grad


def vectorized_derivative_wrt_L(L, d,
                                opt_transport_plan_vec,
                                optimization_opt_value,
                                wasserstein_type=1,
                                eps_div=1e-12):
    """
    Vectorized computation of the derivative of f(L) = c(L)^T x
    with respect to L, given that c(L) depends on L through
    ||L^T d_i||_2 and x = opt_transport_plan_vec.

    Parameters:
      - L: (n, n) lower-triangular matrix
      - d: (num_samples, n) difference vectors d_i
      - opt_transport_plan_vec: (num_samples,) optimal transport plan vector
      - optimization_opt_value: scalar, c(L)^T x
      - wasserstein_type: int (1 or 2)
      - eps_div: small constant to avoid division by near-zero

    Returns:
      - deriv_f_wrt_L: (n, n) gradient matrix
    """
    # Precompute H = L L^T
    H = L @ L.T

    # Compute W = d_i^T H d_i for each sample
    temp = d @ H  # shape: (num_samples, n)
    W = np.einsum('ij,ij->i', temp, d)
    f = np.sqrt(W)  # shape: (num_samples,)

    # Compute the outer products for all samples: (num_samples, n, n)
    d_outer = d[:, :, None] * d[:, None, :]

    # Compute the preliminary gradient for each sample: (num_samples, n, n)
    grad_c_wrt_L = np.einsum('ijk,kl->ijl', d_outer, L)

    if wasserstein_type == 1:
        # Use np.errstate to suppress warnings and handle division safely.
        with np.errstate(divide='ignore', invalid='ignore'):
            # Create a safe denominator: where f is too small, use np.inf so that division gives 0.
            safe_f = np.where(f > eps_div, f, np.inf)
            grad_c_wrt_L = np.divide(grad_c_wrt_L, safe_f[:, None, None])
            # Alternatively, you can also mask out entries where f is near zero:
            # grad_c_wrt_L[f < eps_div, :, :] = 0
    elif wasserstein_type == 2:
        grad_c_wrt_L *= 2
    else:
        raise ValueError("Invalid wasserstein_type. Only 1 and 2 are supported.")

    # Sum over all samples weighted by the transport plan
    deriv_f_wrt_L = np.einsum('i,ijl->jl', opt_transport_plan_vec, grad_c_wrt_L)

    # For wasserstein_type == 2, apply final scaling:
    if wasserstein_type == 2:
        if np.abs(optimization_opt_value) >= 1e-15:
            deriv_f_wrt_L *= 0.5 / np.sqrt(optimization_opt_value)
        else:
            deriv_f_wrt_L = np.zeros_like(deriv_f_wrt_L)

    return deriv_f_wrt_L

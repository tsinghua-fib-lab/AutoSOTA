import numpy as np
from trainable_ot_dro.utils.sampling_from_distributions import ellipsoid_uniformsample

def randgen_gaussian_distribution(dim,
                                  lb_mean, ub_mean,
                                  cov_lb, cov_ub,
                                  seed=None):
  """
  Generate a random Gaussian (multivariate normal) distribution in
  dim dimensions.

  The mean vector is drawn uniformly from [lb_mean, ub_mean]^dim.
  The covariance matrix is built by drawing a random matrix from
  [cov_lb, cov_ub]^(dim x dim), multiplying by its transpose, then
  adding a small identity term to ensure positive definiteness.

  Parameters
  ----------
  dim : int
      Dimension of the Gaussian distribution.
  lb_mean : float
      Lower bound for each entry of the mean vector.
  ub_mean : float
      Upper bound for each entry of the mean vector.
  cov_lb : float
      Lower bound for entries in the random matrix used to form the covariance.
  cov_ub : float
      Upper bound for entries in the random matrix used to form the covariance.
  seed : int, optional
      Seed for the local random number generator.

  Returns
  -------
  mean : np.ndarray, shape (dim,)
      Mean vector for the generated Gaussian.
  cov : np.ndarray, shape (dim, dim)
      Covariance matrix for the generated Gaussian.
  """
  rng = np.random.default_rng(seed)

  mean = rng.uniform(lb_mean, ub_mean, size=dim)
  random_mat = rng.uniform(cov_lb, cov_ub, size=(dim, dim))
  cov = random_mat @ random_mat.T + np.eye(dim) * 1e-5

  assert np.all(np.linalg.eigvals(cov) > 0), "Covariance matrix is not positive definite."
  assert np.allclose(cov, cov.T), "Covariance matrix is not symmetric."

  return mean, cov


def randgen_discrete_distribution(dim,
                                  box_bounds=None,
                                  ellipse_bound=None,
                                  n_support=100,
                                  seed=None):
  """
  Generate a random discrete distribution in `dim` dimensions and sample points from it.

  You can specify EITHER:
    1) An n-dimensional box in which to draw the support uniformly, OR
    2) An n-dimensional ellipsoid in which to draw the support uniformly.

  Parameters
  ----------
  dim : int
      Dimensionality of the support points.
  box_bounds : array-like of shape (dim, 2), optional
      Each row [lb, ub] describes the range for that dimension.
      Provide None if using the ellipsoid method.
  ellipse_bound : tuple or None, optional
      (center, E, radius) describing an n-dimensional ellipsoid, where:
        - center : np.array, shape (dim,)
        - E : np.array, shape (dim, dim), SPD
        - radius : float > 0
      Provide None if using the box method.
  n_support : int, optional
      Number of discrete support points (default 100).
  seed : int, optional
      Seed for the local random number generator.

  Returns
  -------
  support_points : np.ndarray, shape (n_support, dim)
      Discrete support points.
  probabilities : np.ndarray, shape (n_support,)
      Probability weights for each support point (summing to 1).
  """
  rng = np.random.default_rng(seed)

  # Validate input
  assert box_bounds is not None or ellipse_bound is not None, \
      "Must provide either `box_bounds` or `ellipse_bound`."
  assert not (box_bounds is not None and ellipse_bound is not None), \
      "Can only provide one of `box_bounds` or `ellipse_bound`."

  # --------------------------------------------------
  # Generate support points
  # --------------------------------------------------
  if box_bounds is not None:
    box_bounds = np.asarray(box_bounds)
    assert box_bounds.shape == (dim, 2), \
        "`box_bounds` must have shape (dim, 2)."
    # Validate each [lb, ub] range
    for i in range(dim):
      lb_i, ub_i = box_bounds[i]
      assert lb_i < ub_i, f"Box bound invalid for dimension {i}: {lb_i} >= {ub_i}"

    low_vec = box_bounds[:, 0]
    high_vec = box_bounds[:, 1]
    support_points = rng.uniform(low=low_vec,
                                  high=high_vec,
                                  size=(n_support, dim))
  else:
    # ellipse_bound is provided
    assert len(ellipse_bound) == 3, \
        "`ellipse_bound` must be a tuple (center, E, radius)."
    center, E, radius = ellipse_bound
    center = np.asarray(center)
    E = np.asarray(E)

    assert center.shape == (dim,), \
        f"`center` must be a vector of shape ({dim},)."
    assert E.shape == (dim, dim), \
        f"`E` must be a matrix of shape ({dim}, {dim})."
    assert np.all(np.linalg.eigvals(E) > 0) and np.allclose(E, E.T), \
        "E must be positive definite and symmetric."
    assert radius > 0, "`radius` must be positive."

    # Generate random support points in the nD ellipsoid
    support_points = ellipsoid_uniformsample(center, E,
                                             n_points=n_support,
                                             radius=radius,
                                             seed=seed)

  # --------------------------------------------------
  # Generate random probabilities for the discrete support
  # --------------------------------------------------
  probabilities = rng.dirichlet(alpha=np.ones(n_support))
  assert np.isclose(probabilities.sum(), 1.0), \
      "Probabilities must sum to 1 (within tolerance)."

  return support_points, probabilities


def randgen_gmm_distribution(n_components,
                             dim,
                             lb_mean,
                             ub_mean,
                             cov_lb,
                             cov_ub,
                             seed=None):
  """
  Randomly generate parameters for a Gaussian Mixture Model (GMM).

  For each component:
    - A random mean vector in [lb_mean, ub_mean]^dim.
    - A random covariance matrix formed like M M^T + 1e-6 * I,
      with M in [cov_lb, cov_ub]^(dim x dim).

  The mixture weights come from a Dirichlet distribution with alpha=1 (uniform prior).

  Parameters
  ----------
  n_components : int
      Number of mixture components.
  dim : int
      Dimensionality of each Gaussian component.
  lb_mean : float
      Lower bound for each entry of the mean vectors.
  ub_mean : float
      Upper bound for each entry of the mean vectors.
  cov_lb : float
      Lower bound for entries in the random matrix used to form the covariance.
  cov_ub : float
      Upper bound for entries in the random matrix used to form the covariance.
  seed : int, optional
      Seed for the local random number generator.

  Returns
  -------
  means : list of np.ndarray, length = n_components
      Each entry is a mean vector (shape (dim,)).
  covariances : list of np.ndarray, length = n_components
      Each entry is a covariance matrix (shape (dim, dim)).
  weights : np.ndarray, shape (n_components,)
      Mixture weights, summing to 1.
  """
  rng = np.random.default_rng(seed)

  means = []
  covariances = []

  for _ in range(n_components):
    # Random mean in [lb_mean, ub_mean]^dim
    mean_i = rng.uniform(lb_mean, ub_mean, size=dim)

    # Random covariance
    M = rng.uniform(cov_lb, cov_ub, size=(dim, dim))
    cov_i = M @ M.T + np.eye(dim) * 1e-6

    means.append(mean_i)
    covariances.append(cov_i)

  # Random mixture weights from Dirichlet
  weights = rng.dirichlet(np.ones(n_components))

  return means, covariances, weights

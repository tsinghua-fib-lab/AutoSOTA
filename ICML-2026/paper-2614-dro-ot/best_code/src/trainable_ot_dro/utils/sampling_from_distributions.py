import numpy as np
import scipy as sp


def ellipsoid_uniformsample(center, E, n_points=100, radius=1.0, seed=None):
  """
  Returns n_points uniformly sampled points inside an n-dimensional ellipsoid
  defined by:

    Eps_set = { y in R^dim | (y - center)^T * E * (y - center) <= radius^2 }

  Parameters
  ----------
  center : np.array, shape (dim,)
    The center of the ellipsoid.
  E : np.array, shape (dim, dim)
    The symmetric positive definite matrix defining the ellipsoid.
  n_points : int
    Number of points to sample.
  radius : float
    The radius of the ellipsoid: (x - center)^T E (x - center) <= radius^2.
  seed : int, optional
    Seed for the local random number generator.

  Returns
  -------
  sample_points : np.array, shape (n_points, dim)
    An array of points sampled uniformly inside the ellipsoid.
  """
  assert radius > 0
  dim = center.shape[0]
  assert E.shape == (dim, dim), "E must be a square matrix of shape (dim, dim)."
  assert n_points > 0, "n_points must be a positive integer."

  # Check E is SPD
  assert np.all(np.linalg.eigvals(E) > 0), "E must be positive definite."
  # Check E is symmetric
  assert np.allclose(E, E.T), "E must be symmetric."

  # Create local RNG
  rng = np.random.default_rng(seed)

  # We want P such that P^T P = E / radius^2
  # Then P_inv = (E / radius^2)^(-1/2)
  P = sp.linalg.cholesky(E / (radius**2), lower=True)
  P_inv = np.linalg.inv(P)

  # -------- Sample uniformly in the unit n-ball --------
  # Step 1: Sample from standard normal
  normal_samples = rng.standard_normal((dim, n_points))  # shape: (dim, n_points)

  # Step 2: Normalize each sample to lie on the unit n-sphere
  norms = np.linalg.norm(normal_samples, axis=0)
  zero_mask = norms < 1e-15

  # In rare cases where norm = 0, re-sample
  while np.any(zero_mask):
    normal_samples[:, zero_mask] = rng.standard_normal((dim, np.sum(zero_mask)))
    norms[zero_mask] = np.linalg.norm(normal_samples[:, zero_mask], axis=0)
    zero_mask = norms < 1e-15

  unit_sphere_points = normal_samples / norms  # shape: (dim, n_points)

  # Step 3: Sample radii ~ U(0,1)^(1/dim) for uniform distribution in the n-ball
  radii = rng.random(n_points) ** (1.0 / dim)

  # Step 4: Scale the unit sphere points by the random radii
  ball_points = unit_sphere_points * radii

  # -------- Map the unit n-ball points through P_inv to get ellipsoid points --------
  ellipsoid_points = P_inv.T @ ball_points  # shape: (dim, n_points)

  # Shift by the center
  ellipsoid_points += center[:, np.newaxis]

  # Return as (n_points, dim)
  return ellipsoid_points.T


def normal_sample(mean, Cov, n_points, seed=None):
  """
  Sample from an n-dimensional normal distribution.

  Parameters
  ----------
  mean : np.array, shape (dim,)
      The mean of the normal distribution
  Cov : np.array, shape (dim, dim)
      The covariance matrix of the normal distribution
  n_points : int
      Number of points to sample
  seed : int, optional
      Seed for the local random number generator

  Returns
  -------
  sample_points : np.array, shape (n_points, dim)
      Points sampled from N(mean, Cov).
  """
  assert mean.shape[0] == Cov.shape[0]
  assert mean.shape[0] == Cov.shape[1]
  assert n_points > 0
  # Check that Cov is valid
  assert np.all(np.linalg.eigvals(Cov) > 0), "Cov must be positive definite."
  assert np.allclose(Cov, Cov.T), "Cov must be symmetric."

  # Local RNG
  rng = np.random.default_rng(seed)

  # Sample points from the multivariate normal
  sample_points = rng.multivariate_normal(mean, Cov, size=n_points)
  return sample_points


def gmm_sample(means, covariances, weights, n_points, seed=None):
  """
  Sample from a Gaussian Mixture Model (GMM).

  Parameters
  ----------
  means : list of np.array
    List of mean vectors for each Gaussian component, each shape (dim,).
  covariances : list of np.array
    List of covariance matrices for each component, each shape (dim, dim).
  weights : np.array
    Array of weights for each component. Must sum to 1.
  n_points : int
    Number of points to sample
  seed : int, optional
    Seed for the local random number generator

  Returns
  -------
  sample_points : np.array, shape (n_points, dim)
    An array of points sampled from the GMM.
  """
  num_components = len(means)
  assert num_components == len(covariances) == len(weights), \
    "All component lists must have the same length."
  assert np.isclose(np.sum(weights), 1.0), "Weights must sum to 1."
  assert n_points > 0, "n_points must be a positive integer."
  for cov in covariances:
    assert np.all(np.linalg.eigvals(cov) > 0) and np.allclose(cov, cov.T), \
      "Each covariance must be positive definite and symmetric."
  for mean, cov in zip(means, covariances):
    assert mean.shape[0] == cov.shape[0], \
      "Mean and covariance dimension mismatch."

  # Local RNG
  rng = np.random.default_rng(seed)

  # Choose component indices for each sample, according to weights
  component_choices = rng.choice(num_components, size=n_points, p=weights)

  # Sample points from each chosen component
  sample_points = np.array([
    rng.multivariate_normal(means[comp], covariances[comp])
    for comp in component_choices
  ])

  return sample_points


def discrete_sample(support_points, probabilities, n_samples, seed=None):
  """
  Sample from a discrete distribution in n-dimensional space.

  The distribution is defined by:
  - A finite set of support points in n-dim (shape: (num_points, dim)).
  - Probabilities for each support point (shape: (num_points,)).

  Parameters
  ----------
  support_points : np.ndarray, shape (num_points, dim)
    The set of discrete points in n-dimensional space that form the support.
  probabilities : np.ndarray, shape (num_points,)
    Probabilities associated with each support point. Must sum to 1.
  n_samples : int
    Number of samples (points) to draw from this discrete distribution.
  seed : int, optional
    Random seed for reproducibility. Defaults to None (no fixed seed).

  Returns
  -------
  samples : np.ndarray, shape (n_samples, dim)
    An array of sampled points, each one drawn from the support according to
    the specified probabilities.
  """
  # Basic validations
  num_points = support_points.shape[0]
  dim = support_points.shape[1]
  assert n_samples > 0, "n_samples must be positive."
  assert len(probabilities) == num_points, "Length of probabilities must match number of support points."
  assert np.isclose(np.sum(probabilities), 1.0), "Probabilities must sum to 1."

  # Set up random generator
  rng = np.random.default_rng(seed)

  # Draw indices of support points according to probabilities
  chosen_indices = rng.choice(num_points, size=n_samples, p=probabilities)

  # Gather the corresponding support points
  samples = support_points[chosen_indices]

  return samples

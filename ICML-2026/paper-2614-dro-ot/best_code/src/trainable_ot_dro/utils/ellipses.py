import numpy as np
import scipy as sp
from scipy.stats import chi2


def ellipsoid_2d_boundary(center, E, n_points=100, radius=1.0):
  """
  Returns n_points (x,y) on an ellipsoid defined by

  Eps = { y in R^2 | (y-center)^T  E  (y-center) = radius^2 }

  ### Parameters
  - center: np.array
    - The center of the ellipsoid in 2D
  - E: np.array
    - The 2x2 symmetric positive definite matrix defining the ellipsoid
  - n_points: int
    - The number of points on the ellipsoid
  - radius: float
    - The radius of the ellipsoid

  ### Returns
  - ellipse_points: np.array
    - An (n_points, 2) array of points on the ellipsoid

  """
  assert radius > 0
  assert center.shape[0] == 2
  assert E.shape[0] == 2
  assert E.shape[1] == 2

  # check that E is positive definite
  assert np.all(np.linalg.eigvals(E) > 0)
  # check that E is symmetric
  assert np.allclose(E, E.T)

  # first we want P = (E/(radius^2))^(1/2) in the sense that
  # radius^2 * (P^T P) = E
  # SCIPY DEFAULT IS UPPER TRIANGULAR CHOLESKY
  P = sp.linalg.cholesky(E / (radius**2))
  P_inv = np.linalg.inv(P)

  # we create points on a unit circle first
  theta = np.linspace(0, 2*np.pi, n_points)
  x_circ = np.cos(theta)
  y_circ = np.sin(theta)
  circ = np.array([x_circ, y_circ])

  # Map unit circle points to the ellipse by applying P^{-1} and shifting by the center
  if center.ndim == 1:
    ellipse_points = P_inv @ circ + center[:, np.newaxis]
  else:
    ellipse_points = P_inv @ circ + center

  # Returning as an (n_points, 2) array
  return ellipse_points.T


def normal_2d_confidence_ellipse(mean, Cov, p=0.95, n_points=100):
    """
    Get the p-level confidence ellipse for a 2D normal distribution defined
    by mean and Cov.

    ### Parameters
    - mean: np.array of shape (2,)
        - The mean of the normal distribution
    - Cov: np.array of shape (2, 2)
        - The 2x2 covariance matrix of the normal distribution
    - p: float
        - The confidence level (default 0.95)
    - n_points: int
        - The number of points to sample on the ellipse (default 100)

    ### Returns
    - ellipse_points: np.array of shape (n_points, 2)
        - An (n_points, 2) array of points on the confidence ellipse
    """
    # Ensure mean is a 1D array
    mean = np.asarray(mean).reshape(2,)

    # Calculate the scaling factor based on the chi-squared distribution
    s = chi2.ppf(p, df=2)  # s = chi2.ppf(p, 2)

    # Eigen decomposition of the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(Cov)

    # Sort the eigenvalues and eigenvectors in descending order
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Generate points on the unit circle
    theta = np.linspace(0, 2 * np.pi, n_points)
    unit_circle = np.vstack([np.cos(theta), np.sin(theta)])  # Shape: (2, n_points)

    # Scale the unit circle by the square roots of the eigenvalues and scaling factor
    scaled_circle = np.sqrt(s) * np.sqrt(eigvals).reshape(2, 1) * unit_circle  # Shape: (2, n_points)

    # Rotate the scaled circle by the eigenvectors
    rotated_circle = eigvecs @ scaled_circle  # Shape: (2, n_points)

    # Translate the ellipse to be centered at the mean
    ellipse_points = rotated_circle + mean.reshape(2, 1)  # Shape: (2, n_points)

    return ellipse_points.T  # Shape: (n_points, 2)

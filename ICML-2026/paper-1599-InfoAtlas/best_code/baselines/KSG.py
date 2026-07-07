import numpy
import math
from scipy.special import digamma, gamma, loggamma
from sklearn.neighbors import BallTree, KDTree

#from .base import MutualInformationEstimator


_metric_tree_types = {
    "kd_tree": KDTree,
    "ball_tree": BallTree
}

def ball_volume(dimensionality: int, radius: float=1.0) -> float:
    """
    Volume of a multidimensional Eucledian ball.
    
    Parameters
    ----------
    dimensionality : int
        Dimensionality.
    radius : float
        Ball radius.
    """
    
    return ((math.sqrt(math.pi) * radius)**dimensionality) / math.gamma(0.5 * dimensionality + 1.0)


class kNN_based(object):
    def __init__(self, k_neighbors: int=1, tree_type: str='kd_tree', tree_kwargs: dict={}, **kwargs) -> None:
        """
        Create a k-NN based mutual information estimator.

        Parameters
        ----------
        k_neighbors : int, optional
            Number of nearest neighbors to use for estimation.
        tree_type : {'ball_tree', 'kd_tree'}, optional
            Specifies the type of metric tree used for estimation.
            KDTree is used by defalt.
        tree_kwargs : dict, optional
            Metric tree additional arguments.
        """

        super().__init__(**kwargs)

        if k_neighbors < 1:
            raise ValueError("The number of neighbors must be at least 1")

        self.k_neighbors = k_neighbors

        if not tree_type in _metric_tree_types:
            raise ValueError(f"The `tree_type` must be in {set(_metric_tree_types.keys())}")

        self.tree_type = tree_type
        self.tree_kwargs = tree_kwargs

    def make_tree(self, data: numpy.ndarray):
        """
        Build a metric tree over the provided data.

        Parameters
        ----------
        data : array_like
            Data used to build the tree.

        Returns
        -------
        tree : BallTree or KDTree
            The metric tree over `data` points.
        """

        return _metric_tree_types[self.tree_type](data, **self.tree_kwargs)


class KSG(kNN_based):
    """
    Kraskov-Stogbauer-Grassberger k-NN based mutual information estimator.

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    """

    def __init__(self, k_neighbors: int=1, tree_type: str='kd_tree', tree_kwargs: dict={}, **kwargs) -> None:
        """
        Create a Kraskov-Stogbauer-Grassberger k-NN based
        mutual information estimator.

        Parameters
        ----------
        k_neighbors : int, optional
            Number of nearest neighbors to use for estimation.
        tree_type : {'ball_tree', 'kd_tree'}, optional
            Specifies the type of metric tree used for estimation.
            KDTree is used by defalt.
        tree_kwargs : dict, optional
            Metric tree additional arguments.

        References
        ----------
        .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
               information". Phys. Rev. E 69, 2004.
        """

        tree_kwargs["metric"] = "chebyshev"
        super().__init__(k_neighbors, tree_type, tree_kwargs, **kwargs)

    #@MutualInformationEstimator.check_arguments
    def __call__(self, x: numpy.ndarray, y: numpy.ndarray, std: bool=False):
        """
        Estimate the value of mutual information between two random vectors
        using samples `x` and `y`.

        Parameters
        ----------
        x, y : array_like
            Samples from corresponding random vectors.
        std : bool
            Calculate standard deviation.

        Returns
        -------
        mutual_information : float
            Estimated value of mutual information.
        mutual_information_std : float or None
            Standard deviation of the estimate, or None if `std=False`
        """

        n_samples = x.shape[0]
        k_neighbors = min(self.k_neighbors, n_samples-1)

        x = x.reshape(n_samples, -1)
        y = y.reshape(n_samples, -1)
        x_y = numpy.concatenate([x, y], axis=-1)

        # Use metric trees for fast nearest neighbors search.
        x_tree = self.make_tree(x)
        y_tree = self.make_tree(y)
        x_y_tree = self.make_tree(x_y)

        x_y_distances, _ = x_y_tree.query(x_y, k=(k_neighbors+1))
        x_y_distances = x_y_distances[:,k_neighbors] - 1.0e-15 # Subtract a small epsilon for tolerance.

        # Count marginal neighbors within x_y_distances.
        x_count = x_tree.query_radius(x, x_y_distances, count_only=True)
        y_count = y_tree.query_radius(y, x_y_distances, count_only=True)

        array = digamma(x_count) + digamma(y_count)
        mean = max(0.0, digamma(k_neighbors) + digamma(n_samples) - numpy.mean(array))

        if std:
            return mean, numpy.std(array) / math.sqrt(n_samples)
        else:
            return mean


# class WKL(kNN_based):
#     """
#     Weighted Kozachenko-Leonenko k-NN based entropy and mutual information estimator.

#     References
#     ----------
#     .. [1] L. F. Kozachenko, N. N. Leonenko, "Sample Estimate of the Entropy
#            of a Random Vector". Problems Inform. Transmission, 23:2 (1987),
#            95–101
#     .. [2] T. B. Berrett, R. J. Samworth and M. Yuan, "Efficient multivariate
#            entropy estimation via k-nearest neighbour distances".
#            Ann. Statist., 47(1):288–318, 02 2019
#     """

#     def __init__(self, k_neighbors: int=1, tree_type: str='kd_tree', tree_kwargs: dict={}, **kwargs) -> None:
#         """
#         Create a weighted Kozachenko-Leonenko k-NN based
#         entropy and mutual information estimator.

#         Parameters
#         ----------
#         k_neighbors : int, optional
#             Number of nearest neighbors to use for estimation.
#         tree_type : {'ball_tree', 'kd_tree'}, optional
#             Specifies the type of metric tree used for estimation.
#             KDTree is used by defalt.
#         tree_kwargs : dict, optional
#             Metric tree additional arguments.

#         References
#         ----------
#         .. [1] L. F. Kozachenko, N. N. Leonenko, "Sample Estimate of the Entropy
#                of a Random Vector". Problems Inform. Transmission, 23:2 (1987),
#                95–101
#         .. [2] T. B. Berrett, R. J. Samworth and M. Yuan, "Efficient multivariate
#                entropy estimation via k-nearest neighbour distances".
#                Ann. Statist., 47(1):288–318, 02 2019
#         """

#         tree_kwargs["metric"] = "euclidean"
#         super().__init__(k_neighbors, tree_type, tree_kwargs, **kwargs)

#     def get_optimal_weights(self, dimensionality: int, rcond: float=1e-6, zero_constraints: bool=True) -> numpy.ndarray:
#         """
#         Otimal weights calculation.
        
#         Parameters
#         ----------
#         dimensionality: int
#             Data dimensionality.
#         rcond: float
#             Cut-off ratio for small singular values in least squares method.
#         zero_constraints: bool
#             Add constraints, zeroing some of the weights.
#         """
        
#         if dimensionality <= 4:
#             # If the number of utilized neighbours is small, the weights are trivial.
#             weights = numpy.zeros(self.k_neighbors)
#             weights[0] = 1.0

#             return weights

            
#         # Build a linear constraint.
#         constraints = []

#         # Constraint: the sum equals one.
#         constraints.append(numpy.ones(self.k_neighbors) / self.k_neighbors)

#         # Consraint: gamma function.
#         n_gamma_constraints = dimensionality // 4
#         for k in range(1, n_gamma_constraints + 1):
#             constraints.append(
#                 #numpy.array([math.gamma(j + 2*k / dimensionality) / math.gamma(j) for j in range(1, self.k_neighbors + 1)])
#                 numpy.exp(loggamma(numpy.arange(1, self.k_neighbors + 1) + 2 * k / dimensionality) - \
#                           loggamma(numpy.arange(1, self.k_neighbors + 1)))
#             )
#             constraints[-1] /= numpy.linalg.norm(constraints[-1])
                
#         # Constraint: zero out some elements.
#         if zero_constraints:
#             nonzero = set(i * self.k_neighbors // dimensionality - 1 for i in range(1, dimensionality + 1))
#             for j in range(self.k_neighbors):
#                 if not j in nonzero:
#                     constraint = numpy.zeros(self.k_neighbors)
#                     constraint[j] = 1.0
#                     constraints.append(constraint)
                    
#         constraints = numpy.vstack(constraints)
            
#         # Right hand side.
#         rhs = numpy.zeros(constraints.shape[0])
#         rhs[0] = 1.0 / self.k_neighbors

#         weights = numpy.linalg.lstsq(constraints, rhs, rcond=rcond)[0]
        
#         return weights

#     def log_density(self, x: numpy.ndarray) -> numpy.ndarray:
#         """
#         Estimate the log-density values at samples `x`.

#         Parameters
#         ----------
#         x : array_like
#             Samples from a random vectors.

#         Returns
#         -------
#         log_density : numpy.ndarray
#             Estimated values of the log-probability density function.
#         """

#         n_samples = x.shape[0]
#         k_neighbors = min(self.k_neighbors, n_samples-1)

#         x = x.reshape(n_samples, -1)
#         x_tree = self.make_tree(x)

#         dimensionality = x.shape[1]
#         unit_ball_volume = ball_volume(dimensionality)

#         psi = numpy.zeros(self.k_neighbors)
#         psi[0] = -numpy.euler_gamma
#         for index in range(1, self.k_neighbors):
#             psi[index] = psi[index - 1] + 1 /  index

#         # Getting `k_neighbors` nearest neighbors.
#         distances, indexes = x_tree.query(x, k_neighbors + 1, return_distance=True)
#         distances = distances[:,1:] + 1.0e-15

#         # Density estimation.
#         log_density = psi - math.log(unit_ball_volume) - dimensionality * numpy.log(distances) - math.log((n_samples - 1))

#         return log_density

#     def entropy(self, x: numpy.ndarray, std: bool=False) -> float | tuple[float, float]:
#         """
#         Estimate the value of differential entropy using samples `x`.

#         Parameters
#         ----------
#         x : array_like
#             Samples from a random vectors.
#         std : bool
#             Calculate standard deviation.

#         Returns
#         -------
#         entropy : float
#             Estimated value of differential entropy.
#         entropy_std : float or None
#             Standard deviation of the estimate, or None if `std=False`
#         """

#         n_samples = x.shape[0]
        
#         x = x.reshape(n_samples, -1)
#         dimensionality = x.shape[1]

#         log_density = self.log_density(x)
#         weights = self.get_optimal_weights(dimensionality)

#         array = -log_density @ weights
#         mean = array.mean()

#         if std:
#             return mean, array.std() / math.sqrt(n_samples)
#         else:
#             return mean

#     @MutualInformationEstimator.check_arguments
#     def __call__(self, x: numpy.ndarray, y: numpy.ndarray, std: bool=False) -> float | tuple[float, float]:
#         """
#         Estimate the value of mutual information between two random vectors
#         using samples `x` and `y`.

#         Parameters
#         ----------
#         x, y : array_like
#             Samples from corresponding random vectors.
#         std : bool
#             Calculate standard deviation.

#         Returns
#         -------
#         mutual_information : float
#             Estimated value of mutual information.
#         mutual_information_std : float or None
#             Standard deviation of the estimate, or None if `std=False`
#         """

#         n_samples = x.shape[0]

#         x = x.reshape(n_samples, -1)
#         y = y.reshape(n_samples, -1)
#         x_y = numpy.concatenate([x, y], axis=-1)

#         X_dim = x.shape[1]
#         Y_dim = y.shape[1]

#         x_log_density = self.log_density(x)
#         y_log_density = self.log_density(y)
#         x_y_log_density = self.log_density(x_y)
        
#         x_weights = self.get_optimal_weights(X_dim)
#         y_weights = self.get_optimal_weights(Y_dim)
#         x_y_weights = self.get_optimal_weights(X_dim + Y_dim)

#         array = x_y_log_density @ x_y_weights - x_log_density @ x_weights - y_log_density @ y_weights
#         mean = max(0.0, array.mean())

#         if std:
#             return mean, array.std() / math.sqrt(n_samples)
#         else:
#             return mean
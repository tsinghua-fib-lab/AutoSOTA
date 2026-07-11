# Adapted from https://github.com/geomstats/geomstats/blob/main/geomstats/geometry/hypersphere.py
"""The n-dimensional hypersphere.

The n-dimensional hypersphere embedded in (n+1)-dimensional
Euclidean space.
"""
import abc
import torch
from .utils import algebra_utils as autils


class Hypersphere(abc.ABC):
    """Class for the n-dimensional hypersphere.

    Class for the n-dimensional hypersphere embedded in the
    (n+1)-dimensional Euclidean space.

    By default, points are parameterized by their extrinsic
    (n+1)-coordinates.

    Parameters
    ----------
    dim : int
        Dimension of the hypersphere.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def projection(self, point):
        """Project a point on the hypersphere.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point in embedding Euclidean space.

        Returns
        -------
        projected_point : array-like, shape=[..., dim + 1]
            Point projected on the hypersphere.
        """
        norm = torch.linalg.norm(point, axis=-1)
        projected_point = torch.einsum("...,...i->...i", 1.0 / norm, point)
        return projected_point

    def to_tangent(self, vector, base_point):
        """Project a vector to the tangent space.

        Project a vector in Euclidean space
        on the tangent space of the hypersphere at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim + 1]
            Vector in Euclidean space.
        base_point : array-like, shape=[..., dim + 1]
            Point on the hypersphere defining the tangent space,
            where the vector will be projected.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim + 1]
            Tangent vector in the tangent space of the hypersphere
            at the base point.
        """
        sq_norm = torch.linalg.norm(base_point, axis=-1)**2
        inner_prod = self.inner_product(base_point, vector.to(torch.float32))
        coef = inner_prod / sq_norm
        tangent_vec = vector.to(torch.float32) - torch.einsum("...,...j->...j", coef, base_point)
        return tangent_vec

    def is_tangent(self, vector, base_point, atol=1e-6):
        tangent_submersion = 2 * torch.sum(self.inner_product(vector, base_point), axis=-1)
        return torch.allclose(tangent_submersion, torch.zeros_like(tangent_submersion), atol=atol)

    def random_point(self, n_samples=1, bound=1.0, device='cpu'):
        """Sample in the hypersphere from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : unused

        Returns
        -------
        samples : array-like, shape=[..., dim + 1]
            Points sampled on the hypersphere.
        """
        return self.random_uniform(n_samples, device)

    def random_uniform(self, n_samples=1, device='cpu'):
        """Sample in the hypersphere from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., dim + 1]
            Points sampled on the hypersphere.
        """
        size = (n_samples, self.dim + 1)
        samples = torch.randn(size, device=device)
        norms = torch.linalg.norm(samples, axis=1)
        samples = torch.einsum("..., ...i->...i", 1 / norms, samples)    
        return samples
    
    def belongs(self, point, atol=1e-6):
        """Evaluate if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """
        norm2 = torch.linalg.norm(point, axis=-1)**2
        return torch.allclose(norm2, torch.ones_like(norm2), atol=atol)
    
    # Metric
    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the inner-product of two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim + 1]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., dim + 1], optional
            Point on the hypersphere.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Inner-product of the two tangent vectors.
        """
        inner_prod = torch.einsum("...i,...i->...", tangent_vec_a, tangent_vec_b)

        return inner_prod

    def squared_norm(self, vector, base_point=None):
        """Compute the squared norm of a vector.

        Squared norm of a vector associated with the inner-product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim + 1]
            Vector on the tangent space of the hypersphere at base point.
        base_point : array-like, shape=[..., dim + 1], optional
            Point on the hypersphere.

        Returns
        -------
        sq_norm : array-like, shape=[..., 1]
            Squared norm of the vector.
        """
        sq_norm = self.inner_product(vector, vector, base_point)
        return sq_norm

    def norm(self, vector, base_point=None):
        """Compute norm of a vector.

        Norm of a vector associated to the inner product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        norm : array-like, shape=[...,]
            Norm.
        """
        sq_norm = self.squared_norm(vector, base_point)
        norm = torch.sqrt(sq_norm)
        return norm

    def dist(self, point_a, point_b):
        """Compute the geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim + 1]
            First point on the hypersphere.
        point_b : array-like, shape=[..., dim + 1]
            Second point on the hypersphere.

        Returns
        -------
        dist : array-like, shape=[..., 1]
            Geodesic distance between the two points.
        """
        norm_a = self.norm(point_a)
        norm_b = self.norm(point_b)
        inner_prod = self.inner_product(point_a, point_b)

        cos_angle = inner_prod / (norm_a * norm_b)
        cos_angle = torch.clip(cos_angle, -1, 1)
        dist = torch.arccos(cos_angle)

        return dist

    def squared_dist(self, point_a, point_b, **kwargs):
        """Squared geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            Point on the hypersphere.
        point_b : array-like, shape=[..., dim]
            Point on the hypersphere.

        Returns
        -------
        sq_dist : array-like, shape=[...,]
        """
        return self.dist(point_a, point_b) ** 2

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim + 1]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., dim + 1]
            Point on the hypersphere.

        Returns
        -------
        exp : array-like, shape=[..., dim + 1]
            Point on the hypersphere equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        proj_tangent_vec = self.to_tangent(tangent_vec, base_point)
        norm2 = self.squared_norm(proj_tangent_vec)
        
        coef_1 = autils.taylor_exp_even_func(norm2, autils.cos_close_0, order=4)
        coef_2 = autils.taylor_exp_even_func(norm2, autils.sinc_close_0, order=4)
        exp = torch.einsum("...,...j->...j", coef_1, base_point) + torch.einsum(
            "...,...j->...j", coef_2, proj_tangent_vec)

        return exp

    def log(self, point, base_point, eps=1e-6, **kwargs):
        """Compute the Riemannian logarithm of a point.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point on the hypersphere.
        base_point : array-like, shape=[..., dim + 1]
            Point on the hypersphere.

        Returns
        -------
        log : array-like, shape=[..., dim + 1]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        inner_prod = self.inner_product(base_point, point.to(torch.float32))
        cos_angle = torch.clip(inner_prod, -1.0+eps, 1.0-eps) # prevent gradient explosion
        squared_angle = torch.arccos(cos_angle) ** 2
        coef_1_ = autils.taylor_exp_even_func(
            squared_angle, autils.inv_sinc_close_0, order=5
        )
        coef_2_ = autils.taylor_exp_even_func(
            squared_angle, autils.inv_tanc_close_0, order=5
        )
        log = torch.einsum("...,...j->...j", coef_1_, point.to(torch.float32)) - torch.einsum(
            "...,...j->...j", coef_2_, base_point)

        return log

    def random_normal_tangent(self, base_point):
        """Sample in the tangent space from the standard normal distribution.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        return torch.randn_like(base_point)
    
    def map_to_simplex(self, point):
        # project to positive orthant of the sphere and map to simplex
        pos = torch.where(point<0, 0., point)
        pos = pos / pos.square().sum(-1, keepdim=True).sqrt()
        return torch.square(pos) # probs on simplex
    
    def weighted_sum(self, prob, point, eps=1e-6):
        cos_angle = torch.clip(point, -1.0+eps, 1.0-eps)
        cos_angle = torch.arccos(cos_angle)
        drift = prob * cos_angle / torch.sin(cos_angle)
        drift = drift -torch.einsum("...,...i->...i", (drift * point).sum(-1), point)
        return drift
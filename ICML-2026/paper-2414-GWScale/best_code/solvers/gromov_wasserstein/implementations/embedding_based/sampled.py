"""
Sampled Gromov-Wasserstein Solver

Implements a sampled approach to GW using subsampled cost matrices.
This is the KeOps implementation of:

Kerdoncuff, T., Emonet, R., & Sebban, M. (2021).
Sampled gromov wasserstein.
Machine Learning, 110(8), 2151-2186.
"""

import torch
from pykeops.torch import LazyTensor

from solvers.gromov_wasserstein.generic.embedding_based import EmbeddingBasedGW
from utils.implementation.initializations import initialize_sampledgw


class SampledGW(EmbeddingBasedGW):
    """
    Gromov-Wasserstein solver using stochastic sampling of cost matrices.

    Instead of computing full (N, M) cost matrices, this solver maintains a small set of sampled indices and only
    computes costs for these samples.

    Attributes:
        X_init (torch.Tensor): Source point cloud
        Y_init (torch.Tensor): Target point cloud
        cost_x (callable): Cost function for source space
        cost_y (callable): Cost function for target space
        samples (int): Number of samples to use per iteration
        indices_k (torch.Tensor): Current sampled indices in source space
        indices_l (torch.Tensor): Current sampled indices in target space
    """

    def __init__(self, X_init, Y_init, costs, samples=10, **kwargs):
        """
        Initialize the sampled Gromov-Wasserstein solver.

        Args:
            X_init: Source point cloud, shape (N, D_x)
            Y_init: Target point cloud, shape (M, D_y)
            costs: Cost function(s) for computing base costs. Can be:
               - Single function: used for both X and Y
               - Tuple (cost_x, cost_y): separate functions for each space
            samples: Number of samples to use for approximation
            **kwargs: Additional arguments passed to EmbeddingBasedGW
        """
        self.X_init = X_init
        self.Y_init = Y_init

        self.samples = samples
        self.indices_k, self.indices_l = None, None

        super().__init__(X=X_init, Y=Y_init, costs=costs, lazy=True, **kwargs)

    def parameters(self):
        params = super().parameters()
        params['samples'] = self.samples

        return params

    def _initialize_indices(self):
        """
        Initialize sampled indices (k,l) based on the distribution given by diag(a) (if self.initialization_mode ==
        'identity') or by a b.T otherwise.
        """
        if self.initialization_mode == 'identity':
            self.indices_k = torch.multinomial(self.a, num_samples=self.samples, replacement=True)
            self.indices_l = self.indices_k
        else:
            self.indices_k = torch.multinomial(self.a, num_samples=self.samples, replacement=True)
            self.indices_l = torch.multinomial(self.b, num_samples=self.samples, replacement=True)

    def initialize_potential(self):
        """
        Initialize potential using sampled cost matrices.

        Returns:
            tuple: (Dx_is, Dy_js) sampled cost vectors of shape (N, samples) and (M, samples)
        """
        self._initialize_indices()
        return initialize_sampledgw(self.X, self.Y, self.a, self.b, self.cost_x, self.cost_y, self.indices_k,
                                    self.indices_l)

    def _sampling_indices(self):
        """
        Sample new indices based on current transport plan.

        Performs importance sampling from the transport plan to focus on relevant point pairs.

        Returns:
            tuple: (new_indices_k, new_indices_l) sampled indices
        """
        new_indices_k = torch.multinomial(self.a, num_samples=self.samples, replacement=True)  # (S,)

        Cx_is = self.cost_x(self.X[new_indices_k, None, :], self.X[None, self.indices_k, :])  # (S', S)
        Cy_ls = self.cost_y(self.Y[:, None, :], self.Y[None, self.indices_l, :])  # (M, S)
        C_il = ((Cx_is[:, None, :] - Cy_ls[None, :, :]) ** 2).sum(dim=2) / self.samples

        f, g = self.sinkhorn_solver.f, self.sinkhorn_solver.g
        T_js = ((f[new_indices_k, None] + g[None, :] - C_il) / self.eps).exp()  # (M, S)

        new_indices_l = torch.multinomial(T_js, num_samples=1, replacement=True).squeeze()
        return new_indices_k, new_indices_l

    def update_potential(self):
        """
        Update the potentials based on current transport plan.

        The potentials are a tuple of matrices containing the sampled approximations of the base cost matrices

        Returns:
            tuple: (Cx_is, Cy_js) updated sampled cost vectors of shape (N, samples) and (M, samples)
        """
        self.indices_k, self.indices_l = self._sampling_indices()

        Cx_is = self.cost_x(self.X[:, None, :], self.X[None, self.indices_k, :])  # (N, S)
        Cy_js = self.cost_y(self.Y[:, None, :], self.Y[None, self.indices_l, :])  # (M, S)

        return Cx_is, Cy_js

    def cost_matrix(self, lazy=None):
        """
        Compute the cost matrix for Sinkhorn using sampled base costs.

        Args:
            lazy: If True, return C_ij as a LazyTensor

        Returns:
            torch.Tensor or LazyTensor: Cost matrix, shape (N, M)
        """
        if lazy is None:
            lazy = self.lazy

        Cx_is, Cy_js = self.Z[0][:, None, :], self.Z[1][None, :, :]  # (N, 1, S), (1, M, S)

        if lazy:
            Cx_is, Cy_js = LazyTensor(Cx_is), LazyTensor(Cy_js)

        C_ij = ((Cx_is - Cy_js) ** 2).sum(dim=2) / self.samples  # (N, M, 1)
        return C_ij

    def potential_variation(self):
        """
        Define the potential error as the sum of the two potentials variation.

        Returns:
            None

        Note:
            Since potentials (Z[0], Z[1]) are dependent on currently sampled indices, this metric is not a good indicator
            of the solver convergence.

        """
        return ((self.Z[0] - self.Z_new[0]).abs().mean() + (self.Z[1] - self.Z_new[1]).abs().mean()).item()

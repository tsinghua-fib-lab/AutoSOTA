"""
Quadratic GW with Dual Gradient Acceleration

Implements the dual gradient descent for squared Euclidean norms proposed in:

Rioux, G., Goldfeld, Z., & Kato, K. (2024).
Entropic gromov-wasserstein distances: Stability and algorithms.
Journal of Machine Learning Research, 25(363), 1-52.
"""

import torch

from solvers.gromov_wasserstein.implementations.embedding_based.quadratic import QuadraticGW
from utils.math.functions import froebenius, moment, coupling_covariance


class QuadraticDualGradGW(QuadraticGW):
    """
        Quadratic GW solver with dual gradient descent with adaptive step sizes.

        The dual gradient method operates on the potential matrix Z, applying gradient descent with adaptive step sizes
        and momentum terms.

        Attributes:
          X (torch.Tensor): Source point cloud, shape (N, D_x)
          Y (torch.Tensor): Target point cloud, shape (M, D_y)
          L (float): Lipschitz constant for gradient (controls step size)
          M (float): Bound on feasible set
          k (int): Current iteration counter (used for momentum scheduling)
          W (torch.Tensor): Momentum state variable, shape (D_x, D_y)
      """

    def __init__(self, X, Y, L=None, **kwargs):
        """
        Initialize dual gradient quadratic GW solver.

        Args:
            X: Source point cloud, shape (N, D_x)
            Y: Target point cloud, shape (M, D_y)
            L: Lipschitz constant for gradient. If None, estimated from 4th moments:
                L ≈ 256 * sqrt(M4(X) * M4(Y)) / eps
            **kwargs: Additional arguments passed to QuadraticGW

        """
        super().__init__(X=X, Y=Y, **kwargs)

        if L is None:
            self.L = max(16., float(256 * torch.sqrt(moment(X, self.a, p=4) * moment(Y, self.b, p=4)) / self.eps))
        else:
            self.L = L

        self.M = 2 * torch.sqrt(moment(X, self.a, p=2) * moment(Y, self.b, p=2))
        self.k = 1

        self.W = self.Z.clone()

    def clear(self):
        """
        Reset solver state including momentum and iteration counter.
        """
        super().clear()
        self.k = 1
        self.W = self.Z.clone()

    def update_potential(self):
        """
        Update potential using accelerated dual gradient method.

        Returns:
            torch.Tensor: Updated potential matrix Z, shape (D_x, D_y)
        """
        beta, gamma, tau = 1 / (2 * self.L), self.k / (4 * self.L), 2 / (self.k + 2)

        G = 16 * self.Z - 16 * coupling_covariance(self.X_centered, self.Y_centered, self.transport_plan(lazy=True))
        B = min(1, self.M / (2 * froebenius(self.Z - beta * G))) * (self.Z - beta * G)
        self.W = min(1, self.M / (2 * froebenius(self.W - gamma * G))) * (self.W - gamma * G)

        Z = tau * self.W + (1 - tau) * B

        self.k += 1

        return Z

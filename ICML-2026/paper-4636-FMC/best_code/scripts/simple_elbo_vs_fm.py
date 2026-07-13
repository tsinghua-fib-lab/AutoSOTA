#!/usr/bin/env python3
"""Simple comparison of ELBO vs Flow Matching for source learning.

This is a minimal demonstration using a simple Gaussian hierarchy:
- theta ~ N(0, 1)
- x ~ N(A*theta + b, I)
- y ~ N(C*x + d, I)

We train two methods to learn p(x|y):
1. ELBO with true posterior p(theta|x)
2. Flow Matching (standard conditional flow)

Both use the same architecture (Zuko NSF for ELBO, FlowMatching for FM).
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from zuko.flows import NSF
from zuko.distributions import DiagNormal

from flow_matching.torch_flow import FlowMatching
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier


def c2st(
    X: torch.Tensor,
    Y: torch.Tensor,
    seed: int = 1,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
) -> torch.Tensor:
    """Classifier-based 2-sample test returning accuracy

    Trains classifiers with N-fold cross-validation [1]. Scikit learn MLPClassifier are
    used, with 2 hidden layers of 10x dim each, where dim is the dimensionality of the
    samples X and Y.

    Args:
        X: Sample 1
        Y: Sample 2
        seed: Seed for sklearn
        n_folds: Number of folds
        z_score: Z-scoring using X
        noise_scale: If passed, will add Gaussian noise with std noise_scale to samples

    References:
        [1]: https://scikit-learn.org/stable/modules/cross_validation.html
    """
    if z_score:
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * torch.randn(X.shape)
        Y += noise_scale * torch.randn(Y.shape)

    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    ndim = X.shape[1]

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter=10000,
        solver="adam",
        random_state=seed,
    )

    data = np.concatenate((X, Y))
    target = np.concatenate(
        (
            np.zeros((X.shape[0],)),
            np.ones((Y.shape[0],)),
        )
    )

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    scores = np.asarray(np.mean(scores)).astype(np.float32)
    return torch.from_numpy(np.atleast_1d(scores))


# ============================================================================
# 1. Simple Gaussian Model
# ============================================================================


class SimpleGaussianModel:
    """Simple hierarchical Gaussian model: theta -> x -> y."""

    def __init__(self, theta_dim=2, x_dim=3, y_dim=3, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.y_dim = y_dim

        # x = A*theta + b + noise
        self.A = torch.randn(x_dim, theta_dim)
        self.b = torch.randn(x_dim)

        # y = C*x + d + noise
        self.C = torch.randn(y_dim, x_dim)
        self.d = torch.randn(y_dim)

        print(f"Model: theta({theta_dim}) -> x({x_dim}) -> y({y_dim})")
        print(f"  x = A*theta + b + N(0,I)")
        print(f"  y = C*x + d + N(0,I)")

    def sample(self, n_samples):
        """Sample from the generative model."""
        # theta ~ N(0, I)
        theta = torch.randn(n_samples, self.theta_dim)

        # x ~ N(A*theta + b, I)
        x_mean = theta @ self.A.T + self.b
        x = x_mean + torch.randn(n_samples, self.x_dim)

        # y ~ N(C*x + d, I)
        y_mean = x @ self.C.T + self.d
        y = y_mean + torch.randn(n_samples, self.y_dim)

        return theta, x, y

    def true_posterior_theta_given_x(self, x):
        """Compute true posterior p(theta|x) analytically.

        For Gaussian model: x = A*theta + b + noise
        where theta ~ N(0, I) and noise ~ N(0, I)

        Posterior is: theta|x ~ N(mu_post, Sigma_post)
        where:
          Sigma_post = (A^T A + I)^{-1}
          mu_post = Sigma_post @ A^T @ (x - b)
        """
        # Posterior covariance
        ATA = self.A.T @ self.A
        Sigma_post = torch.inverse(ATA + torch.eye(self.theta_dim))

        # Posterior mean for each x
        x_centered = x - self.b
        mu_post = (Sigma_post @ self.A.T @ x_centered.T).T

        return mu_post, Sigma_post

    def log_prob_theta_given_x(self, theta, x):
        """Compute log p(theta|x) using true posterior."""
        # Move model matrices to same device as input
        device = x.device
        A = self.A.to(device)
        b = self.b.to(device)

        # Compute posterior on device
        ATA = A.T @ A
        Sigma_post = torch.inverse(ATA + torch.eye(self.theta_dim, device=device))
        x_centered = x - b
        mu_post = (Sigma_post @ A.T @ x_centered.T).T

        # Log probability of multivariate Gaussian
        diff = theta - mu_post
        log_prob = -0.5 * (diff @ torch.inverse(Sigma_post) @ diff.T).diag()
        log_prob -= 0.5 * torch.logdet(Sigma_post)
        log_prob -= 0.5 * self.theta_dim * np.log(2 * np.pi)

        return log_prob

    def true_posterior_theta_given_y(self, y):
        """Compute true posterior p(theta|y) analytically.

        For the hierarchical model:
        - theta ~ N(0, I)
        - x ~ N(A*theta + b, I)
        - y ~ N(C*x + d, I)

        We marginalize over x to get p(theta|y).

        Joint (theta, y) is Gaussian:
        - theta ~ N(0, I)
        - y|theta ~ N(C*(A*theta + b) + d, C*C^T + I)

        Using Gaussian conditioning formulas:
        - Sigma_y = C*A*A^T*C^T + C*C^T + I (marginal covariance of y)
        - Sigma_theta_y = A^T * C^T (cross-covariance)
        - mu_post = Sigma_theta_y * Sigma_y^{-1} * (y - (C*b + d))
        - Sigma_post = I - Sigma_theta_y * Sigma_y^{-1} * Sigma_theta_y^T
        """
        device = y.device
        A = self.A.to(device)
        b = self.b.to(device)
        C = self.C.to(device)
        d = self.d.to(device)

        # Marginal covariance of y: Var(y) = C*A*A^T*C^T + C*C^T + I
        # = C*(A*A^T + I)*C^T + I
        AAt = A @ A.T
        Sigma_y = C @ (AAt + torch.eye(self.x_dim, device=device)) @ C.T + torch.eye(
            self.y_dim, device=device
        )

        # Cross-covariance: Cov(theta, y) = A^T * C^T
        Sigma_theta_y = A.T @ C.T

        # Inverse of marginal covariance
        Sigma_y_inv = torch.inverse(Sigma_y)

        # Posterior mean: mu_post = Sigma_theta_y @ Sigma_y_inv @ (y - mean_y)
        mean_y = C @ b + d
        y_centered = y - mean_y
        mu_post = (Sigma_theta_y @ Sigma_y_inv @ y_centered.T).T

        # Posterior covariance: Sigma_post = I - Sigma_theta_y @ Sigma_y_inv @ Sigma_theta_y^T
        Sigma_post = (
            torch.eye(self.theta_dim, device=device) - Sigma_theta_y @ Sigma_y_inv @ Sigma_theta_y.T
        )

        return mu_post, Sigma_post

    def true_posterior_x_given_y(self, y):
        """Compute true posterior p(x|y) analytically.

        For the hierarchical model:
        - theta ~ N(0, I)
        - x ~ N(A*theta + b, I) -> marginal: x ~ N(b, A*A^T + I)
        - y ~ N(C*x + d, I)

        Joint (x, y) is Gaussian:
        - mu_x = b
        - mu_y = C*b + d
        - Sigma_xx = A*A^T + I
        - Sigma_yy = C*(A*A^T + I)*C^T + I
        - Sigma_xy = (A*A^T + I)*C^T

        Using Gaussian conditioning:
        - mu_x|y = mu_x + Sigma_xy * Sigma_yy^{-1} * (y - mu_y)
        - Sigma_x|y = Sigma_xx - Sigma_xy * Sigma_yy^{-1} * Sigma_xy^T
        """
        device = y.device
        A = self.A.to(device)
        b = self.b.to(device)
        C = self.C.to(device)
        d = self.d.to(device)

        # Marginal covariance of x: Var(x) = A*A^T + I
        AAt = A @ A.T
        Sigma_xx = AAt + torch.eye(self.x_dim, device=device)

        # Marginal covariance of y: Var(y) = C*Sigma_xx*C^T + I
        Sigma_yy = C @ Sigma_xx @ C.T + torch.eye(self.y_dim, device=device)

        # Cross-covariance: Cov(x, y) = Sigma_xx * C^T
        Sigma_xy = Sigma_xx @ C.T

        # Inverse of marginal covariance
        Sigma_yy_inv = torch.inverse(Sigma_yy)

        # Posterior mean: mu_x|y = b + Sigma_xy @ Sigma_yy_inv @ (y - (C*b + d))
        mu_x = b
        mu_y = C @ b + d
        y_centered = y - mu_y
        mu_post = mu_x + (Sigma_xy @ Sigma_yy_inv @ y_centered.T).T

        # Posterior covariance: Sigma_x|y = Sigma_xx - Sigma_xy @ Sigma_yy_inv @ Sigma_xy^T
        Sigma_post = Sigma_xx - Sigma_xy @ Sigma_yy_inv @ Sigma_xy.T

        return mu_post, Sigma_post

    def eval_posterior_x_given_y_on_grid(self, y, x_range=None, n_grid=100):
        """Evaluate p(x|y) on a 2D grid for contour plotting (x[0] vs x[1]).

        Args:
            y: Observation (1, y_dim)
            x_range: Range for x grid, e.g., [(-3, 3), (-3, 3)]
            n_grid: Number of grid points per dimension

        Returns:
            x0_grid, x1_grid, density_grid
        """
        if x_range is None:
            x_range = [(-4, 4), (-4, 4)]

        # Get posterior parameters
        mu_post, Sigma_post = self.true_posterior_x_given_y(y)
        mu_post = mu_post.squeeze()  # (x_dim,)

        # Create grid for first two dimensions of x
        x0 = torch.linspace(x_range[0][0], x_range[0][1], n_grid)
        x1 = torch.linspace(x_range[1][0], x_range[1][1], n_grid)
        x0_grid, x1_grid = torch.meshgrid(x0, x1, indexing="ij")

        # For each grid point (x0, x1), marginalize over x[2:] if x_dim > 2
        if self.x_dim == 2:
            # Simple case: just evaluate 2D Gaussian
            x_flat = torch.stack([x0_grid.flatten(), x1_grid.flatten()], dim=1)
            diff = x_flat - mu_post
            Sigma_inv = torch.inverse(Sigma_post)
            log_prob = -0.5 * (diff @ Sigma_inv * diff).sum(dim=1)
            log_prob -= 0.5 * torch.logdet(Sigma_post)
            log_prob -= 0.5 * self.x_dim * np.log(2 * np.pi)
        else:
            # Marginalize over x[2:] to get 2D marginal for (x[0], x[1])
            # Marginal of 2D Gaussian is just the top-left 2x2 block
            mu_post_2d = mu_post[:2]
            Sigma_post_2d = Sigma_post[:2, :2]

            x_flat = torch.stack([x0_grid.flatten(), x1_grid.flatten()], dim=1)
            diff = x_flat - mu_post_2d
            Sigma_inv = torch.inverse(Sigma_post_2d)
            log_prob = -0.5 * (diff @ Sigma_inv * diff).sum(dim=1)
            log_prob -= 0.5 * torch.logdet(Sigma_post_2d)
            log_prob -= 0.5 * 2 * np.log(2 * np.pi)  # 2D marginal

        # Convert to probability and reshape
        prob = torch.exp(log_prob).reshape(n_grid, n_grid)

        return x0_grid.numpy(), x1_grid.numpy(), prob.numpy()

    def eval_posterior_theta_given_y_on_grid(self, y, theta_range=None, n_grid=100):
        """Evaluate p(theta|y) on a 2D grid for contour plotting.

        Args:
            y: Observation (1, y_dim)
            theta_range: Range for theta grid, e.g., [(-3, 3), (-3, 3)]
            n_grid: Number of grid points per dimension

        Returns:
            theta0_grid, theta1_grid, density_grid
        """
        if theta_range is None:
            theta_range = [(-3, 3), (-3, 3)]

        # Get posterior parameters
        mu_post, Sigma_post = self.true_posterior_theta_given_y(y)
        mu_post = mu_post.squeeze()  # (theta_dim,)

        # Create grid
        theta0 = torch.linspace(theta_range[0][0], theta_range[0][1], n_grid)
        theta1 = torch.linspace(theta_range[1][0], theta_range[1][1], n_grid)
        theta0_grid, theta1_grid = torch.meshgrid(theta0, theta1, indexing="ij")

        # Flatten grid
        theta_flat = torch.stack([theta0_grid.flatten(), theta1_grid.flatten()], dim=1)

        # Compute log probability for each grid point
        diff = theta_flat - mu_post
        Sigma_inv = torch.inverse(Sigma_post)
        log_prob = -0.5 * (diff @ Sigma_inv * diff).sum(dim=1)
        log_prob -= 0.5 * torch.logdet(Sigma_post)
        log_prob -= 0.5 * self.theta_dim * np.log(2 * np.pi)

        # Convert to probability and reshape
        prob = torch.exp(log_prob).reshape(n_grid, n_grid)

        return theta0_grid.numpy(), theta1_grid.numpy(), prob.numpy()


# ============================================================================
# 1b. Theta-Dependent Gaussian Model (y depends on both x and theta)
# ============================================================================


class ThetaDependentGaussianModel:
    """Hierarchical Gaussian model where y depends on both x and theta.

    Model:
    - theta ~ N(0, I)
    - x ~ N(A*theta + b, I)
    - y ~ N(C*x + M*theta + d, Sigma_y)

    This creates a more complex dependency structure where the noisy observation
    y carries direct information about theta, not just through x.
    """

    def __init__(self, theta_dim=2, x_dim=3, y_dim=3, noise_scale=1.0, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.y_dim = y_dim

        # x = A*theta + b + noise_x
        self.A = torch.randn(x_dim, theta_dim)
        self.b = torch.randn(x_dim)

        # y = C*x + M*theta + d + noise_y
        self.C = torch.randn(y_dim, x_dim)
        self.M = torch.randn(y_dim, theta_dim)  # Direct theta dependence
        self.d = torch.randn(y_dim)

        # Noise covariance for y (scaled identity)
        self.noise_scale = noise_scale
        self.Sigma_y_noise = noise_scale * torch.eye(y_dim)

        print(f"Theta-Dependent Model: theta({theta_dim}) -> x({x_dim}) -> y({y_dim})")
        print(f"  x = A*theta + b + N(0, I)")
        print(f"  y = C*x + M*theta + d + N(0, {noise_scale}*I)")
        print(f"  (y depends on BOTH x and theta)")

    def sample(self, n_samples):
        """Sample from the generative model."""
        # theta ~ N(0, I)
        theta = torch.randn(n_samples, self.theta_dim)

        # x ~ N(A*theta + b, I)
        x_mean = theta @ self.A.T + self.b
        x = x_mean + torch.randn(n_samples, self.x_dim)

        # y ~ N(C*x + M*theta + d, Sigma_y_noise)
        y_mean = x @ self.C.T + theta @ self.M.T + self.d
        y = y_mean + self.noise_scale**0.5 * torch.randn(n_samples, self.y_dim)

        return theta, x, y

    def true_posterior_theta_given_x(self, x):
        """Compute true posterior p(theta|x) analytically.

        This is the same as SimpleGaussianModel since x generation is unchanged:
        x = A*theta + b + noise, theta ~ N(0, I), noise ~ N(0, I)

        Posterior: theta|x ~ N(mu_post, Sigma_post)
        """
        device = x.device
        A = self.A.to(device)
        b = self.b.to(device)

        # Posterior covariance: (A^T A + I)^{-1}
        ATA = A.T @ A
        Sigma_post = torch.inverse(ATA + torch.eye(self.theta_dim, device=device))

        # Posterior mean: Sigma_post @ A^T @ (x - b)
        x_centered = x - b
        mu_post = (Sigma_post @ A.T @ x_centered.T).T

        return mu_post, Sigma_post

    def log_prob_theta_given_x(self, theta, x):
        """Compute log p(theta|x) using true posterior."""
        device = x.device
        mu_post, Sigma_post = self.true_posterior_theta_given_x(x)

        # Log probability of multivariate Gaussian
        diff = theta - mu_post
        Sigma_inv = torch.inverse(Sigma_post)
        log_prob = -0.5 * (diff @ Sigma_inv @ diff.T).diag()
        log_prob -= 0.5 * torch.logdet(Sigma_post)
        log_prob -= 0.5 * self.theta_dim * np.log(2 * np.pi)

        return log_prob

    def true_posterior_theta_given_y(self, y):
        """Compute true posterior p(theta|y) analytically.

        For this model:
        - theta ~ N(0, I)
        - x|theta ~ N(A*theta + b, I)
        - y|x,theta ~ N(C*x + M*theta + d, Sigma_noise)

        Marginalizing x out:
        y|theta ~ N((C*A + M)*theta + C*b + d, C*C^T + Sigma_noise)

        Let F = C*A + M (combined linear mapping from theta to y)
        Let c = C*b + d (combined bias)
        Let Sigma_y|theta = C*C^T + Sigma_noise

        Then using standard Gaussian conditioning:
        theta|y ~ N(mu_post, Sigma_post)
        """
        device = y.device
        A = self.A.to(device)
        b = self.b.to(device)
        C = self.C.to(device)
        M = self.M.to(device)
        d = self.d.to(device)
        Sigma_noise = self.Sigma_y_noise.to(device)

        # Combined linear mapping: F = C*A + M
        F = C @ A + M

        # Bias: c = C*b + d
        c = C @ b + d

        # Conditional covariance of y|theta: Sigma_y|theta = C*C^T + Sigma_noise
        Sigma_y_given_theta = C @ C.T + Sigma_noise

        # Marginal covariance of y: Sigma_y = F @ I @ F^T + Sigma_y|theta = F @ F^T + C*C^T + Sigma_noise
        Sigma_y = F @ F.T + Sigma_y_given_theta

        # Cross-covariance: Cov(theta, y) = I @ F^T = F^T
        Sigma_theta_y = F.T

        # Inverse of marginal covariance
        Sigma_y_inv = torch.inverse(Sigma_y)

        # Posterior covariance: Sigma_post = I - F^T @ Sigma_y_inv @ F
        Sigma_post = (
            torch.eye(self.theta_dim, device=device) - Sigma_theta_y @ Sigma_y_inv @ Sigma_theta_y.T
        )

        # Posterior mean: mu_post = F^T @ Sigma_y_inv @ (y - c)
        y_centered = y - c
        mu_post = (Sigma_theta_y @ Sigma_y_inv @ y_centered.T).T

        return mu_post, Sigma_post

    def true_posterior_x_given_y(self, y):
        """Compute true posterior p(x|y) analytically.

        For this model we need to marginalize over theta:
        - theta ~ N(0, I)
        - x|theta ~ N(A*theta + b, I)
        - y|x,theta ~ N(C*x + M*theta + d, Sigma_noise)

        Joint (x, y) after marginalizing theta:
        - x ~ N(b, A*A^T + I)  [marginal of x]
        - y|x ~ requires integrating over theta

        Actually, let's work with the joint (x, y, theta) and marginalize theta:

        [x]     [b]       [A*A^T + I,    A*A^T*C^T + A*M^T]
        [y] ~ N([C*b+d], [C*A*A^T + M*A, C*(A*A^T+I)*C^T + M*M^T + C*A*M^T + M*A^T*C^T + Sigma_noise])

        Cross-covariance Cov(x,y):
        Cov(x,y) = E[x*y^T] - E[x]*E[y]^T
        = E[(A*theta + b + eps_x) * (C*x + M*theta + d + eps_y)^T] - b*(C*b+d)^T
        = E[(A*theta + b + eps_x) * (C*(A*theta + b + eps_x) + M*theta + d + eps_y)^T] - b*(C*b+d)^T

        Let's expand step by step:
        x = A*theta + b + eps_x
        y = C*x + M*theta + d + eps_y = C*A*theta + C*b + C*eps_x + M*theta + d + eps_y
          = (C*A + M)*theta + C*b + d + C*eps_x + eps_y

        Cov(x, y) = Cov(A*theta + eps_x, (C*A+M)*theta + C*eps_x + eps_y)
                  = A * Cov(theta) * (C*A+M)^T + Cov(eps_x) * C^T
                  = A * (C*A+M)^T + C^T
                  = A*A^T*C^T + A*M^T + C^T

        Let Sigma_xx = A*A^T + I
        Let Sigma_xy = A*A^T*C^T + A*M^T + C^T = (A*A^T + I)*C^T + A*M^T = Sigma_xx @ C^T + A @ M^T
        Let Sigma_yy = (C*A+M)*(C*A+M)^T + C*C^T + Sigma_noise
        """
        device = y.device
        A = self.A.to(device)
        b = self.b.to(device)
        C = self.C.to(device)
        M = self.M.to(device)
        d = self.d.to(device)
        Sigma_noise = self.Sigma_y_noise.to(device)

        # Marginal covariance of x
        Sigma_xx = A @ A.T + torch.eye(self.x_dim, device=device)

        # Combined mapping F = C*A + M
        F = C @ A + M

        # Marginal covariance of y
        Sigma_yy = F @ F.T + C @ C.T + Sigma_noise

        # Cross-covariance Cov(x, y)
        Sigma_xy = Sigma_xx @ C.T + A @ M.T

        # Inverse of marginal covariance of y
        Sigma_yy_inv = torch.inverse(Sigma_yy)

        # Marginal mean of y
        mu_y = C @ b + d

        # Posterior mean: mu_x|y = b + Sigma_xy @ Sigma_yy_inv @ (y - mu_y)
        mu_x = b
        y_centered = y - mu_y
        mu_post = mu_x + (Sigma_xy @ Sigma_yy_inv @ y_centered.T).T

        # Posterior covariance: Sigma_x|y = Sigma_xx - Sigma_xy @ Sigma_yy_inv @ Sigma_xy^T
        Sigma_post = Sigma_xx - Sigma_xy @ Sigma_yy_inv @ Sigma_xy.T

        return mu_post, Sigma_post

    def eval_posterior_theta_given_y_on_grid(self, y, theta_range=None, n_grid=100):
        """Evaluate p(theta|y) on a 2D grid for contour plotting."""
        if theta_range is None:
            theta_range = [(-3, 3), (-3, 3)]

        mu_post, Sigma_post = self.true_posterior_theta_given_y(y)
        mu_post = mu_post.squeeze()

        theta0 = torch.linspace(theta_range[0][0], theta_range[0][1], n_grid)
        theta1 = torch.linspace(theta_range[1][0], theta_range[1][1], n_grid)
        theta0_grid, theta1_grid = torch.meshgrid(theta0, theta1, indexing="ij")

        theta_flat = torch.stack([theta0_grid.flatten(), theta1_grid.flatten()], dim=1)

        diff = theta_flat - mu_post
        Sigma_inv = torch.inverse(Sigma_post)
        log_prob = -0.5 * (diff @ Sigma_inv * diff).sum(dim=1)
        log_prob -= 0.5 * torch.logdet(Sigma_post)
        log_prob -= 0.5 * self.theta_dim * np.log(2 * np.pi)

        prob = torch.exp(log_prob).reshape(n_grid, n_grid)

        return theta0_grid.numpy(), theta1_grid.numpy(), prob.numpy()

    def eval_posterior_x_given_y_on_grid(self, y, x_range=None, n_grid=100):
        """Evaluate p(x|y) on a 2D grid for contour plotting (x[0] vs x[1])."""
        if x_range is None:
            x_range = [(-4, 4), (-4, 4)]

        mu_post, Sigma_post = self.true_posterior_x_given_y(y)
        mu_post = mu_post.squeeze()

        x0 = torch.linspace(x_range[0][0], x_range[0][1], n_grid)
        x1 = torch.linspace(x_range[1][0], x_range[1][1], n_grid)
        x0_grid, x1_grid = torch.meshgrid(x0, x1, indexing="ij")

        if self.x_dim == 2:
            x_flat = torch.stack([x0_grid.flatten(), x1_grid.flatten()], dim=1)
            diff = x_flat - mu_post
            Sigma_inv = torch.inverse(Sigma_post)
            log_prob = -0.5 * (diff @ Sigma_inv * diff).sum(dim=1)
            log_prob -= 0.5 * torch.logdet(Sigma_post)
            log_prob -= 0.5 * self.x_dim * np.log(2 * np.pi)
        else:
            # Marginalize to first 2 dimensions
            mu_post_2d = mu_post[:2]
            Sigma_post_2d = Sigma_post[:2, :2]

            x_flat = torch.stack([x0_grid.flatten(), x1_grid.flatten()], dim=1)
            diff = x_flat - mu_post_2d
            Sigma_inv = torch.inverse(Sigma_post_2d)
            log_prob = -0.5 * (diff @ Sigma_inv * diff).sum(dim=1)
            log_prob -= 0.5 * torch.logdet(Sigma_post_2d)
            log_prob -= 0.5 * 2 * np.log(2 * np.pi)

        prob = torch.exp(log_prob).reshape(n_grid, n_grid)

        return x0_grid.numpy(), x1_grid.numpy(), prob.numpy()


# ============================================================================
# 2. Data Standardization
# ============================================================================


class Standardizer:
    """Standardize data to zero mean and unit variance."""

    def __init__(self, data):
        """Compute mean and std from training data.

        Args:
            data: torch.Tensor of shape (n_samples, dim)
        """
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True).clamp(min=1e-6)  # Avoid division by zero

    def standardize(self, data):
        """Transform data to standardized space."""
        mean = self.mean.to(data.device)
        std = self.std.to(data.device)
        return (data - mean) / std

    def unstandardize(self, data):
        """Transform data back to original space."""
        mean = self.mean.to(data.device)
        std = self.std.to(data.device)
        return data * std + mean


# ============================================================================
# 3. Linear Source Flow: T_φ(ε, y) → x (for Gaussian posteriors)
# ============================================================================


class LinearSourceFlow(nn.Module):
    """Linear source flow for Gaussian posteriors: x = L(y)*ε + μ(y).

    Since p(x|y) is Gaussian for our hierarchical model, we can use a
    linear transformation that outputs x ~ N(μ(y), Σ(y)) where Σ = LL^T.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_dims=[64, 64]):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        # MLP to compute μ(y) and L(y) from y
        layers = []
        input_dim = y_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ELU())
            input_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output mean μ(y)
        self.mean_layer = nn.Linear(input_dim, x_dim)

        # Output Cholesky factor L(y) - lower triangular
        self.n_tril_params = x_dim * (x_dim + 1) // 2
        self.chol_layer = nn.Linear(input_dim, self.n_tril_params)

    def forward(self, y: torch.Tensor):
        """Return a distribution object that can transform ε → x.

        Args:
            y: Conditioning variable (batch_size, y_dim)

        Returns:
            LinearFlowDistribution that supports .transform(eps) and .sample()
        """
        h = self.encoder(y)
        mean = self.mean_layer(h)
        chol_params = self.chol_layer(h)

        # Build Cholesky factor
        batch_size = y.shape[0]
        L = torch.zeros(batch_size, self.x_dim, self.x_dim, device=y.device, dtype=y.dtype)
        tril_indices = torch.tril_indices(self.x_dim, self.x_dim, device=y.device)
        L[:, tril_indices[0], tril_indices[1]] = chol_params

        # Ensure positive diagonal
        diag_indices = torch.arange(self.x_dim, device=y.device)
        L[:, diag_indices, diag_indices] = (
            torch.nn.functional.softplus(L[:, diag_indices, diag_indices]) + 1e-4
        )

        return LinearFlowDistribution(mean, L, self.x_dim)


class LinearFlowDistribution:
    """Wrapper to make LinearSourceFlow compatible with NSF interface."""

    def __init__(self, mean: torch.Tensor, L: torch.Tensor, x_dim: int):
        self.mean = mean  # (batch_size, x_dim)
        self.L = L  # (batch_size, x_dim, x_dim) Cholesky factor
        self.x_dim = x_dim
        # Create a mock transform object for compatibility
        self.transform = self

    @property
    def domain(self):
        """Mock domain for compatibility with sample_from_source."""

        class MockDomain:
            def __init__(self, x_dim):
                self.event_shape = (x_dim,)

        return MockDomain(self.x_dim)

    def __call__(self, eps: torch.Tensor) -> torch.Tensor:
        """Transform ε → x: x = L*ε + μ."""
        # eps: (batch_size, x_dim)
        # L: (batch_size, x_dim, x_dim)
        # mean: (batch_size, x_dim)
        x = torch.bmm(self.L, eps.unsqueeze(-1)).squeeze(-1) + self.mean
        return x

    def sample(self, shape: tuple) -> torch.Tensor:
        """Sample x by sampling ε ~ N(0,I) and transforming."""
        n_samples = shape[0]
        batch_size = self.mean.shape[0]
        device = self.mean.device

        # Sample ε ~ N(0, I)
        eps = torch.randn(n_samples, batch_size, self.x_dim, device=device)

        # Transform each sample
        # Expand mean and L for n_samples
        mean_expanded = self.mean.unsqueeze(0).expand(n_samples, -1, -1)
        L_expanded = self.L.unsqueeze(0).expand(n_samples, -1, -1, -1)

        # x = L*ε + μ
        x = torch.einsum("nbij,nbj->nbi", L_expanded, eps) + mean_expanded
        return x


# ============================================================================
# 3. Epsilon Encoder: q_ψ(ε|θ,y)
# ============================================================================


class EpsilonEncoder(nn.Module):
    """Parametrized distribution q_ψ(ε|θ,y) for sampling epsilon during training.

    This learns to map (θ, y) to a Gaussian distribution over ε with full covariance,
    which is used during training instead of sampling from the standard Gaussian prior p(ε).
    """

    def __init__(self, theta_dim: int, y_dim: int, epsilon_dim: int, hidden_dims=[64, 64]):
        """Initialize epsilon encoder with full covariance.

        Args:
            theta_dim: Dimension of theta
            y_dim: Dimension of y
            epsilon_dim: Dimension of epsilon (should equal x_dim)
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()

        self.theta_dim = theta_dim
        self.y_dim = y_dim
        self.epsilon_dim = epsilon_dim

        # Build MLP: (θ, y) → (μ_ε, L_ε) where L is Cholesky factor
        layers = []
        input_dim = theta_dim + y_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ELU())
            input_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output layer for mean
        self.mean_layer = nn.Linear(input_dim, epsilon_dim)

        # Output layer for Cholesky factor (lower triangular matrix)
        # Number of parameters needed: d + d(d-1)/2 = d(d+1)/2
        self.n_tril_params = epsilon_dim * (epsilon_dim + 1) // 2
        self.chol_layer = nn.Linear(input_dim, self.n_tril_params)

    def forward(self, theta: torch.Tensor, y: torch.Tensor):
        """Compute distribution q_ψ(ε|θ,y) with full covariance.

        Args:
            theta: Parameters (batch_size, theta_dim)
            y: Noisy observations (batch_size, y_dim)

        Returns:
            MultivariateNormal distribution over ε
        """
        # Concatenate inputs
        inputs = torch.cat([theta, y], dim=-1)

        # Encode
        h = self.encoder(inputs)

        # Get mean
        mean = self.mean_layer(h)

        # Get Cholesky factor parameters
        chol_params = self.chol_layer(h)

        # Construct lower triangular matrix L such that Σ = LL^T
        batch_size = theta.shape[0]
        L = torch.zeros(
            batch_size, self.epsilon_dim, self.epsilon_dim, device=theta.device, dtype=theta.dtype
        )

        # Fill in the lower triangular part
        tril_indices = torch.tril_indices(self.epsilon_dim, self.epsilon_dim, device=theta.device)
        L[:, tril_indices[0], tril_indices[1]] = chol_params

        # Ensure diagonal elements are positive (for positive definiteness)
        # Apply softplus to diagonal elements
        diag_indices = torch.arange(self.epsilon_dim, device=theta.device)
        L[:, diag_indices, diag_indices] = (
            torch.nn.functional.softplus(L[:, diag_indices, diag_indices]) + 1e-4
        )  # Add small constant for numerical stability

        # Return multivariate Gaussian with scale_tril=L (Cholesky factor)
        return torch.distributions.MultivariateNormal(loc=mean, scale_tril=L)

    def sample(self, theta: torch.Tensor, y: torch.Tensor):
        """Sample ε ~ q_ψ(ε|θ,y).

        Args:
            theta: Parameters (batch_size, theta_dim)
            y: Noisy observations (batch_size, y_dim)

        Returns:
            epsilon: Sampled epsilon (batch_size, epsilon_dim)
        """
        dist = self.forward(theta, y)
        return dist.rsample()  # Reparametrized sample

    def log_prob(self, epsilon: torch.Tensor, theta: torch.Tensor, y: torch.Tensor):
        """Compute log q_ψ(ε|θ,y).

        Args:
            epsilon: Epsilon values (batch_size, epsilon_dim)
            theta: Parameters (batch_size, theta_dim)
            y: Noisy observations (batch_size, y_dim)

        Returns:
            log_prob: (batch_size,)
        """
        dist = self.forward(theta, y)
        return dist.log_prob(epsilon)


# ============================================================================
# 4. Training Functions
# ============================================================================


def train_source_elbo_simple(
    model,
    theta,
    y,
    theta_standardizer: Standardizer,
    epochs=100,
    lr=1e-5,
    batch_size=256,
    device="cpu",
    patience=10,
    train_size=0.8,
    source_flow_type="nsf",
):
    """Train source flow p(x|y) using ELBO with regularization.

    This version learns:
    1. q_ψ(ε|θ,y): Distribution to sample ε from during training
    2. T_ψ(ε,y) → x: Source flow transformation

    The ELBO objective with regularization is:
    L = E_{ε~q_ψ(ε|θ,y)}[log p(θ|x=T_ψ(ε,y)) + log p(ε) - β*log q_ψ(ε|θ,y)]
      + λ_std * penalty(σ_ψ < σ_min)
      + λ_div * diversity_loss(x)

    Args:
        model: SimpleGaussianModel
        theta: Training parameters
        y: Training observations
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        device: Device to train on
        beta_kl: Weight for KL term (β-VAE style, default 1.0)
        min_std: Minimum std to encourage (default 0.1)
        lambda_std: Weight for minimum std penalty (default 0.1)
        lambda_diversity: Weight for diversity regularization (default 0.01)
        use_kl_annealing: If True, anneal β from 0 to beta_kl
        patience: Early stopping patience (default 10 epochs)
        train_size: Fraction of data for training vs validation (default 0.8)

    Regularization terms:
    1. β-VAE weighting: Allows controlling KL term influence
    2. Minimum std penalty: Prevents variance collapse
    3. Diversity loss: Encourages diverse outputs in x

    Early stopping:
    - Monitors validation loss
    - Stops if no improvement for 'patience' epochs
    - Restores best model weights

    Args:
        source_flow_type: "nsf" for Neural Spline Flow, "linear" for linear Gaussian
    """
    print("\n" + "=" * 60)
    print("Training Source Flow with ELBO + Regularization")
    print("=" * 60)
    print(f"Source flow type: {source_flow_type}")
    print("Learning q_ψ(ε|θ,y) and source flow T_ψ(ε,y) → x")

    # Create epsilon encoder: (θ, y) -> q_ψ(ε|θ,y)
    epsilon_encoder = EpsilonEncoder(
        theta_dim=model.theta_dim,
        y_dim=model.y_dim,
        epsilon_dim=model.x_dim,
        hidden_dims=[64, 64],
    ).to(device)

    # Create source flow: (eps, y) -> x
    if source_flow_type == "linear":
        source_flow = LinearSourceFlow(
            x_dim=model.x_dim,
            y_dim=model.y_dim,
            hidden_dims=[64, 64],
        ).to(device)
    else:  # nsf
        source_flow = NSF(
            features=model.x_dim,
            context=model.y_dim,
            transforms=3,
            hidden_features=[64, 64],
        ).to(device)

    # Optimize both networks jointly
    optimizer = torch.optim.Adam(
        list(source_flow.parameters()) + list(epsilon_encoder.parameters()), lr=lr
    )

    # Split into train/val
    n_train = int(len(theta) * train_size)
    theta_train, theta_val = theta[:n_train], theta[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    train_dataset = TensorDataset(theta_train, y_train)
    val_dataset = TensorDataset(theta_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(theta_train)} samples, Val: {len(theta_val)} samples")

    # Track training progress
    losses = []
    val_losses = []
    elbo_terms = []
    kl_terms = []

    # Early stopping
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_source_flow_state = None
    best_epsilon_encoder_state = None

    pbar = trange(epochs, desc="ELBO Training")

    for epoch in pbar:
        epoch_losses = []
        epoch_elbos = []
        epoch_kls = []
        epoch_regs = []

        for theta_batch, y_batch in train_loader:
            theta_batch = theta_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # 1. Get distribution parameters from encoder
            dist = epsilon_encoder(theta_batch, y_batch)

            # 2. Sample ε ~ q_ψ(ε|θ,y) using reparameterization trick
            eps = dist.rsample()

            # 3. Generate x = T_ψ(ε, y) via source flow
            x_generated = source_flow(y_batch).transform(eps)

            # 4. Evaluate log p(θ|x_generated) using true posterior
            theta_batch_unstd = theta_standardizer.unstandardize(theta_batch)
            log_p_theta_given_x = model.log_prob_theta_given_x(theta_batch_unstd, x_generated)

            # 5. Compute log p(ε) under standard Gaussian prior
            log_p_eps = -0.5 * eps.pow(2).sum(-1) - 0.5 * model.x_dim * np.log(2 * np.pi)

            # 6. Compute log q_ψ(ε|θ,y)
            log_q_eps = dist.log_prob(eps)

            # 7. ELBO with β-weighting
            #    ELBO = E[log p(θ|x) + log p(ε) - β*log q(ε|θ,y)]
            reconstruction_term = log_p_theta_given_x.mean()
            kl_div = (log_q_eps - log_p_eps).mean()

            elbo = reconstruction_term - kl_div

            # 9. Total loss
            loss = -elbo
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(source_flow.parameters()) + list(epsilon_encoder.parameters()), 5.0
            )
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_elbos.append(elbo.item())
            epoch_kls.append(kl_div.item())

        avg_loss = np.mean(epoch_losses)
        avg_elbo = np.mean(epoch_elbos)
        avg_kl = np.mean(epoch_kls)

        losses.append(avg_loss)
        elbo_terms.append(avg_elbo)
        kl_terms.append(avg_kl)

        # Validation phase
        source_flow.eval()
        epsilon_encoder.eval()
        val_epoch_losses = []
        val_epoch_elbos = []
        val_epoch_kls = []

        with torch.no_grad():
            for theta_batch, y_batch in val_loader:
                theta_batch = theta_batch.to(device)
                y_batch = y_batch.to(device)

                # Same forward pass as training
                dist = epsilon_encoder(theta_batch, y_batch)
                eps = dist.rsample()
                x_generated = source_flow(y_batch).transform(eps)

                # Compute ELBO
                theta_batch_unstd = theta_standardizer.unstandardize(theta_batch)
                log_p_theta_given_x = model.log_prob_theta_given_x(theta_batch_unstd, x_generated)
                log_p_eps = -0.5 * eps.pow(2).sum(-1) - 0.5 * model.x_dim * np.log(2 * np.pi)
                log_q_eps = dist.log_prob(eps)
                kl_div = (log_q_eps - log_p_eps).mean()

                elbo = log_p_theta_given_x.mean() - kl_div
                loss = -elbo  # No regularization on validation

                val_epoch_losses.append(loss.item())
                val_epoch_elbos.append(elbo.item())
                val_epoch_kls.append(kl_div.item())

        avg_val_loss = np.mean(val_epoch_losses)
        avg_val_elbo = np.mean(val_epoch_elbos)
        avg_val_kl = np.mean(val_epoch_kls)
        val_losses.append(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            best_source_flow_state = source_flow.state_dict()
            best_epsilon_encoder_state = epsilon_encoder.state_dict()
        else:
            patience_counter += 1

        pbar.set_postfix(
            {
                "loss": f"{avg_loss:.4f}",
                "val_loss": f"{avg_val_loss:.4f}",
                "elbo": f"{avg_elbo:.4f}",
                "kl": f"{avg_kl:.4f}",
            }
        )

        # Check early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

        source_flow.train()
        epsilon_encoder.train()

    print(f"✓ ELBO training complete. Final loss: {losses[-1]:.4f}")
    print(f"  Final ELBO: {elbo_terms[-1]:.4f}, Final KL: {kl_terms[-1]:.4f}")

    # Restore best model weights
    if best_source_flow_state is not None:
        source_flow.load_state_dict(best_source_flow_state)
        epsilon_encoder.load_state_dict(best_epsilon_encoder_state)
        print(f"  Restored best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    return source_flow, epsilon_encoder, losses


def train_source_fm_simple(
    model,
    x,
    y,
    epochs=100,
    lr=1e-4,
    batch_size=256,
    device="cpu",
    cache_path="logs/elbo_source/fm_model_cache.pt",
):
    """Train source flow p(x|y) using Flow Matching.

    Args:
        cache_path: Path to cache trained model. If exists, loads from cache.
    """
    print("\n" + "=" * 60)
    print("Training Source Flow with FLOW MATCHING")
    print("=" * 60)

    # Create flow matching model
    fm_config = {
        "drift": {
            "architecture": "cfnet",
            "posterior_kwargs": {
                "input_dim": model.x_dim,
                "context_dim": model.y_dim,
                "theta_with_glue": True,
                "hidden_dims": [64, 64, 64],
                "dropout": 0.5,
                "batch_norm": True,
                "activation": "elu",
            },
        }
    }

    flow_matching = FlowMatching(
        conditional=True,
        probability_path="ot2",
        prior="uniform",
        base_dist="gaussian",
        dim=x.shape[1:],
        cond_dim=y.shape[1:],
        **fm_config,
    ).to(device)

    # No rescaling
    flow_matching.set_scales(x, y, "none")

    # Check for cached model (only if cache_path is provided)
    if cache_path is not None and os.path.exists(cache_path):
        print(f"Loading cached FM model from {cache_path}")
        checkpoint = torch.load(cache_path, map_location=device, weights_only=False)
        flow_matching.load_state_dict(checkpoint["model_state_dict"])
        losses = checkpoint["losses"]
        print(f"✓ Loaded FM model (final loss: {losses[-1]:.4f})")
        return flow_matching, losses

    optimizer = torch.optim.Adam(flow_matching.parameters(), lr=lr)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    pbar = trange(epochs, desc="FM Training")

    for epoch in pbar:
        epoch_losses = []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            time = flow_matching.sample_time(x_batch.size(0)).to(device)

            source = flow_matching.sample_source(y_batch).to(device)

            x_t = flow_matching.interpolant(time, source, x_batch)

            v = flow_matching.target(source, x_batch)

            pred = flow_matching(x_t, y_batch, time)

            loss = (pred - v).pow(2).mean()
            optimizer.zero_grad()

            # Flow matching loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(flow_matching.parameters(), 5.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    print(f"✓ Flow Matching training complete. Final loss: {losses[-1]:.4f}")

    # Save to cache (only if cache_path is provided)
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": flow_matching.state_dict(),
                "losses": losses,
            },
            cache_path,
        )
        print(f"✓ Saved FM model to {cache_path}")

    return flow_matching, losses


# ============================================================================
# 4. Sampling and Visualization
# ============================================================================


def sample_from_source(source_flow, y, n_samples, device="cpu"):
    """Sample x from source flow given y.

    At inference time, we sample ε ~ p(ε) = N(0,I) from the prior,
    NOT from q_ψ(ε|θ,y) which is only used during training.
    Then we transform: x = T_ψ(ε, y)

    Works with both NSF (Zuko) and LinearSourceFlow.
    """
    source_flow.eval()

    with torch.no_grad():
        y = y.to(device)
        # Sample ε from prior p(ε) = N(0, I)
        eps = torch.randn(n_samples, y.shape[0], y.shape[1], device=device)
        # Transform: x = T_ψ(ε, y)
        # Expand y for n_samples: (1, y_dim) -> (n_samples, y_dim)
        y_expanded = y.expand(n_samples, -1, -1).reshape(n_samples * y.shape[0], -1)
        eps_flat = eps.reshape(n_samples * y.shape[0], -1)

        dist = source_flow(y_expanded)
        x_samples = dist.transform(eps_flat)

        x_samples = x_samples.reshape(n_samples, y.shape[0], -1)
    return x_samples


def sample_from_fm(flow_matching, y, n_samples, device="cpu"):
    """Sample x from flow matching given y."""
    flow_matching.eval()
    with torch.no_grad():
        y = y.to(device)
        # Sample from flow matching
        source = flow_matching.sample_source(y, n_samples).to(device)
        broadcast_shape = (n_samples, y.shape[0], *([-1] * (y.dim() - 1)))
        cond = y.unsqueeze(0).expand(*broadcast_shape)  # shape (nsamples, N, *obs_dim)
        x_samples = flow_matching.sample(
            source.reshape(-1, *source.shape[2:]),
            cond.reshape(-1, *cond.shape[2:]),
            device,
            only_last=True,
        )
    return x_samples


# ============================================================================
# 5. Evaluation Metrics
# ============================================================================


def compute_log_posterior_probability(
    model,
    theta_samples: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Compute Log Posterior Probability (LPP).

    LPP = E_{y, theta_sampled}[log p_true(theta_sampled | y)]

    This measures how well the sampled thetas match the true posterior.
    Higher is better.

    Args:
        model: Gaussian model with true_posterior_theta_given_y method
        theta_samples: Samples from the method (n_samples, n_obs, theta_dim)
        y: Observations (n_obs, y_dim)

    Returns:
        Mean log posterior probability
    """
    # Ensure all tensors are on CPU
    theta_samples = theta_samples.cpu()
    y = y.cpu()

    n_samples, n_obs, theta_dim = theta_samples.shape

    # Get true posterior parameters for each y (on CPU)
    mu_post, Sigma_post = model.true_posterior_theta_given_y(y)
    # mu_post: (n_obs, theta_dim), Sigma_post: (theta_dim, theta_dim)

    Sigma_inv = torch.inverse(Sigma_post)
    log_det = torch.logdet(Sigma_post)

    total_log_prob = 0.0

    for i in range(n_samples):
        # theta_samples[i]: (n_obs, theta_dim)
        diff = theta_samples[i] - mu_post  # (n_obs, theta_dim)

        # Compute log prob for each observation
        # log p(theta|y) = -0.5 * (theta - mu)^T Sigma^{-1} (theta - mu) - 0.5 * log|Sigma| - d/2 * log(2*pi)
        mahalanobis = (diff @ Sigma_inv * diff).sum(dim=1)  # (n_obs,)
        log_prob = -0.5 * mahalanobis - 0.5 * log_det - 0.5 * theta_dim * np.log(2 * np.pi)

        total_log_prob += log_prob.mean().item()

    return total_log_prob / n_samples


def compute_c2st(
    theta_true: torch.Tensor,
    theta_method: torch.Tensor,
    n_samples: int = 5000,
) -> float:
    """Compute Classifier 2-Sample Test (C2ST) using SBI library.

    C2ST trains a classifier to distinguish between samples from two distributions.
    Returns accuracy: 0.5 means distributions are identical, 1.0 means perfectly distinguishable.

    Args:
        theta_true: Samples from true posterior (n_samples, theta_dim)
        theta_method: Samples from method's distribution (n_samples, theta_dim)
        n_samples: Number of samples to use for C2ST

    Returns:
        C2ST accuracy (0.5 = identical, 1.0 = perfectly distinguishable)
    """
    # Ensure CPU tensors for C2ST
    theta_true = theta_true.cpu()
    theta_method = theta_method.cpu()

    # Limit to n_samples
    n = min(n_samples, len(theta_true), len(theta_method))
    theta_true = theta_true[:n]
    theta_method = theta_method[:n]

    # Ensure 2D tensors
    if theta_true.dim() == 1:
        theta_true = theta_true.unsqueeze(1)
    if theta_method.dim() == 1:
        theta_method = theta_method.unsqueeze(1)

    # Use SBI's c2st (runs on CPU)
    accuracy = c2st(theta_true, theta_method)

    return accuracy.mean().item()


def evaluate_method(
    model,
    source_flow,
    y_test: torch.Tensor,
    theta_std: "Standardizer",
    x_std: "Standardizer",
    y_std: "Standardizer",
    n_samples: int = 5000,
    device: str = "cpu",
    method_name: str = "ELBO",
    is_flow_matching: bool = False,
    batch_size: int = 100,  # Batch size for sampling to avoid OOM
) -> Dict[str, float]:
    """Evaluate a source learning method with LPP and C2ST metrics.

    Args:
        model: Gaussian model
        source_flow: Trained source flow (ELBO or FM)
        y_test: Test observations (original space)
        theta_std, x_std, y_std: Standardizers
        n_samples: Number of samples for evaluation
        device: Device
        method_name: Name for logging
        is_flow_matching: Whether this is a FlowMatching model (vs NSF)
        batch_size: Number of samples per batch to avoid GPU OOM

    Returns:
        Dictionary with LPP and C2ST metrics
    """
    print(f"\nEvaluating {method_name}...")

    # Ensure y_test is on CPU for metric computation
    y_test_cpu = y_test.cpu()
    n_obs = y_test.shape[0]

    # Sample x from source flow in batches to avoid OOM
    print(f"  Sampling {n_samples} x from source flow (batch_size={batch_size})...")
    theta_samples_list = []

    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, n_samples)
        current_batch_size = batch_end - batch_start

        # Standardize y and move to device for sampling
        y_test_std = y_std.standardize(y_test_cpu).to(device)

        # Sample x from source flow (on GPU)
        with torch.no_grad():
            if is_flow_matching:
                x_samples_std = sample_from_fm(source_flow, y_test_std, current_batch_size, device)
                x_samples_std = x_samples_std.reshape(current_batch_size, n_obs, -1)
            else:
                x_samples_std = sample_from_source(
                    source_flow, y_test_std, current_batch_size, device
                )

        # Move to CPU immediately and unstandardize
        x_samples_std_cpu = x_samples_std.cpu()
        del x_samples_std  # Free GPU memory
        if device != "cpu":
            torch.cuda.empty_cache()

        x_samples = x_std.unstandardize(x_samples_std_cpu.reshape(-1, x_samples_std_cpu.shape[-1]))
        x_samples = x_samples.reshape(current_batch_size, n_obs, -1)

        # Sample theta from p(theta|x) for each x (all on CPU)
        for i in range(current_batch_size):
            mu_post, Sigma_post = model.true_posterior_theta_given_x(x_samples[i])
            # Sample from Gaussian posterior
            L = torch.linalg.cholesky(Sigma_post)
            eps = torch.randn(n_obs, model.theta_dim)
            theta_sampled = mu_post + (eps @ L.T)
            theta_samples_list.append(theta_sampled)

    theta_samples = torch.stack(theta_samples_list, dim=0)  # (n_samples, n_obs, theta_dim)

    # Compute LPP (all on CPU)
    print(f"  Computing LPP...")
    lpp = compute_log_posterior_probability(model, theta_samples, y_test_cpu)

    # Sample from true posterior for C2ST (all on CPU)
    print(f"  Sampling from true posterior for C2ST...")
    mu_true, Sigma_true = model.true_posterior_theta_given_y(y_test_cpu)
    L_true = torch.linalg.cholesky(Sigma_true)

    # Sample n_samples thetas for each y, then flatten
    theta_true_list = []
    for i in range(n_samples):
        eps = torch.randn(n_obs, model.theta_dim)
        theta_true_i = mu_true + (eps @ L_true.T)
        theta_true_list.append(theta_true_i)
    theta_true = torch.stack(theta_true_list, dim=0)  # (n_samples, n_obs, theta_dim)

    # Flatten for C2ST: concatenate all samples across observations
    theta_method_flat = theta_samples[:, 0, :]
    theta_true_flat = theta_true[:, 0, :]

    # Compute C2ST
    print(f"  Computing C2ST...")
    c2st_score = compute_c2st(theta_true_flat, theta_method_flat, n_samples=5000)

    metrics = {
        "LPP": lpp,
        "C2ST": c2st_score,
    }

    print(f"  {method_name} Results:")
    print(f"    LPP:  {lpp:.4f} (higher is better)")
    print(f"    C2ST: {c2st_score:.4f} (0.5 = perfect, 1.0 = completely different)")

    return metrics


def visualize_results(
    model,
    source_flow_elbo,
    flow_matching,
    y_test,
    n_samples=500,
    device="cpu",
    theta_std=None,
    x_std=None,
    y_std=None,
    model_type="simple",
):
    """Visualize samples from both methods against true posterior p(theta|y).

    Args:
        model: SimpleGaussianModel or ThetaDependentGaussianModel
        source_flow_elbo: Trained ELBO source flow
        flow_matching: Trained Flow Matching model
        y_test: Test observations
        n_samples: Number of samples to generate
        device: Device to use
        theta_std: Standardizer for theta (optional)
        x_std: Standardizer for x (optional)
        y_std: Standardizer for y (optional)
        model_type: "simple" or "theta_dependent" (for output file naming)
    """
    print("\n" + "=" * 60)
    print("Visualization")
    print("=" * 60)

    # Take first test point
    y_single = y_test[35].unsqueeze(0)  # (1, y_dim)

    # Standardize y if standardizer provided
    if y_std is not None:
        y_single_std = y_std.standardize(y_single)
    else:
        y_single_std = y_single

    # Sample from both methods (in standardized space)
    print("Sampling from ELBO source flow...")
    x_samples_elbo_std = sample_from_source(source_flow_elbo, y_single_std, n_samples, device)
    x_samples_elbo_std = x_samples_elbo_std.squeeze(1)

    # De-standardize x samples
    if x_std is not None:
        x_samples_elbo = x_std.unstandardize(x_samples_elbo_std).cpu().numpy()
    else:
        x_samples_elbo = x_samples_elbo_std.cpu().numpy()

    print("Sampling from Flow Matching...")
    x_samples_fm_std = sample_from_fm(flow_matching, y_single_std, n_samples, device)

    # De-standardize x samples
    if x_std is not None:
        x_samples_fm = x_std.unstandardize(x_samples_fm_std).cpu().numpy()
    else:
        x_samples_fm = x_samples_fm_std.cpu().numpy()

    # Get corresponding theta via true posterior p(theta|x)
    # Move everything to CPU for visualization
    print("Sampling theta via true posterior p(theta|x)...")
    x_elbo_cpu = torch.from_numpy(x_samples_elbo)
    mu_post_elbo, Sigma_post = model.true_posterior_theta_given_x(x_elbo_cpu)
    mu_post_elbo = mu_post_elbo.cpu()
    Sigma_post = Sigma_post.cpu()
    theta_samples_elbo = (
        mu_post_elbo + torch.randn(n_samples, model.theta_dim) @ torch.linalg.cholesky(Sigma_post).T
    )
    theta_samples_elbo = theta_samples_elbo.numpy()

    x_fm_cpu = torch.from_numpy(x_samples_fm)
    mu_post_fm, _ = model.true_posterior_theta_given_x(x_fm_cpu)
    mu_post_fm = mu_post_fm.cpu()
    theta_samples_fm = (
        mu_post_fm + torch.randn(n_samples, model.theta_dim) @ torch.linalg.cholesky(Sigma_post).T
    )
    theta_samples_fm = theta_samples_fm.numpy()

    # Compute true posterior p(theta|y) and sample from it
    print("Computing true posterior p(theta|y)...")
    y_single_cpu = y_single.cpu()
    mu_true_theta, Sigma_true_theta = model.true_posterior_theta_given_y(y_single_cpu)
    mu_true_theta = mu_true_theta.cpu()
    Sigma_true_theta = Sigma_true_theta.cpu()
    theta_samples_true = (
        mu_true_theta
        + torch.randn(n_samples, model.theta_dim) @ torch.linalg.cholesky(Sigma_true_theta).T
    )
    theta_samples_true = theta_samples_true.numpy()

    # Compute true posterior p(x|y) and sample from it
    print("Computing true posterior p(x|y)...")
    mu_true_x, Sigma_true_x = model.true_posterior_x_given_y(y_single_cpu)
    mu_true_x = mu_true_x.cpu()
    Sigma_true_x = Sigma_true_x.cpu()
    x_samples_true = (
        mu_true_x + torch.randn(n_samples, model.x_dim) @ torch.linalg.cholesky(Sigma_true_x).T
    )
    x_samples_true = x_samples_true.numpy()

    # Evaluate posteriors on grid for contour plots (use CPU tensors)
    print("Evaluating p(theta|y) on grid...")
    theta0_grid, theta1_grid, theta_density_grid = model.eval_posterior_theta_given_y_on_grid(
        y_single_cpu
    )

    print("Evaluating p(x|y) on grid...")
    x0_grid, x1_grid, x_density_grid = model.eval_posterior_x_given_y_on_grid(y_single_cpu)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle(f"Source Learning Comparison (y = {y_single[0].cpu().numpy()})", fontsize=14)

    # Row 1: x samples with true p(x|y) contours
    # ELBO samples with contours
    axes[0, 0].contour(
        x0_grid, x1_grid, x_density_grid, levels=8, colors="black", alpha=0.4, linewidths=1
    )
    axes[0, 0].scatter(
        x_samples_elbo[:, 0], x_samples_elbo[:, 1], alpha=0.5, s=15, c="blue", label="ELBO samples"
    )
    axes[0, 0].set_xlabel("x[0]")
    axes[0, 0].set_ylabel("x[1]")
    axes[0, 0].set_title("ELBO: x samples vs true p(x|y)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # FM samples with contours
    axes[0, 1].contour(
        x0_grid, x1_grid, x_density_grid, levels=8, colors="black", alpha=0.4, linewidths=1
    )
    axes[0, 1].scatter(
        x_samples_fm[:, 0], x_samples_fm[:, 1], alpha=0.5, s=15, c="orange", label="FM samples"
    )
    axes[0, 1].set_xlabel("x[0]")
    axes[0, 1].set_ylabel("x[1]")
    axes[0, 1].set_title("Flow Matching: x samples vs true p(x|y)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot x distributions (histograms) - compare to true p(x|y)
    axes[0, 2].hist(
        x_samples_elbo[:, 0], bins=30, alpha=0.6, label="ELBO", density=True, color="blue"
    )
    axes[0, 2].hist(
        x_samples_fm[:, 0], bins=30, alpha=0.6, label="FM", density=True, color="orange"
    )
    axes[0, 2].hist(
        x_samples_true[:, 0], bins=30, alpha=0.6, label="True p(x|y)", density=True, color="green"
    )
    axes[0, 2].set_xlabel("x[0]")
    axes[0, 2].set_ylabel("Density")
    axes[0, 2].set_title("x[0] Distribution")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[0, 3].hist(
        x_samples_elbo[:, 1], bins=30, alpha=0.6, label="ELBO", density=True, color="blue"
    )
    axes[0, 3].hist(
        x_samples_fm[:, 1], bins=30, alpha=0.6, label="FM", density=True, color="orange"
    )
    axes[0, 3].hist(
        x_samples_true[:, 1], bins=30, alpha=0.6, label="True p(x|y)", density=True, color="green"
    )
    axes[0, 3].set_xlabel("x[1]")
    axes[0, 3].set_ylabel("Density")
    axes[0, 3].set_title("x[1] Distribution")
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)

    # Row 2: theta samples with true posterior contours
    # ELBO samples with contours
    contour = axes[1, 0].contour(
        theta0_grid,
        theta1_grid,
        theta_density_grid,
        levels=8,
        colors="black",
        alpha=0.4,
        linewidths=1,
    )
    axes[1, 0].contourf(
        theta0_grid, theta1_grid, theta_density_grid, levels=20, cmap="Greys", alpha=0.3
    )
    axes[1, 0].scatter(
        theta_samples_elbo[:, 0],
        theta_samples_elbo[:, 1],
        alpha=0.5,
        s=15,
        c="blue",
        label="ELBO samples",
    )
    axes[1, 0].set_xlabel("θ[0]")
    axes[1, 0].set_ylabel("θ[1]")
    axes[1, 0].set_title("ELBO: θ samples vs true p(θ|y)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # FM samples with contours
    axes[1, 1].contour(
        theta0_grid,
        theta1_grid,
        theta_density_grid,
        levels=8,
        colors="black",
        alpha=0.4,
        linewidths=1,
    )
    axes[1, 1].contourf(
        theta0_grid, theta1_grid, theta_density_grid, levels=20, cmap="Greys", alpha=0.3
    )
    axes[1, 1].scatter(
        theta_samples_fm[:, 0],
        theta_samples_fm[:, 1],
        alpha=0.5,
        s=15,
        c="orange",
        label="FM samples",
    )
    axes[1, 1].set_xlabel("θ[0]")
    axes[1, 1].set_ylabel("θ[1]")
    axes[1, 1].set_title("Flow Matching: θ samples vs true p(θ|y)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot theta distributions (histograms) - compare to true posterior
    axes[1, 2].hist(
        theta_samples_elbo[:, 0], bins=30, alpha=0.6, label="ELBO", density=True, color="blue"
    )
    axes[1, 2].hist(
        theta_samples_fm[:, 0], bins=30, alpha=0.6, label="FM", density=True, color="orange"
    )
    axes[1, 2].hist(
        theta_samples_true[:, 0],
        bins=30,
        alpha=0.6,
        label="True p(θ|y)",
        density=True,
        color="green",
    )
    axes[1, 2].set_xlabel("θ[0]")
    axes[1, 2].set_ylabel("Density")
    axes[1, 2].set_title("θ[0] Distribution")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    axes[1, 3].hist(
        theta_samples_elbo[:, 1], bins=30, alpha=0.6, label="ELBO", density=True, color="blue"
    )
    axes[1, 3].hist(
        theta_samples_fm[:, 1], bins=30, alpha=0.6, label="FM", density=True, color="orange"
    )
    axes[1, 3].hist(
        theta_samples_true[:, 1],
        bins=30,
        alpha=0.6,
        label="True p(θ|y)",
        density=True,
        color="green",
    )
    axes[1, 3].set_xlabel("θ[1]")
    axes[1, 3].set_ylabel("Density")
    axes[1, 3].set_title("θ[1] Distribution")
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f"logs/elbo_source/comparison_{model_type}.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved visualization to {output_path}")
    plt.close()


# ============================================================================
# 6. Sample Size Experiment
# ============================================================================


def run_experiment(
    model,
    sample_sizes: List[int],
    n_seeds: int,
    n_test: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    test_seed: int = 999,
) -> Dict:
    """Run sample size experiment comparing FM, ELBO-NSF, and ELBO-Linear.

    Args:
        model: SimpleGaussianModel or ThetaDependentGaussianModel
        sample_sizes: List of training sample sizes to test
        n_seeds: Number of random seeds per sample size
        n_test: Number of test samples (fixed across all runs)
        epochs: Training epochs per method
        lr: Learning rate
        batch_size: Batch size
        device: Device to use
        test_seed: Fixed seed for test data generation

    Returns:
        results[sample_size][method] = {
            "LPP": [values per seed],
            "C2ST": [values per seed],
            "LPP_mean": float,
            "LPP_std": float,
            "C2ST_mean": float,
            "C2ST_std": float,
        }
    """
    methods = ["FM", "ELBO-NSF", "ELBO-Linear"]

    # Generate test data ONCE with fixed seed
    print("\n" + "=" * 60)
    print("Generating Fixed Test Data")
    print("=" * 60)
    torch.manual_seed(test_seed)
    np.random.seed(test_seed)
    _, x_test, y_test = model.sample(n_test)
    print(f"Generated {n_test} test samples (seed={test_seed})")

    # Initialize results structure
    results = {}
    for sample_size in sample_sizes:
        results[sample_size] = {}
        for method in methods:
            results[sample_size][method] = {
                "LPP": [],
                "C2ST": [],
            }

    # Run experiment
    total_runs = len(sample_sizes) * n_seeds
    run_idx = 0

    for sample_size in sample_sizes:
        print("\n" + "=" * 60)
        print(f"SAMPLE SIZE: {sample_size}")
        print("=" * 60)

        for seed in range(n_seeds):
            run_idx += 1
            print(f"\n--- Run {run_idx}/{total_runs}: sample_size={sample_size}, seed={seed} ---")

            # Set seed for reproducible training data
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Generate training data
            theta_train, x_train, y_train = model.sample(sample_size)

            # Create standardizers from training data
            theta_std = Standardizer(theta_train)
            x_std = Standardizer(x_train)
            y_std = Standardizer(y_train)

            # Standardize training data
            theta_train_std = theta_std.standardize(theta_train).to(device)
            x_train_std = x_std.standardize(x_train).to(device)
            y_train_std = y_std.standardize(y_train).to(device)

            # Train FM (no caching)
            print(f"\nTraining FM (sample_size={sample_size}, seed={seed})...")
            fm, _ = train_source_fm_simple(
                model,
                x_train_std,
                y_train_std,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                device=device,
                cache_path=None,  # Disable caching for experiment
            )

            # Train ELBO-NSF
            print(f"\nTraining ELBO-NSF (sample_size={sample_size}, seed={seed})...")
            elbo_nsf, _, _ = train_source_elbo_simple(
                model,
                theta_train_std,
                y_train_std,
                theta_std,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                device=device,
                source_flow_type="nsf",
            )

            # Train ELBO-Linear
            print(f"\nTraining ELBO-Linear (sample_size={sample_size}, seed={seed})...")
            elbo_linear, _, _ = train_source_elbo_simple(
                model,
                theta_train_std,
                y_train_std,
                theta_std,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                device=device,
                source_flow_type="linear",
            )

            # Evaluate all 3 methods on fixed test set
            print(f"\nEvaluating methods (sample_size={sample_size}, seed={seed})...")

            # Evaluate FM
            metrics_fm = evaluate_method(
                model=model,
                source_flow=fm,
                y_test=y_test,
                theta_std=theta_std,
                x_std=x_std,
                y_std=y_std,
                n_samples=1000,  # Fewer samples for speed
                device=device,
                method_name="FM",
                is_flow_matching=True,
                batch_size=100,
            )
            results[sample_size]["FM"]["LPP"].append(metrics_fm["LPP"])
            results[sample_size]["FM"]["C2ST"].append(metrics_fm["C2ST"])

            # Evaluate ELBO-NSF
            metrics_nsf = evaluate_method(
                model=model,
                source_flow=elbo_nsf,
                y_test=y_test,
                theta_std=theta_std,
                x_std=x_std,
                y_std=y_std,
                n_samples=1000,
                device=device,
                method_name="ELBO-NSF",
                is_flow_matching=False,
                batch_size=100,
            )
            results[sample_size]["ELBO-NSF"]["LPP"].append(metrics_nsf["LPP"])
            results[sample_size]["ELBO-NSF"]["C2ST"].append(metrics_nsf["C2ST"])

            # Evaluate ELBO-Linear
            metrics_linear = evaluate_method(
                model=model,
                source_flow=elbo_linear,
                y_test=y_test,
                theta_std=theta_std,
                x_std=x_std,
                y_std=y_std,
                n_samples=1000,
                device=device,
                method_name="ELBO-Linear",
                is_flow_matching=False,
                batch_size=100,
            )
            results[sample_size]["ELBO-Linear"]["LPP"].append(metrics_linear["LPP"])
            results[sample_size]["ELBO-Linear"]["C2ST"].append(metrics_linear["C2ST"])

            # Clear GPU memory
            del fm, elbo_nsf, elbo_linear
            if device != "cpu":
                torch.cuda.empty_cache()

    # Compute mean and std for each method/sample_size
    for sample_size in sample_sizes:
        for method in methods:
            lpp_values = results[sample_size][method]["LPP"]
            c2st_values = results[sample_size][method]["C2ST"]
            results[sample_size][method]["LPP_mean"] = np.mean(lpp_values)
            results[sample_size][method]["LPP_std"] = np.std(lpp_values)
            results[sample_size][method]["C2ST_mean"] = np.mean(c2st_values)
            results[sample_size][method]["C2ST_std"] = np.std(c2st_values)

    return results


def plot_metrics_vs_sample_size(
    results: Dict,
    sample_sizes: List[int],
    output_dir: str = "logs/elbo_source",
):
    """Plot LPP and C2ST vs sample size for all methods.

    Creates two separate PDF plots:
    - experiment_lpp.pdf: LPP vs sample size
    - experiment_c2st.pdf: C2ST vs sample size

    Args:
        results: Results dictionary from run_experiment()
        sample_sizes: List of sample sizes
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    methods = ["FM", "ELBO-NSF", "ELBO-Linear"]
    colors = {"FM": "tab:orange", "ELBO-NSF": "tab:blue", "ELBO-Linear": "tab:green"}
    markers = {"FM": "o", "ELBO-NSF": "s", "ELBO-Linear": "^"}

    # Plot 1: LPP vs Sample Size
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    for method in methods:
        lpp_means = [results[s][method]["LPP_mean"] for s in sample_sizes]
        lpp_stds = [results[s][method]["LPP_std"] for s in sample_sizes]

        ax1.errorbar(
            sample_sizes,
            lpp_means,
            yerr=lpp_stds,
            label=method,
            marker=markers[method],
            color=colors[method],
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=2,
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("Training Sample Size", fontsize=12)
    ax1.set_ylabel("Log Posterior Probability (LPP)", fontsize=12)
    ax1.set_title("LPP vs Training Sample Size", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(sample_sizes)
    ax1.set_xticklabels(sample_sizes)

    plt.tight_layout()
    lpp_path = os.path.join(output_dir, "experiment_lpp.pdf")
    fig1.savefig(lpp_path, dpi=300, bbox_inches="tight")
    print(f"Saved LPP plot to {lpp_path}")
    plt.close(fig1)

    # Plot 2: C2ST vs Sample Size
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    for method in methods:
        c2st_means = [results[s][method]["C2ST_mean"] for s in sample_sizes]
        c2st_stds = [results[s][method]["C2ST_std"] for s in sample_sizes]

        ax2.errorbar(
            sample_sizes,
            c2st_means,
            yerr=c2st_stds,
            label=method,
            marker=markers[method],
            color=colors[method],
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=2,
        )

    # Add reference line at 0.5 (perfect)
    ax2.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.5, label="Perfect (0.5)")

    ax2.set_xscale("log")
    ax2.set_xlabel("Training Sample Size", fontsize=12)
    ax2.set_ylabel("C2ST Accuracy", fontsize=12)
    ax2.set_title("C2ST vs Training Sample Size", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(sample_sizes)
    ax2.set_xticklabels(sample_sizes)

    plt.tight_layout()
    c2st_path = os.path.join(output_dir, "experiment_c2st.pdf")
    fig2.savefig(c2st_path, dpi=300, bbox_inches="tight")
    print(f"Saved C2ST plot to {c2st_path}")
    plt.close(fig2)


def print_summary_table(results: Dict, sample_sizes: List[int]):
    """Print a summary table of results."""
    methods = ["FM", "ELBO-NSF", "ELBO-Linear"]

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    # LPP Table
    print("\nLog Posterior Probability (LPP) - Higher is better")
    print("-" * 80)
    header = f"{'Sample Size':<15}"
    for method in methods:
        header += f"{method:>20}"
    print(header)
    print("-" * 80)

    for sample_size in sample_sizes:
        row = f"{sample_size:<15}"
        for method in methods:
            mean = results[sample_size][method]["LPP_mean"]
            std = results[sample_size][method]["LPP_std"]
            row += f"{mean:>12.3f} +/- {std:<5.3f}"
        print(row)

    # C2ST Table
    print("\nC2ST Accuracy - Closer to 0.5 is better")
    print("-" * 80)
    header = f"{'Sample Size':<15}"
    for method in methods:
        header += f"{method:>20}"
    print(header)
    print("-" * 80)

    for sample_size in sample_sizes:
        row = f"{sample_size:<15}"
        for method in methods:
            mean = results[sample_size][method]["C2ST_mean"]
            std = results[sample_size][method]["C2ST_std"]
            row += f"{mean:>12.3f} +/- {std:<5.3f}"
        print(row)

    print("-" * 80)


# ============================================================================
# 7. Main
# ============================================================================


def main():
    # =========================================================================
    # Experiment Configuration
    # =========================================================================
    sample_sizes = [10, 50, 200, 1000]
    n_seeds = 10
    n_test = 1000  # Fixed test set size
    epochs = 100
    lr = 1e-5
    batch_size = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_seed = 42  # Seed for model initialization

    # Model configuration
    theta_dim = 2
    x_dim = 3
    y_dim = 3
    model_type = "theta_dependent"  # "simple" or "theta_dependent"
    noise_scale = 1.0

    print("=" * 60)
    print("SAMPLE SIZE EXPERIMENT: ELBO vs FM COMPARISON")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Seeds per sample size: {n_seeds}")
    print(f"Test samples: {n_test}")
    print(f"Epochs per run: {epochs}")
    print(f"Model type: {model_type}")
    print("Methods: FM, ELBO-NSF, ELBO-Linear")
    print("=" * 60)

    # Create model with fixed seed
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)

    if model_type == "theta_dependent":
        model = ThetaDependentGaussianModel(
            theta_dim, x_dim, y_dim, noise_scale=noise_scale, seed=model_seed
        )
    else:
        model = SimpleGaussianModel(theta_dim, x_dim, y_dim, seed=model_seed)

    # Run the experiment
    results = run_experiment(
        model=model,
        sample_sizes=sample_sizes,
        n_seeds=n_seeds,
        n_test=n_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        test_seed=999,
    )

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    plot_metrics_vs_sample_size(results, sample_sizes)

    # Print summary table
    print_summary_table(results, sample_sizes)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE!")
    print("=" * 60)
    print("Output files:")
    print("  - logs/elbo_source/experiment_lpp.pdf")
    print("  - logs/elbo_source/experiment_c2st.pdf")


if __name__ == "__main__":
    main()

import math

import torch
import torch.nn as nn


def compute_polynomial_terms_unique(x, degree):
    """
    Compute unique polynomial terms (monomials) of given degree from x.
    Includes symmetric unique combinations (z1*z2, not z2*z1).

    Args:
        x: (batch_size, latent_dim)
        degree: polynomial degree (int)

    Returns:
        (batch_size, num_unique_terms)
    """
    batch_size, latent_dim = x.shape

    if degree == 0:
        return torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)
    elif degree == 1:
        return x

    # recursive unique expansion
    out = x
    for d in range(2, degree + 1):
        out = out.unsqueeze(2) * x.unsqueeze(1)
        idx_triu = torch.triu_indices(out.shape[1], out.shape[1])
        out = out[:, idx_triu[0], idx_triu[1]]

    return out.reshape(batch_size, -1)


class PolyEncoder(nn.Module):
    """
    Polynomial Encoder that approximates the inverse of a polynomial mixing.
    Uses unique polynomial features up to a given degree.
    """

    def __init__(self, data_dim, latent_dim, poly_degree, device):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.poly_degree = poly_degree
        self.device = device

        # compute the number of unique polynomial terms up to degree d
        self.total_poly_terms = sum(
            math.comb(data_dim + d - 1, d) for d in range(poly_degree + 1)
        )

        self.linear = nn.Linear(self.total_poly_terms, latent_dim)
        self.batchnorm = nn.BatchNorm1d(latent_dim, affine=False)

    def forward(self, x):
        poly_terms = []
        for d in range(self.poly_degree + 1):
            term = compute_polynomial_terms_unique(x, d)
            poly_terms.append(term)
        poly = torch.cat(poly_terms, dim=1)  # shape: (batch, total_poly_terms)

        z_hat = self.linear(poly)
        z_hat = self.batchnorm(z_hat)

        return z_hat


class OraclePolyEncoder(nn.Module):
    """
    Oracle Polynomial Encoder that uses the true polynomial mixing weights
    to invert the polynomial mixing exactly.
    """

    def __init__(self, poly_mix_weights, latent_dim, device):
        super().__init__()
        self.poly_mix_weights = poly_mix_weights.to(device)
        self.device = device
        self.latent_dim = latent_dim

        # Precompute the pseudo-inverse of the mixing weights
        self.pseudo_inv_weights = (
            torch.linalg.pinv(self.poly_mix_weights.T @ self.poly_mix_weights)
            @ self.poly_mix_weights.T
        )

    def forward(self, x):
        z_hat = (x @ self.pseudo_inv_weights.T)[
            :, 1 : (self.latent_dim + 1)
        ]  # exclude constant term
        return z_hat

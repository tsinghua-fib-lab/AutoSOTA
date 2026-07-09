import itertools
import math

import torch
import torch.nn as nn


class PolyDecoder(nn.Module):
    """
    Polynomial decoder using unique monomials up to a given degree.

    Computes all unique polynomial terms (symmetric combinations) of the latent
    variables up to poly_degree, then applies a linear map to predict data.

    Args:
        data_dim (int): output dimension
        latent_dim (int): dimension of latent variable
        poly_degree (int): maximum polynomial degree
        device (torch.device): computation device
    """

    def __init__(self, data_dim, latent_dim, poly_degree, device):
        super(PolyDecoder, self).__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.poly_degree = poly_degree
        self.device = device

        # total number of unique polynomial features (including constant term)
        self.total_poly_terms = self.compute_total_polynomial_terms()

        # simple linear map from polynomial features to data
        self.coff_matrix = nn.Sequential(
            nn.Linear(self.total_poly_terms, self.data_dim, bias=True)
        )

    def compute_total_polynomial_terms(self):
        """Compute total number of unique polynomial terms (symmetric)."""
        p = self.latent_dim
        D = self.poly_degree
        return sum(math.comb(p + d - 1, d) for d in range(D + 1))

    def compute_polynomial_terms(self, latent):
        """Compute all unique polynomial monomials up to given degree."""
        latent = latent.unsqueeze(0) if latent.dim() == 1 else latent
        n, p = latent.shape

        terms = []
        for d in range(self.poly_degree + 1):
            if d == 0:
                terms.append(
                    torch.ones((n, 1), dtype=latent.dtype, device=latent.device)
                )
            else:
                combos = list(
                    itertools.combinations_with_replacement(range(p), d)
                )
                deg_terms = [
                    latent[:, idx].prod(dim=1, keepdim=True) for idx in combos
                ]
                deg_terms = torch.cat(deg_terms, dim=1)
                terms.append(deg_terms)

        return torch.cat(terms, dim=1)

    def forward(self, z):
        """
        Input:
            z: tensor of shape (batch_size, latent_dim)
        Output:
            decoded x of shape (batch_size, data_dim)
        """
        poly_features = self.compute_polynomial_terms(z)
        return self.coff_matrix(poly_features)

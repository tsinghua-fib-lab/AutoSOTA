import math

import torch
import torch.nn as nn
from torch_losses import compute_polynomial_terms


class PolyMixing(nn.Module):
    """
    Fixed polynomial mixing layer of given degree.

    If output_dim == total_dim, generates an orthogonal mixing matrix (invertible).
    If output_dim > total_dim, generates a tall full-rank random matrix.
    If output_dim < total_dim, prints a warning (non-invertible).
    """

    def __init__(self, degree, latent_dim, output_dim=None, weights=None):
        super().__init__()
        self.degree = degree
        self.latent_dim = latent_dim

        # compute dimension of concatenated polynomial features
        # total_dim = sum([latent_dim**d for d in range(degree + 1)])
        total_dim = sum(
            math.comb(latent_dim + d - 1, d) for d in range(degree + 1)
        )
        self.total_dim = total_dim

        if weights is not None:
            weights = torch.as_tensor(weights, dtype=torch.float32)
            assert (
                weights.shape[1] == total_dim
            ), f"weights should have {total_dim} input features"
            self.linear = nn.Linear(total_dim, weights.shape[0], bias=False)
            self.linear.weight = nn.Parameter(weights, requires_grad=False)

        else:
            if output_dim is None:
                output_dim = total_dim

            if output_dim < total_dim:
                print(
                    f"⚠️ Warning: output_dim={output_dim} < total_dim={total_dim} "
                    "→ mapping will not be invertible."
                )

            self.linear = nn.Linear(total_dim, output_dim, bias=False)

            # Initialize an invertible or full-rank matrix
            if output_dim == total_dim:
                # Orthogonal (invertible) matrix
                Q, _ = torch.linalg.qr(torch.randn(total_dim, total_dim))
                self.linear.weight = nn.Parameter(Q.T, requires_grad=False)
            else:
                # Full-rank random matrix
                W = torch.randn(output_dim, total_dim)
                self.linear.weight = nn.Parameter(W, requires_grad=False)

    def forward(self, latent):

        device = self.linear.weight.device
        poly_terms = []
        for d in range(self.degree + 1):
            term = compute_polynomial_terms(latent, d)
            poly_terms.append(term)
        poly = torch.column_stack(poly_terms).to(device)

        return self.linear(poly)

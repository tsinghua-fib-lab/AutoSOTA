"""
Domain Axes construction module
Following the paper's design, the domain axes are actually a one-hot structure over the selected domains,
not a true PCA space but a simple one-hot vector representation
"""

import torch
import numpy as np
from typing import List, Optional
import logging

from ..utils import get_logger

logger = get_logger(__name__)


def create_domain_axes_onehot(
    num_domains: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create the domain axes (one-hot structure)

    As described: the PCA axes in the paper are actually the one-hot structure over the selected domains.
    Each domain corresponds to a one-hot vector.

    Args:
        num_domains: Number of domains (the selected domains)
        device: Compute device

    Returns:
        domain_axes: [num_domains, num_domains] one-hot matrix
        Each row is the one-hot vector of a domain
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the one-hot matrix
    # The k-th domain corresponds to row k being 1 and all others 0
    domain_axes = torch.eye(num_domains, device=device, dtype=torch.float32)

    logger.info(f"Created domain axes (one-hot): {num_domains} domains, shape={domain_axes.shape}")
    
    return domain_axes


def project_to_domain_space(
    residual: torch.Tensor,  # [batch, hidden_dim] post-attention residual
    domain_probes: Optional[torch.nn.Module] = None,
    use_probe_projection: bool = True
) -> torch.Tensor:
    """
    Project the post-attention residual into the domain space

    As described: this is not a true PCA projection, but maps the residual into the
    domain space via a probe, then computes alignment using the one-hot domain axes.

    Method 1: Use the probe output (recommended)
    - Use the trained probe to map the residual to [num_domains] dimensions
    - The output is the sigmoid probability of each domain

    Method 2: Directly use certain dimensions of the residual (if the residual is already domain-related)

    Args:
        residual: post-attention residual [batch, hidden_dim]
        domain_probes: Probe model (if using probe projection)
        use_probe_projection: Whether to use probe projection

    Returns:
        residual_domain: [batch, num_domains] residual projected into the domain space
    """
    if use_probe_projection and domain_probes is not None:
        # Use the probe to map the residual into the domain space
        with torch.no_grad():
            logits = domain_probes(residual)  # [batch, num_domains]
            # Use sigmoid output (1-vs-rest)
            residual_domain = torch.sigmoid(logits)  # [batch, num_domains]
    else:
        # Without a probe, projection cannot be done correctly
        raise ValueError(
            "A probe must be provided to project the residual into the domain space. "
            "By design, the domain axes are a one-hot structure, so a probe is required "
            "to map the residual into the domain space."
        )
    
    return residual_domain


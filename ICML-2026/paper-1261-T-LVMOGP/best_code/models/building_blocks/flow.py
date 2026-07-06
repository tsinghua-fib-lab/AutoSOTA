import torch.nn as nn

__all__ = ["ArcsinhFlow"]


class ArcsinhFlow(nn.Module):
    """Placeholder ArcsinhFlow - not used in current code path."""
    def __init__(self, n_blocks=5, add_init_f0=True):
        super().__init__()
        self.n_blocks = n_blocks
    
    def forward(self, x):
        return x

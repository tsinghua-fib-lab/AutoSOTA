import torch
import torch.nn as nn

class IdentityDropout(nn.Module):
    """
    Dropout that drops values to 1.0 (identity) instead of 0.0
    Useful for multiplicative filters where identity = no filtering
    """
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        if self.p == 1:
            return torch.ones_like(x)
            
        # Create dropout mask (1.0 for keep, 0.0 for drop)
        dropout_mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p))
        
        # Convert dropped values (0.0) to identity (1.0)
        dropout_mask = torch.where(dropout_mask == 0, 1.0, dropout_mask)
        
        # Scale up kept values to maintain expected value
        dropout_mask = dropout_mask / (1 - self.p)
        
        return x * dropout_mask
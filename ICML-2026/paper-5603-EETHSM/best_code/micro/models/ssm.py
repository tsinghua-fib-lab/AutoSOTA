import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

class SimpleSSMLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv=4, expand=2):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        self.mamba = Mamba(
            d_model=d_model,       # Model dimension
            d_state=d_state,       # SSM state expansion factor
            d_conv=d_conv,         # Local convolution width
            expand=expand,         # Block expansion factor
            dt_rank=1              # DESIGN DECISION, for construction
        )

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        return x + self.mamba(x)
        # return self.mamba(x)
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.Wg = nn.Linear(hidden_dim, hidden_dim * 8//3, bias=False)
        self.W1 = nn.Linear(hidden_dim, hidden_dim * 8//3, bias=False)
        self.W2 = nn.Linear(hidden_dim * 8//3, hidden_dim, bias=False)
    
    def forward(self, x):
        gate = F.silu(self.Wg(x))
        linear = self.W1(x)
        
        gated_flow = gate * linear
        
        return self.W2(gated_flow)
import torch
import torch.nn as nn
from .blocks import FeatureExtractor, ProcessingModule

class NoProp_Network(nn.Module):
    def __init__(self, num_steps, num_classes, in_channels, feature_dim=128):
        super().__init__()
        self.T = num_steps
        self.num_classes = num_classes
        self.cnn = FeatureExtractor(in_channels, feature_dim)
        self.mlps = nn.ModuleList([ProcessingModule(num_classes, feature_dim) for _ in range(self.T)])
        self.register_buffer('alpha', torch.linspace(1.0, 0.1, self.T))
        
    def forward(self, x):
        z_t = torch.randn(x.shape[0], self.num_classes, device=x.device)
        x_features = self.cnn(x)
        for t in reversed(range(self.T)): 
            z_t = self.mlps[t](z_t, x_features)
        return z_t

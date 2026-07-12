import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import FeatureExtractor, ProcessingModule, NonDiffModule

class DMoL_Network(nn.Module):
    def __init__(self, num_modules, num_classes, in_channels, feature_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.num_modules = num_modules
        self.feature_extractor = FeatureExtractor(in_channels, feature_dim)
        self.processing_modules = nn.ModuleList([ProcessingModule(num_classes, feature_dim) for _ in range(num_modules)])
        
    def forward(self, x):
        p_current = torch.full((x.shape[0], self.num_classes), 1.0 / self.num_classes, device=x.device)
        shared_features = self.feature_extractor(x)
        final_logits = None
        for module in self.processing_modules:
            final_logits = module(p_current, shared_features)
            p_current = F.softmax(final_logits, dim=1)
        return final_logits

class DMoL_NonDiff_Network(DMoL_Network):
    def __init__(self, num_modules, num_classes, in_channels, feature_dim=128):
        super().__init__(num_modules, num_classes, in_channels, feature_dim)
        self.nondiff_module = NonDiffModule()
        self.insert_point = num_modules // 2
        
    def forward(self, x):
        p_current = torch.full((x.shape[0], self.num_classes), 1.0/self.num_classes, device=x.device)
        shared_features = self.feature_extractor(x)
        final_logits = None
        for i, module in enumerate(self.processing_modules):
            if i == self.insert_point: 
                p_current = self.nondiff_module(p_current)
            final_logits = module(p_current, shared_features)
            p_current = F.softmax(final_logits, dim=1)
        return final_logits

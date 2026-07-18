from diffusers.models.embeddings import TimestepEmbedding
from diffusers import UNet2DModel
from torch import nn
import torch
from .label_encoder import MultiLabelEncoder

try:
    import xformers
    _XFORMERS_AVAILABLE = True
except Exception:
    _XFORMERS_AVAILABLE = False

def initialize_weights(m):
    for name, param in m.named_parameters():
        if isinstance(m, nn.Conv2d):
            if 'weight' in name:
                torch.nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
        elif isinstance(m, nn.Linear):
            if 'weight' in name:
                torch.nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
        elif isinstance(m, nn.BatchNorm2d):
            if 'weight' in name:
                torch.nn.init.ones_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

class ClassConditionalUnet(UNet2DModel):
    def __init__(self, num_class_per_label, interaction=None,**kwargs):
        super().__init__(**kwargs)
        self.class_embedding = None
        time_embed_dim = self.config.block_out_channels[0] * 4
        if interaction in ['cat','sum']:
            self.class_embedding = MultiLabelEncoder(num_class_per_label = num_class_per_label,
                                                        interaction = interaction,
                                                        d_latent = time_embed_dim)
                                                        
        else:
            self.class_embedding = TimestepEmbedding(
                in_channels=len(num_class_per_label),
                time_embed_dim=time_embed_dim
            )
        if _XFORMERS_AVAILABLE:
            self.enable_xformers_memory_efficient_attention()
        initialize_weights(self)
        self.num_classes_per_label = num_class_per_label

    def forward(self, xt, t, y=None, scheduler=None):
        eps = super().forward(xt, t, y).sample
        
        if scheduler is not None:
            sigma_t = torch.sqrt(1 - scheduler.alphas_cumprod[t])
            score_i = -eps / sigma_t
            return eps, score_i
            
        return eps

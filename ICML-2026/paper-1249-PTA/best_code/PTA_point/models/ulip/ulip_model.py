import torch
from torch import nn

from .pointbert.point_encoder import PointTransformer


class ULIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # --- point encoder
        self.point_encoder = PointTransformer(args)
        self.pc_projection = nn.Parameter(torch.empty(args.pc_feat_dim, 512))

    def forward(self, pc):
        pc_feat = self.point_encoder(pc)
        pc_embed = pc_feat @ self.pc_projection
        return pc_embed

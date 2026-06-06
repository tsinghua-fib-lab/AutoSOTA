import sys
from pathlib import Path

from timm.models.vision_transformer import trunc_normal_
import torch.nn as nn
import torch
from mmseg.utils import get_root_logger
from mmcv.runner import _load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from models.utils import LayerNorm1D, LayerNorm2D, FFN, Stem, PatchMerging
from models.layers import HoGEdgeGateConv

from VMamba.models.vmamba import TransMixer



class VSS(nn.Module):
    def __init__(self, in_dim, depth, mlp_ratio=4., state_dim=64):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = nn.Sequential(
                TransMixer(hidden_dim=in_dim, ssm_d_state=state_dim, mlp_ratio=mlp_ratio, channel_first=True),
                HoGEdgeGateConv(
                            in_dim=in_dim,
                            nbins=180
                )
            )
            self.blocks.append(block)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class VSSEncoder(nn.Module):
    def __init__(self, in_dim=3,  
                 embed_dim=[128,256,512], 
                 depths=[2, 2, 2], 
                 mlp_ratio=4.,
                 state_dim=[49,25,9], distillation=False,
                 is_patch_embed=True,
                 ):
        super().__init__()
        self.num_layers = len(depths)
        self.distillation =distillation
        if is_patch_embed:
            self.patch_embed = Stem(in_dim=in_dim, dim=embed_dim[0])
        self.is_patch_embed = is_patch_embed    

        # build stages
        self.vss_layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i_layer in range(self.num_layers):

            vss = VSS(in_dim=int(embed_dim[i_layer]),
                      depth=depths[i_layer],
                      mlp_ratio=mlp_ratio,
                      state_dim = state_dim[i_layer])
            self.vss_layers.append(vss)

            if i_layer < self.num_layers - 1:
                downsample = PatchMerging(in_dim=int(embed_dim[i_layer]), 
                                          out_dim=int(embed_dim[i_layer+1])) 
                self.downsamples.append(downsample)

        self.apply(self._init_weights)
        self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
        self.train()
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm2D):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
            logger.info(f"Miss {missing_keys}")
            logger.info(f"Unexpected {unexpected_keys}")

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(VSSEncoder, self).train(mode)
        if mode:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        if self.is_patch_embed:
            x = self.patch_embed(x)
            
        outs = []
        for i in range(self.num_layers):
            vss = self.vss_layers[i]
            x = vss(x)
        
            outs.append(x)     
            if i < self.num_layers - 1:
                down = self.downsamples[i]
                x = down(x)

        return outs
    




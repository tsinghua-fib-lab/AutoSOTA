import torch
import torch.nn as nn
import torch.nn.functional as F

from clip.model import ModifiedResNet as CLIP_RN

from .peft_modules import *


class RN_Tuner(nn.Module):
    """ All instance variables in this class will be optimized.
    """
    def __init__(self, cfg, rn_model, num_classes):
        super().__init__()

        if isinstance(rn_model, CLIP_RN):
            feat_dim = rn_model.output_dim
            dtype = rn_model.conv1.weight.dtype

            use_full_tuning = cfg.full_tuning
            use_bias_tuning = cfg.bias_tuning
            use_bn_tuning = cfg.bn_tuning
            use_ssf_attn = cfg.ssf_attn

        if use_full_tuning:
            full_list = nn.ParameterList([
                param for name, param in rn_model.named_parameters()
            ])
        else:
            full_list = None

        if use_bias_tuning:
            bias_list = nn.ParameterList([
                param for name, param in rn_model.named_parameters()
                if name.endswith("bias")
            ])
        else:
            bias_list = None

        if use_bn_tuning:
            bn_list = nn.ModuleList([
                mod for name, mod in rn_model.named_modules()
                if isinstance(mod, nn.BatchNorm2d)
            ])
        else:
            bn_list = None

        if use_ssf_attn:
            ssf_attn = SSF(feat_dim, dtype=dtype)
        else:
            ssf_attn = None

        # To be optimized
        self.full_list = full_list
        self.bias_list = bias_list
        self.bn_list = bn_list
        self.ssf_attn = ssf_attn


class Peft_RN(nn.Module):
    def __init__(self, rn_model):
        super().__init__()
        
        if isinstance(rn_model, CLIP_RN):
            self.backbone = "CLIP-RN"
            self.conv1 = rn_model.conv1
            self.bn1 = rn_model.bn1
            self.conv2 = rn_model.conv2
            self.bn2 = rn_model.bn2
            self.conv3 = rn_model.conv3
            self.bn3 = rn_model.bn3
            self.avgpool = rn_model.avgpool
            self.relu = rn_model.relu
            self.layer1 = rn_model.layer1
            self.layer2 = rn_model.layer2
            self.layer3 = rn_model.layer3
            self.layer4 = rn_model.layer4
            self.attnpool = rn_model.attnpool
            self.out_dim = rn_model.attnpool.c_proj.bias.shape[0]
    
    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def forward(self, x, tuner=None, head=None, text_features=None, samples=0):
        
        x = x.to(self.dtype)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        
        if tuner is not None and tuner.ssf_attn is not None:
            x = tuner.ssf_attn(x)

        if head is None:
            return x
        else:
            return head(x)



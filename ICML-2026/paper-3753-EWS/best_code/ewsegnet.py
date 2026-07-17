import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple
from mmseg.models.builder import BACKBONES
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
import warnings
import torch.nn.functional as F
from scipy import signal
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time



def gaussian(
    M,
    std = 1.0,
    layout = torch.strided,
    device = None,
    requires_grad = False,
):
    start = -(M - 1) / 2.0
    constant = 1 / (std * math.sqrt(2))

    k = torch.linspace(
        start=start * constant,
        end=(start + (M - 1)) * constant,
        steps=M,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )

    return torch.exp(-(k**2))


def gaussian_kernel(n, std, device=None, normalised=True):
    '''
    Generates a n x n matrix with a centered gaussian 
    of standard deviation std centered on it. If normalised,
    its volume equals 1.'''

    ## numpy version
    #gaussian1D = signal.gaussian(n, std)
    #gaussian2D = np.outer(gaussian1D, gaussian1D)

    ## get Gaussian using torch
    gaussian1D = gaussian(n, std=std, device=device)
    gaussian2D = torch.outer(gaussian1D, gaussian1D)

    if normalised:
        gaussian2D /= (2*np.pi*(std**2))

    return gaussian2D



class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SCM(nn.Module):
    """
    Spatial Context Module
    """
    def __init__(self, dim, kernel_size=3, act_layer=nn.GELU):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim*3, kernel_size=kernel_size, padding=kernel_size//2, stride=1, bias=False, groups=dim)
        self.act = act_layer()
        self.proj = nn.Conv2d(dim*2, dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.dwconv(x)
        x, soft, sig = torch.split(x, (C, C, C), dim=1)
        x = self.act(x)

        x_sig = x * torch.sigmoid(sig.mean(dim=1).unsqueeze(1))
        soft = torch.softmax(soft.mean([-2, -1]), dim=-1)
        x_soft = x * soft.unsqueeze(-1).unsqueeze(-1)
        x = self.proj(torch.cat([x_sig, x_soft], dim=1))
        x = self.act(x)

        return x


class FCM(nn.Module):
    """
    Frequency Context Module
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Conv2d(dim, dim*2, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.act(self.proj(x))
        x1, x2 = torch.split(x, (C,C), dim=1)

        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)

        fx1 = torch.fft.rfft2(x1, dim=(-2, -1), norm='ortho')
        fx2 = torch.fft.rfft2(x2, dim=(-2, -1), norm='ortho')

        fx = fx1 * fx2

        x = torch.fft.irfft2(fx, s=(H, W), dim=(-2, -1), norm='ortho')

        x = torch.fft.fftshift(x)

        return x


class AFEM(nn.Module):
    """
    Auxialiary Feature Enhancement Module
    """
    def __init__(self, in_dim=128, out_dim=128, std1=1.5, std2=3.0):
        super().__init__()

        self.norm1 = LayerNorm(out_dim, eps=1e-6, data_format="channels_first")

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.proj = nn.Conv2d(in_dim, in_dim*3, kernel_size=1, stride=1, padding=0)

        self.act = nn.GELU()
        self.post_proj = nn.Conv2d(in_dim*2, out_dim, kernel_size=1, stride=1, padding=0)
        
        n1 = int(std1*6)
        if n1%2 == 0:
            n1 = n1 + 1

        n2 = int(std2*6)
        if n2%2 == 0:
            n2 = n2 + 1

        self.gauss1 = gaussian_kernel(n=n1, std=std1)
        self.gauss2 = gaussian_kernel(n=n2, std=std2)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):

        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B,C,H,W = x.shape

        #################### pooled attention #############       
        xp = self.act(self.proj(x))

        q, k, v = torch.split(xp, (C,C,C), dim=1)
        
        q = self.avg_pool(q)
        k = self.max_pool(k)

        q = q.flatten(2)
        k = k.flatten(2).permute(0,2,1).contiguous()
        v = v.flatten(2)

        qk = torch.softmax(q@k, dim=-1)
        qkv = (qk@v).reshape(B, C, H, W)
        qkv = self.act(qkv)

        ##################### difference of Gaussian filtering #################
        x1 = x.to(torch.float32)
        fx1 = torch.fft.rfft2(x1, dim=(-2, -1), norm='ortho')
        fx1 = torch.fft.fftshift(fx1)

        #with torch.no_grad():
        h1 = self.gauss1.to(fx1.device)
        h2 = self.gauss2.to(fx1.device)
        h1 = torch.nn.functional.interpolate(h1.unsqueeze(0).unsqueeze(0), size=fx1.shape[2:], mode='bilinear')
        h2 = torch.nn.functional.interpolate(h2.unsqueeze(0).unsqueeze(0), size=fx1.shape[2:], mode='bilinear')

        mag = fx1.abs()
        ang = fx1.angle()

        mag1 = mag * h1.to(torch.float32)
        mag2 = mag * h2.to(torch.float32)

        realp = torch.cos(ang)
        imagp = torch.sin(ang)

        f1 = torch.fft.ifftshift(torch.complex(mag1*realp, mag1*imagp))
        f2 = torch.fft.ifftshift(torch.complex(mag2*realp, mag2*imagp))

        x1 = torch.fft.irfft2(f1, s=(H, W), dim=(-2, -1), norm='ortho')
        x2 = torch.fft.irfft2(f2, s=(H, W), dim=(-2, -1), norm='ortho')

        amt = x2 - x1
        amt = x.mean(axis=[-2,-1]).unsqueeze(-1).unsqueeze(-1) * amt
        xs = self.act(x2 + amt)

        out = self.norm1(self.act(self.post_proj(torch.cat([qkv, xs], dim=1))))

        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = x + self.act(self.dwconv(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        
###################################################################
#########      Efficient Waste Feature Extraction Block   #########

class Block(nn.Module):
    def __init__(self, dim, dw_kernel_size=3, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, layer_scale_init_value=1e-2):
        super().__init__()
        self.dwconv = SCM(dim, kernel_size=dw_kernel_size)

        self.dwnorm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.act = act_layer()

        self.fattn = FCM(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
                    
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.dwnorm(self.dwconv(x))
        fx = self.act(self.fattn(x))

        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.norm1(fx))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.norm2(self.mlp(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """ 
    Image to Patch Embedding

    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size//2)
        self.norm = LayerNorm(embed_dim, eps=1e-6, data_format="channels_first")
        self.act_layer = nn.GELU()


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        return x, H, W


###################################################################

@BACKBONES.register_module()
class EWSegNet(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[80, 160, 320, 640],
                dw_kernel_sizes=[5, 5, 3, 3], mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0.1,
                depths=[2, 2, 8, 2], num_stages=4, init_cfg=None, pretrained=None, norm_cfg=None, num_classes=1000, **kwargs):

        super().__init__()
        
        self.num_classes = num_classes        
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i], dw_kernel_size=dw_kernel_sizes[i], drop=drop_rate, drop_path=dpr[cur + j])
                                    for j in range(depths[i])]
                                    )
            norm = LayerNorm(embed_dims[i], eps=1e-6, data_format="channels_first")
            cur = cur + depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.aux_ref = AFEM(in_dim=embed_dims[2], out_dim=embed_dims[2])

        #self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        #self.apply(self._init_weights)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            cur_state_dict = self.state_dict()
            checkpoint = torch.load(pretrained, map_location="cpu")
            model_keys = list(checkpoint["state_dict"].keys())
            print(checkpoint.keys())

            # remove keys whose shape doesn't match
            for key in model_keys:
                if key not in cur_state_dict:
                    continue
                if checkpoint["state_dict"][key].shape != cur_state_dict[key].shape:
                    val = checkpoint["state_dict"][key]
                    val = F.interpolate(val, size=cur_state_dict[key].shape[2:], mode='bilinear', align_corners=True)
                    #del checkpoint["state_dict"][key]
                    checkpoint["state_dict"][key] = val

            msg = self.load_state_dict(checkpoint["state_dict"], strict=False)
            print(msg)
    

    def forward_features(self, x):
        B, C, H, W  = x.shape
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, _, _ = patch_embed(x)

            for blk in block:
                x = blk(x)

            x = norm(x)

            if i == (self.num_stages-2):
                x5 = self.aux_ref(x)
                x = x + x5
            
            outs.append(x)

        outs.append(x5)

        return outs


    def forward(self, x):
        feats = self.forward_features(x)

        return feats




###########################################################

if __name__ == "__main__":

    model = EWSegNet()
    model.eval()
    model.to('cuda')

    def count_parameters(model):
        total_trainable_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            total_trainable_params += params
        return total_trainable_params

    total_params = count_parameters(model)

    print(f"Total Trainable Params: {round(total_params * 1e-6, 2)} M")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    input_res = (1, 3, 512, 512)
    input = torch.ones(input_res, dtype=next(model.parameters()).dtype,
                                     device=next(model.parameters()).device)
    flops = FlopCountAnalysis(model, input)
    print(flop_count_table(flops))
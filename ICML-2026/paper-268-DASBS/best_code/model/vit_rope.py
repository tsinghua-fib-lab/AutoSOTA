"""
Adapted from https://github.com/yuchen-zhu-zyc/MDNS/blob/main/model/vit_rope.py

This code was originally obtained from:
https://github.com/naver-ai/rope-vit/blob/main/models/vit_rope.py
"""

import numpy as np
import torch
import torch.nn as nn
from functools import partial

import torch.nn.functional as F

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg

from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import register_model

from .vit import vit_models, Layer_scale_init_Block, Attention

def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    depth = freqs.shape[1]
    # No float 16 for this range
    with torch.amp.autocast('cuda',enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
                    
    return freqs_cis


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class RoPEAttention(Attention):
    """Multi-head Attention block with rotary position embeddings."""
    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
class RoPE_Layer_scale_init_Block(Layer_scale_init_Block):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, *args, **kwargs):
        kwargs["Attention_block"] = RoPEAttention
        super().__init__(*args, **kwargs)

    def forward(self, x, freqs_cis):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), freqs_cis=freqs_cis))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        
        return x


class VocabEmbedding(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Args:
            dim: dimension of the embedding (hidden size)
            vocab_dim: size of the vocabulary including the mask token (N)
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=5**.5)

    def forward(self, x):
        """
        Args:
            x: shape (B, D), values in range(N)
        
        Returns:
            vocab embedding, shape (B, D, dim)
        """
        return self.embedding[x]


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/model/transformer.py
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: [B]
        :param dim: the dimension of the output, =frequency_embedding_size.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [B, frequency_embedding_size] Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
            ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """[B] -> [B, hidden_size]"""
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class TimeScalarNet(torch.nn.Module):
    r"""
    Adapted from https://github.com/malbergo/leaps/blob/main/src/nets/F_net.py
    """
    def __init__(self, hidden_sizes=(32, 32, 32), 
                 activation=torch.nn.SiLU, input_dim = 1):
        super(TimeScalarNet, self).__init__()
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(input_dim, hidden_size))
            layers.append(activation())
            input_dim = hidden_size
        layers.append(torch.nn.Linear(input_dim, 1))
        # layers[-1].weight.data.fill_(0.01)
        # layers[-1].bias.data.fill_(1.1)
        self.net = torch.nn.Sequential(*layers)
        
    def forward(self, t):
        """t: [B] -> [B]"""
        return self.net(t[:, None]).squeeze()


class RopeViT(vit_models):
    def __init__(self, rope_theta=100.0, rope_mixed=False, use_ape=False, 
                 vocab_size=2, patch_size=1, in_chans=1, num_classes=0,
                 dtype='bfloat16', require_time=False, **kwargs):
        """
        For Ising model / Potts model:
            img_size: L, D = L ** 2
            patch_size and in_chans are always 1
            num_classes is always 0 (we don't use this feature)
        """
        super().__init__(patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, **kwargs)
        img_size = kwargs['img_size'] if 'img_size' in kwargs else 224
        # patch_size = kwargs['patch_size'] if 'patch_size' in kwargs else 16
        num_heads = kwargs['num_heads'] if 'num_heads' in kwargs else 12
        embed_dim = kwargs['embed_dim'] if 'embed_dim' in kwargs else 768
        mlp_ratio = kwargs['mlp_ratio'] if 'mlp_ratio' in kwargs else 4.
        
        self.require_time = require_time
        if not require_time:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=.02)
        else: # cls_token is given by time embedding
            self.time_embed = TimestepEmbedder(hidden_size=embed_dim)
        
        self.use_ape = use_ape
        if not self.use_ape:
            self.pos_embed = None            
        
        self.rope_mixed = rope_mixed
        self.num_heads = num_heads
        self.patch_size = patch_size
        
        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)
            
            freqs = []
            for i, _ in enumerate(self.blocks):
                freqs.append(
                    init_random_2d_freqs(dim=embed_dim // num_heads, num_heads=num_heads, theta=rope_theta)
                )
            freqs = torch.stack(freqs, dim=1).view(2, len(self.blocks), -1)
            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
            
            t_x, t_y = init_t_xy(end_x = img_size // patch_size, end_y = img_size // patch_size)
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)
        else:
            self.compute_cis = partial(compute_axial_cis, dim=embed_dim//num_heads, theta=rope_theta)
            
            freqs_cis = self.compute_cis(end_x = img_size // patch_size, end_y = img_size // patch_size)
            self.freqs_cis = freqs_cis
        
        # The following are new for our settings
        self.vocab_size = vocab_size
        self.L = img_size
        self.vocab_embed = VocabEmbedding(dim=self.embed_dim, vocab_dim=self.vocab_size)
        self.head = nn.Linear(self.embed_dim, self.vocab_size)
        self.dtype = {'float64': torch.float64,
                      'float32': torch.float32,
                      'float16': torch.float16,
                      'bfloat16': torch.bfloat16}.get(dtype, torch.bfloat16)

    def zero_init(self):
        """
        Use this for initializing the log corrector model to zero output at the beginning
        """
        self.head.weight.data.zero_()
        self.head.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'freqs'}
        
    def forward_features(self, x, t=None, **kwargs):
        """
        Args:
            x: shape (B, D=L^2), values in range(N)
            t: if require time input, shape (B, 1), values in [0, 1]; otherwise, do not pass

        Returns:
            shape (B, D+1, embed_dim)
        """
        B = x.shape[0]; H = W = self.L
        # x = self.patch_embed(x) # we don't use patch embedding, but use vocab embedding instead
        x = self.vocab_embed(x) # [B, D, embed_dim]

        if t is None and not self.require_time:
            cls_tokens = self.cls_token.expand(B, -1, -1) # [1, 1, embed_dim] -> [B, 1, embed_dim]
        elif t is not None and self.require_time:
            t = t * 1000  # scale t to [0, 1000]
            cls_tokens = self.time_embed(t.squeeze(-1)).unsqueeze(1) # [B, embed_dim] -> [B, 1, embed_dim]
        else:
            raise ValueError

        if self.use_ape: # go this way in our settings
            pos_embed = self.pos_embed
            if pos_embed.shape[-2] != x.shape[-2]:
                img_size = self.patch_embed.img_size
                patch_size = self.patch_embed.patch_size
                pos_embed = pos_embed.view(
                    1, (img_size[1] // patch_size[1]), (img_size[0] // patch_size[0]), self.embed_dim
                ).permute(0, 3, 1, 2)
                pos_embed = F.interpolate(
                    pos_embed, size=(H // patch_size[1], W // patch_size[0]), mode='bicubic', align_corners=False
                )
                pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            x = x + pos_embed
        
        x = torch.cat((cls_tokens, x), dim=1) # [B, D+1, embed_dim] from here on till the end
        
        if self.rope_mixed: # go this way in our settings
            if self.freqs_t_x.shape[0] != x.shape[1] - 1:
                t_x, t_y = init_t_xy(end_x = W // self.patch_size, end_y = H // self.patch_size)
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            else:
                t_x, t_y = self.freqs_t_x, self.freqs_t_y
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
            
            for i , blk in enumerate(self.blocks):
                x = blk(x, freqs_cis=freqs_cis[i])
        else:
            if self.freqs_cis.shape[0] != x.shape[1] - 1:
                freqs_cis = self.compute_cis(end_x = W // self.patch_size, end_y = H // self.patch_size)
            else:
                freqs_cis = self.freqs_cis
            freqs_cis = freqs_cis.to(x.device)
            
            for i , blk in enumerate(self.blocks):
                x = blk(x, freqs_cis=freqs_cis)
                
        x = self.norm(x)
        return x
    
    def forward(self, x, t=None, **kwargs):
        """
        Args:
            x: shape (B, D=L^2), values in range(N)
            t: if require time input, shape (B, 1), values in [0, 1]; otherwise, None

        Returns:
            shape (B, D, N)
        """
        with torch.amp.autocast('cuda', dtype=self.dtype):
            out = self.forward_features(x, t, **kwargs) # [B, D+1, embed_dim]
            out = self.head(out)[:, 1:, :] # [B, D+1, N] -> [B, D, N]
        return out


def get_rope_vit_model(args, require_time):
    # separate require_time for controller and corrector
    return RopeViT(img_size=args.L, patch_size=1, vocab_size=args.vocab_size, 
                   embed_dim=args.model.hidden_size, depth=args.model.n_blocks,
                   num_heads=args.model.n_heads, mlp_ratio=4, qkv_bias=True, in_chans=1, num_classes=0, 
                   norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=RoPE_Layer_scale_init_Block,
                   Attention_block=RoPEAttention,
                   rope_theta=10.0, rope_mixed=True, use_ape=True, dtype=args.model.dtype,
                   require_time=require_time)


# def hf_checkpoint_load(model_name):
#     try:
#         from huggingface_hub import hf_hub_download

#         ckpt_path = hf_hub_download(
#             repo_id="naver-ai/" + model_name, filename= "pytorch_model.bin"
#         )
#         checkpoint = torch.load(ckpt_path, map_location='cpu')
#     except:
#         _HF_URL = "https://huggingface.co/naver-ai/" + model_name + "/resolve/main/pytorch_model.bin"
#         checkpoint = torch.hub.load_state_dict_from_url(_HF_URL, map_location='cpu')

#     state_dict = checkpoint['model']
#     for k in ['freqs_t_x', 'freqs_t_y']:
#         if k in state_dict:
#             print(f"Removing key {k} from pretrained checkpoint")
#             del state_dict[k]
            
#     return checkpoint

# def adjust_pos_embed_size(model, state_dict):
    
#     # interpolate position embedding
#     if 'pos_embed' in state_dict:
#         pos_embed_checkpoint = state_dict['pos_embed']
#         embedding_size = pos_embed_checkpoint.shape[-1]
#         num_patches = model.patch_embed.num_patches
#         num_extra_tokens = model.pos_embed.shape[-2] - num_patches
#         # height (== width) for the checkpoint position embedding
#         orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
#         # height (== width) for the new position embedding
#         new_size = int(num_patches ** 0.5)
#         # class_token and dist_token are kept unchanged
#         extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
#         # only the position tokens are interpolated
#         pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
#         pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
#         pos_tokens = torch.nn.functional.interpolate(
#             pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
#         pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
#         new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
#         state_dict['pos_embed'] = new_pos_embed
        
#     return state_dict


# # RoPE-Axial
# @register_model
# def rope_axial_deit_small_patch16_LS(pretrained=False, img_size=224,  **kwargs):
#     model = rope_vit_models(
#         img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
#         rope_theta=100.0, rope_mixed=False, **kwargs)
#     model.default_cfg = _cfg()
    
#     if pretrained:
#         state_dict = hf_checkpoint_load("rope_axial_deit_small_patch16_LS")
#         model.load_state_dict(state_dict, strict=False)
        
#     return model

# @register_model
# def rope_axial_deit_base_patch16_LS(pretrained=False, img_size=224,  **kwargs):
#     model = rope_vit_models(
#         img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
#         rope_theta=100.0, rope_mixed=False, **kwargs)
    
#     if pretrained:
#         state_dict = hf_checkpoint_load("rope_axial_deit_base_patch16_LS")
#         model.load_state_dict(state_dict, strict=False)
        
#     return model

# @register_model
# def rope_axial_deit_large_patch16_LS(pretrained=False, img_size=224,  **kwargs):
#     model = rope_vit_models(
#         img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
#         rope_theta=100.0, rope_mixed=False, **kwargs)
    
#     if pretrained:
#         state_dict = hf_checkpoint_load("rope_axial_deit_large_patch16_LS")
#         model.load_state_dict(state_dict, strict=False)
        
#     return model

# # RoPE-Mixed
# @register_model
# def rope_mixed_deit_small_patch16_LS(pretrained=False, img_size=224,  **kwargs):
#     model = rope_vit_models(
#         img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
#         rope_theta=10.0, rope_mixed=True, **kwargs)
#     model.default_cfg = _cfg()
    
#     if pretrained:
#         state_dict = hf_checkpoint_load("rope_mixed_deit_small_patch16_LS")
#         model.load_state_dict(state_dict, strict=False)
    
#     return model

# @register_model
# def rope_mixed_deit_base_patch16_LS(pretrained=False, img_size=224,  **kwargs):
#     model = rope_vit_models(
#         img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
#         rope_theta=10.0, rope_mixed=True, **kwargs)
    
#     if pretrained:
#         state_dict = hf_checkpoint_load("rope_mixed_deit_base_patch16_LS")
#         model.load_state_dict(state_dict, strict=False)
        
#     return model

# @register_model
# def rope_mixed_deit_large_patch16_LS(pretrained=False, img_size=224,  **kwargs):
#     model = rope_vit_models(
#         img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
#         rope_theta=10.0, rope_mixed=True, **kwargs)
    
#     if pretrained:
#         state_dict = hf_checkpoint_load("rope_mixed_deit_large_patch16_LS")
#         model.load_state_dict(state_dict, strict=False)
        
#     return model


# # RoPE-Axial + APE
# @register_model
# def rope_axial_ape_deit_small_patch16_LS(pretrained=False, img_size=224,  **kwargs):
#     model = rope_vit_models(
#         img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
#         rope_theta=100.0, rope_mixed=False, use_ape=True, **kwargs)
#     model.default_cfg = _cfg()
    
#     if pretrained:
#         state_dict = hf_checkpoint_load("rope_axial_ape_deit_small_patch16_LS")
#         state_dict = adjust_pos_embed_size(model, state_dict)
#         model.load_state_dict(state_dict, strict=False)
        
#     return model

# @register_model
# def rope_axial_ape_deit_base_patch16_LS(pretrained=False, img_size=224,  **kwargs):
#     model = rope_vit_models(
#         img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
#         rope_theta=100.0, rope_mixed=False, use_ape=True, **kwargs)
    
#     if pretrained:
#         state_dict = hf_checkpoint_load("rope_axial_ape_deit_base_patch16_LS")
#         state_dict = adjust_pos_embed_size(model, state_dict)
#         model.load_state_dict(state_dict, strict=False)
        
#     return model

# @register_model
# def rope_axial_ape_deit_large_patch16_LS(pretrained=False, img_size=224,  **kwargs):
#     model = rope_vit_models(
#         img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
#         rope_theta=100.0, rope_mixed=False, use_ape=True, **kwargs)
    
#     if pretrained:
#         state_dict = hf_checkpoint_load("rope_axial_ape_deit_large_patch16_LS")
#         state_dict = adjust_pos_embed_size(model, state_dict)
#         model.load_state_dict(state_dict, strict=False)
        
#     return model

# # RoPE-Mixed + APE
# @register_model
# def rope_mixed_ape_deit_small_patch16_LS(pretrained=False, img_size=224,  **kwargs):
#     model = rope_vit_models(
#         img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
#         rope_theta=10.0, rope_mixed=True, use_ape=True, **kwargs)
#     model.default_cfg = _cfg()
    
#     if pretrained:
#         state_dict = hf_checkpoint_load("rope_mixed_ape_deit_small_patch16_LS")
#         state_dict = adjust_pos_embed_size(model, state_dict)
#         model.load_state_dict(state_dict, strict=False)
        
#     return model

# @register_model
# def rope_mixed_ape_deit_base_patch16_LS(pretrained=False, img_size=224,  **kwargs):
#     model = rope_vit_models(
#         img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
#         rope_theta=10.0, rope_mixed=True, use_ape=True, **kwargs)
    
#     if pretrained:
#         state_dict = hf_checkpoint_load("rope_mixed_ape_deit_base_patch16_LS")
#         state_dict = adjust_pos_embed_size(model, state_dict)
#         model.load_state_dict(state_dict, strict=False)
        
#     return model

# @register_model
# def rope_mixed_ape_deit_large_patch16_LS(pretrained=False, img_size=224,  **kwargs):
#     model = rope_vit_models(
#         img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
#         rope_theta=10.0, rope_mixed=True, use_ape=True, **kwargs)
    
#     if pretrained:
#         state_dict = hf_checkpoint_load("rope_mixed_ape_deit_large_patch16_LS")
#         state_dict = adjust_pos_embed_size(model, state_dict)
#         model.load_state_dict(state_dict, strict=False)
        
#     return model
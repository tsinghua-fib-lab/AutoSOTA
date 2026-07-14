"""
Adapted from https://github.com/naver-ai/rope-vit/blob/main/models/vit_rope.py

This code was originally obtained from:
https://github.com/facebookresearch/deit
and
https://github.com/meta-llama/codellama/blob/main/llama/model.py

Copyright (c) Meta Platforms, Inc. and affiliates.

All rights reserved.
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import register_model
from timm.models.vision_transformer import Mlp, PatchEmbed, _cfg

from .vit import Attention, Layer_scale_init_Block, vit_models


class ThermodynamicEmbedder(nn.Module):
    """Embeds scalar thermodynamic quantities using sinusoidal frequency embeddings.
    
    Similar to positional encodings, but for continuous scalar values like temperature
    and magnetic field. Uses sinusoidal embeddings which are better for continuous
    values than simple linear layers.
    """
    
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        use_mlp: bool = False,
    ):
        """Initialize the embedder.
        
        Args:
            hidden_size: Output embedding dimension
            frequency_embedding_size: Dimension of sinusoidal embeddings
            use_mlp: Whether to use MLP on top of frequency embeddings
        """
        super().__init__()
        self.use_mlp = use_mlp
        self.frequency_embedding_size = frequency_embedding_size
        
        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(frequency_embedding_size, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
        else:
            assert frequency_embedding_size == hidden_size, \
                "frequency_embedding_size must equal hidden_size when use_mlp=False"
    
    @staticmethod
    def embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal embeddings.
        
        Args:
            t: Input scalars [B]
            dim: Output embedding dimension
            max_period: Controls the minimum frequency of the embeddings
        
        Returns:
            Embeddings [B, dim]
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]  # (B, D/2)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, D)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            t: Input scalars [B]
        
        Returns:
            Embeddings [B, hidden_size]
        """
        t_freq = self.embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq) if self.use_mlp else t_freq

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
        """Output shape: [B, D, dim]"""
        if x.ndim == 2: # [B, D], values in range(N)
            return self.embedding[x]
        elif x.ndim == 3: # [B, D, N], last dimension sums to 1
            return torch.matmul(x.to(dtype=self.embedding.dtype), self.embedding)
        else:
            raise ValueError(f"Invalid input shape {x.shape}, expected 2D or 3D tensor.")


class RopeVIT(vit_models):
    def __init__(self, rope_theta=100.0, rope_mixed=False, use_ape=False, 
                 vocab_size=3, patch_size=1, in_chans=1, num_classes=0,
                 dtype='float16', **kwargs):
        """
        For Ising model learning:
            img_size: L, D = L ** 2
            vocab_size is always 3 (vocab 0, 1, 2=mask)
            patch_size and in_chans are always 1
            num_classes is always 0 (we don't use this feature)
        """
        super().__init__(patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, **kwargs)
        
        img_size = kwargs['img_size'] if 'img_size' in kwargs else 224
        # patch_size = kwargs['patch_size'] if 'patch_size' in kwargs else 16
        num_heads = kwargs['num_heads'] if 'num_heads' in kwargs else 12
        embed_dim = kwargs['embed_dim'] if 'embed_dim' in kwargs else 768
        mlp_ratio = kwargs['mlp_ratio'] if 'mlp_ratio' in kwargs else 4.
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        
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
        
        # The following are new for Ising model
        self.vocab_size = vocab_size
        self.length = img_size ** 2
        self.vocab_embed = VocabEmbedding(dim=self.embed_dim, vocab_dim=self.vocab_size)
        self.head = nn.Linear(self.embed_dim, self.vocab_size)
        self.dtype = {'float64': torch.float64,
                      'float32': torch.float32,
                      'float16': torch.float16,
                      'bfloat16': torch.bfloat16}.get(dtype, torch.float16)
        
        # Thermodynamic conditioning (temperature and field)
        # Match snowyflow's approach: separate embedders, concatenate embeddings
        # Each gets embed_dim // 4, concatenated gives embed_dim // 2, then we project to embed_dim
        self.thermo_embed_dim = embed_dim // 4  # Can be made configurable
        self.beta_embedder = ThermodynamicEmbedder(
            hidden_size=self.thermo_embed_dim,
            frequency_embedding_size=embed_dim // 2,
            use_mlp=True,  # Use MLP for better expressiveness
        )
        self.h_embedder = ThermodynamicEmbedder(
            hidden_size=self.thermo_embed_dim,
            frequency_embedding_size=embed_dim // 2,
            use_mlp=True,
        )
        # Project concatenated embeddings to full embed_dim for addition
        # Concatenated: [B, thermo_embed_dim * 2] = [B, embed_dim // 2] -> [B, embed_dim]
        self.thermo_proj = nn.Linear(self.thermo_embed_dim * 2, embed_dim)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'freqs', 'beta_embedder', 'h_embedder', 'thermo_proj'}
        
    def forward_features(self, x, beta=None, h=None):
        B = x.shape[0]; H = W = int(x.shape[1] ** 0.5)
        # x = self.patch_embed(x) # we don't use patch embedding, but use vocab embedding instead
        x = self.vocab_embed(x) # [B, D, embed_dim]

        # Add thermodynamic conditioning if provided
        # Concatenate beta and h embeddings, then project (similar to snowyflow)
        if beta is not None or h is not None:
            # Convert beta to temperature for embedding (beta = 1/T, so T = 1/beta)
            if beta is not None:
                temp = 1.0 / beta  # [B]
                beta_emb = self.beta_embedder(temp)  # [B, thermo_embed_dim]
            else:
                # If beta not provided, use zeros (for backward compatibility)
                beta_emb = torch.zeros(x.shape[0], self.thermo_embed_dim, device=x.device, dtype=x.dtype)
            
            if h is not None:
                h_emb = self.h_embedder(h)  # [B, thermo_embed_dim]
            else:
                # If h not provided, use zeros (for backward compatibility)
                h_emb = torch.zeros(x.shape[0], self.thermo_embed_dim, device=x.device, dtype=x.dtype)
            
            # Concatenate embeddings and project to full embed_dim
            combined_emb = torch.cat([beta_emb, h_emb], dim=-1)  # [B, thermo_embed_dim * 2]
            combined_emb = self.thermo_proj(combined_emb)  # [B, embed_dim]
            combined_emb = combined_emb.unsqueeze(1).expand(-1, x.shape[1], -1)  # [B, D, embed_dim]
            x = x + combined_emb

        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.use_ape:
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
        
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.rope_mixed:
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
    
    def logits(self, x, beta=None, h=None):
        """
        input: x: [B, D], values in range(N) or [B, D, N], last dimension sums to 1
               beta: [B] tensor of inverse temperatures (optional)
               h: [B] tensor of field values (optional)
        output: logits [B, D, N] (not log-softmaxed for non-mask positions)
        """
        with torch.amp.autocast('cuda', dtype=self.dtype):
            x = self.forward_features(x, beta=beta, h=h) # [B, embed_dim]
            x = self.head(x)
            # [B, D * N] -> [B, D, N]
        return x[:, 1:, :]
    
    def forward(self, x, beta=None, h=None):
        x = self.logits(x, beta=beta, h=h)
        x[:, :, :-1] = x[:, :, :-1].log_softmax(dim=-1)
        return x



def hf_checkpoint_load(model_name):
    try:
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(
            repo_id="naver-ai/" + model_name, filename= "pytorch_model.bin"
        )
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    except:
        _HF_URL = "https://huggingface.co/naver-ai/" + model_name + "/resolve/main/pytorch_model.bin"
        checkpoint = torch.hub.load_state_dict_from_url(_HF_URL, map_location='cpu')

    state_dict = checkpoint['model']
    for k in ['freqs_t_x', 'freqs_t_y']:
        if k in state_dict:
            print(f"Removing key {k} from pretrained checkpoint")
            del state_dict[k]
            
    return checkpoint

def adjust_pos_embed_size(model, state_dict):
    
    # interpolate position embedding
    if 'pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        state_dict['pos_embed'] = new_pos_embed
        
    return state_dict


def get_rope_vit_model(L, embed_dim=128, depth=8, num_heads=8, vocab_size=3, device='cuda', **kwargs):
    return RopeVIT(
        img_size=L, patch_size=1, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        mlp_ratio=4, qkv_bias=True, in_chans=1, num_classes=0, vocab_size=vocab_size,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=10.0, rope_mixed=True, use_ape=True, **kwargs).to(device)

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
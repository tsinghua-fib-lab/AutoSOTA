"""
Adapted from the following code:

https://github.com/yuchen-zhu-zyc/MDNS/blob/main/model/vit_rope.py

https://github.com/willisma/SiT/blob/main/models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import Mlp, Attention
from timm.layers import DropPath, trunc_normal_

from .vit_rope import VocabEmbedding, TimestepEmbedder, apply_rotary_emb, compute_axial_cis, compute_mixed_cis, init_random_2d_freqs, init_t_xy
from .sit import get_2d_sincos_pos_embed


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RoPEAttention(Attention):
    """Multi-head Attention block with rotary position embeddings."""
    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RoPE_Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=RoPEAttention,
                 Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x, freqs_cis, **kwargs):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), freqs_cis=freqs_cis))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class RoPE_Layer_scale_init_Block_adaLN_modulation(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=RoPEAttention,
                 Mlp_block=Mlp, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x, freqs_cis, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), freqs_cis=freqs_cis))
        x = x + self.drop_path(gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


class FinalLayer_adaLN_modulation(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT without conditioning.
    """
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)

    def forward(self, x, **kwargs):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class RopeSiT(nn.Module):
    def __init__(self, img_size=224, embed_dim=768, depth=12,
                 num_heads=12, vocab_size=2, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 block_layers=[RoPE_Layer_scale_init_Block_adaLN_modulation, RoPE_Layer_scale_init_Block],
                 head_layers=[FinalLayer_adaLN_modulation, FinalLayer],
                 act_layer=nn.GELU, Attention_block=RoPEAttention, Mlp_block=Mlp,
                 initime_scale=1e-4, rope_theta=100.0, rope_mixed=False, use_ape=False, require_time=False):
        """
        For Ising model / Potts model:
            img_size = L, seq_len = L ^ 2
        """
        super().__init__()
        self.img_size = img_size
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = drop_rate
        
        self.vocab_size = vocab_size
        self.vocab_embed = VocabEmbedding(dim=embed_dim, vocab_dim=vocab_size)
     
        self.require_time = require_time
        if require_time:
            self.time_embed = TimestepEmbedder(hidden_size=embed_dim)
            self.time_scale = 1000
            block_layer, head_layer = block_layers[0], head_layers[0]
        else:
            block_layer, head_layer = block_layers[1], head_layers[1]
        
        self.use_ape = use_ape
        if self.use_ape:
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size ** 2, embed_dim))
            pos_embed = get_2d_sincos_pos_embed(self.embed_dim, self.img_size)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.blocks = nn.ModuleList([
            block_layer(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                drop=0.0, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block, init_values=initime_scale)
            for _ in range(depth)])
        self.head = head_layer(embed_dim, vocab_size)
    
        self.rope_mixed = rope_mixed
        if self.rope_mixed: # go this way in our settings
            self.compute_cis = partial(compute_mixed_cis, num_heads=num_heads)
            
            freqs = []
            for i, _ in enumerate(self.blocks):
                freqs.append(
                    init_random_2d_freqs(dim=embed_dim // num_heads, num_heads=num_heads, theta=rope_theta)
                )
            freqs = torch.stack(freqs, dim=1).view(2, len(self.blocks), -1)
            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
            
            t_x, t_y = init_t_xy(end_x = img_size, end_y = img_size)
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)
        else:
            self.compute_cis = partial(compute_axial_cis, dim=embed_dim//num_heads, theta=rope_theta)
            
            freqs_cis = self.compute_cis(end_x = img_size, end_y = img_size)
            self.freqs_cis = freqs_cis

        # self.initialize_weights()
        # Not recommended to initialize weights at the start of training as this may lead to slower convergence

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        if self.require_time:
            trunc_normal_(self.time_embed.mlp[0].weight, std=0.02)
            trunc_normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        if self.require_time:
            for block in self.blocks:
                trunc_normal_(block.adaLN_modulation[-1].weight, std=0.02)
                if block.adaLN_modulation[-1].bias is not None:
                   nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        if self.require_time:
            trunc_normal_(self.head.adaLN_modulation[-1].weight, std=0.02)
            if self.head.adaLN_modulation[-1].bias is not None:
                nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        trunc_normal_(self.head.linear.weight, std=0.02)
        if self.head.linear.bias is not None:
            nn.init.constant_(self.head.linear.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'freqs'}
        
    @torch.amp.autocast('cuda', dtype=torch.bfloat16)
    def forward(self, x, t=None, **kwargs):
        """
        Args:
            x: shape (B, D=L^2), values in range(N)
            t: if require time input, shape (B,) or (B, 1), values in [0, 1]; otherwise, do not pass

        Returns:
            shape (B, D+1, embed_dim)
        """
        x = self.vocab_embed(x) # [B, D, embed_dim]

        if t is None and not self.require_time:
            c = None
        elif t is not None and self.require_time:
            t = t.view(-1) * self.time_scale
            c = self.time_embed(t) # [B, embed_dim]
        else:
            raise ValueError

        if self.use_ape: # go this way in our settings
            x = x + self.pos_embed
        
        if self.rope_mixed: # go this way in our settings
            if self.freqs_t_x.shape[0] != x.shape[1]:
                t_x, t_y = init_t_xy(end_x = self.img_size, end_y = self.img_size)
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            else:
                t_x, t_y = self.freqs_t_x, self.freqs_t_y
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
            
            for i , blk in enumerate(self.blocks):
                x = blk(x, freqs_cis=freqs_cis[i], c=c)
        else:
            if self.freqs_cis.shape[0] != x.shape[1]:
                freqs_cis = self.compute_cis(end_x = self.img_size, end_y = self.img_size)
            else:
                freqs_cis = self.freqs_cis
            freqs_cis = freqs_cis.to(x.device)
            
            for i , blk in enumerate(self.blocks):
                x = blk(x, freqs_cis=freqs_cis, c=c)
                
        x = self.head(x, c=c) # [B, D, N]
        return x
    

def get_rope_sit_model(args, require_time):
    # separate require_time for controller and corrector
    return RopeSiT(img_size=args.L, embed_dim=args.model.hidden_size, depth=args.model.n_blocks,
                   num_heads=args.model.n_heads, vocab_size=args.vocab_size, mlp_ratio=4,
                   qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                   rope_theta=10.0, rope_mixed=True, use_ape=True,
                   require_time=require_time)
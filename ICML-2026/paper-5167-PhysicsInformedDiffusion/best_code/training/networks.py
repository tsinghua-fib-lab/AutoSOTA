# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from torch_utils import persistence
from torch.nn.functional import silu
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange


#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

@persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

#----------------------------------------------------------------------------
# Group normalization.

@persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

@persistence.persistent_class
class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)


    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

@persistence.persistent_class
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
@persistence.persistent_class
    
class  ViT3D(nn.Module):
    def __init__(self,
                  img_resolution,    # spatial size (e.g. 32)
                  in_channels,       # ignored here: we assume input will be unsqueezed to (B,1,T,H,W)
                  out_channels,      # 
                  patch_size=(2, 4, 4),
                  stride=(1, 2, 2),
                  embed_dim=128,
                  depth=8,
                  num_heads=8,
                  mlp_ratio=4.0,
                  noise_channels=128,   # for noise embedding
                  label_dim=0,
                  augment_dim=0,
                  num_classes=0,
                  label_dropout=0.0,
                  ):
        super().__init__()

        # store
        self.img_resolution = img_resolution
        self.time_size = in_channels
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.label_dropout = label_dropout
        self.num_classes = num_classes
        self.noise_channels = noise_channels

        # ---- SongUNet-style noise / label embeddings ----
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
        self.map_label = nn.Linear(label_dim, noise_channels) if label_dim else None
        self.map_augment = nn.Linear(augment_dim, noise_channels, bias=False) if augment_dim else None
        self.map_layer0 = nn.Linear(noise_channels, noise_channels)
        self.map_layer1 = nn.Linear(noise_channels, noise_channels)

        # ---- patch embed (Conv3d) ----
        pt, ph, pw = patch_size
        st, sh, sw = stride

        # we assume single-channel input (B,1,T,H,W). If you have more channels set in_channels appropriately.
        self.patch_embed = nn.Conv3d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=(pt, ph, pw),
            stride=(st, sh, sw),
            bias=True
        )

        # compute expected number of patches along each axis (init-time)
        # formula: floor((input - kernel)/stride) + 1  (no padding)
        time_size = in_channels  # or infer from input instead of hard-coding
        n_t = (time_size - patch_size[0]) // stride[0] + 1
        self.n_t = (time_size - pt) // st + 1
        self.n_h = (img_resolution - ph) // sh + 1
        self.n_w = (img_resolution - pw) // sw + 1
        self.num_patches = self.n_t * self.n_h * self.n_w
        
        # class token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # transformer (we'll use a stack of encoder layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # project conditioning (noise emb) to token dim
        self.cond_to_token = nn.Linear(noise_channels, embed_dim)

        # projection from token space back to patch values
        self.token_to_patch = nn.Linear(embed_dim, embed_dim)  # intermediate
        # inverse of patch embedding: ConvTranspose3d to reconstruct (B,1,T,H,W)
        self.reconstruct = nn.ConvTranspose3d(
            in_channels=embed_dim,
            out_channels=1,
            kernel_size=(pt, ph, pw),
            stride=(st, sh, sw),
            bias=True
        )

        # initialize small values for pos token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.xavier_uniform_(self.reconstruct.weight)

    def forward(self, x, noise_labels, class_labels=None, augment_labels=None):
        """
        x: (B, T, H, W)  -- interleaved Re/Im along T
        noise_labels: (B,) or (B,1)
        returns: (B, T, H, W)
        """
        # ensure channel dim for Conv3d
        if x.ndim == 4:
            x_in = x.unsqueeze(1)   # (B,1,T,H,W)
        elif x.ndim == 5:
            x_in = x
        else:
            raise ValueError("x must be (B,T,H,W) or (B,C,T,H,W)")

        B = x_in.shape[0]

        # ---- SongUNet noise embedding pipeline ----
        if noise_labels.ndim == 2 and noise_labels.shape[1] == 1:
            noise_labels = noise_labels.squeeze(1)
        emb = self.map_noise(noise_labels)  # (B, noise_channels)
        #emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)  # sin/cos swap
        
        half = emb.shape[1] // 2
        emb = torch.cat([emb[:, half:], emb[:, :half]], dim=1)  # (B, noise_channels)        
        if self.map_label is not None and class_labels is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                mask = (torch.rand([B, 1], device=x_in.device) >= self.label_dropout).to(tmp.dtype)
                tmp = tmp * mask
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))

        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)

        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))   # (B, noise_channels)

        # project conditioning to token dim
        cond = self.cond_to_token(emb)    # (B, embed_dim)

        # ---- patch embedding (Conv3d) ----
        patches = self.patch_embed(x_in)  # (B, D, Nt, Nh, Nw)
        _, D, Nt, Nh, Nw = patches.shape
        # flatten to token sequence
        tokens = patches.permute(0, 2, 3, 4, 1).contiguous().view(B, Nt * Nh * Nw, D)  # (B, Ntokens, D)

        # If pos_embed was created with different num_patches, interpolate it to current shape
        current_n = tokens.shape[1]
        expected_n = self.num_patches
        if current_n != expected_n:
            # interpolate positional embeddings (exclude cls token)
            # reshape stored pos_embed[:,1:] to (1, D, nt, nh, nw) using stored dims expected_n -> self.n_t, self.n_h, self.n_w
            old_nt, old_nh, old_nw = self.n_t, self.n_h, self.n_w
            old_grid = self.pos_embed[:, 1:, :].reshape(1, old_nt, old_nh, old_nw, -1).permute(0, 4, 1, 2, 3)
            # interpolate to new grid (Nt,Nh,Nw)
            new_grid = nn.functional.interpolate(old_grid, size=(Nt, Nh, Nw), mode='trilinear', align_corners=False)
            new_pos = new_grid.permute(0, 2, 3, 4, 1).reshape(1, Nt * Nh * Nw, -1)  # (1, Nnew, D)
            pos0 = self.pos_embed[:, :1, :].detach().clone()  # cls token pos
            self.pos_embed = nn.Parameter(torch.cat([pos0, new_pos], dim=1).to(self.pos_embed.device))

        # add cls token and pos emb
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 1 + N, D)
        tokens = tokens + self.pos_embed.to(tokens.device)

        # inject conditioning
        tokens = tokens + cond.unsqueeze(1)

        # ---- transformer ----
        encoded = self.encoder(tokens)  # (B, 1 + N, D)

        # ---- reconstruct patches from tokens (use token outputs, not only cls) ----
        # drop cls token
        token_patches = encoded[:, 1:, :]  # (B, N, D)
        # map tokens back to patch "feature maps"
        token_feats = self.token_to_patch(token_patches)  # (B, N, D)
        # reshape to (B, D, Nt, Nh, Nw)
        token_feats = token_feats.view(B, Nt, Nh, Nw, D).permute(0, 4, 1, 2, 3).contiguous()

        # invert patch embedding via ConvTranspose3d
        recon = self.reconstruct(token_feats)  # (B, 1, T, H, W)
        recon = recon.squeeze(1)  # (B, T, H, W)

        return recon 
    
SongUNet = ViT3D # Back-compat alias so old .pkl files (pickled as 'SongUNet') still load.   

########################################################################################
class ComplexConv2d(nn.Module):
    """Complex-aware 2D convolution.
       Input: (B,4,H,W) -> (B,4,H,W) preserving real/imag structure.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        # treat real/imag pairs as in_ch/2 complex channels
        self.real_conv = nn.Conv2d(in_ch//2, out_ch//2, kernel_size, padding=padding)
        self.imag_conv = nn.Conv2d(in_ch//2, out_ch//2, kernel_size, padding=padding)

    def forward(self, x):
        xr, xi = x.chunk(2, dim=1)
        # (a+ib)(wR+i wI) = (a*wR - b*wI) + i(a*wI + b*wR)
        real = self.real_conv(xr) - self.imag_conv(xi)
        imag = self.real_conv(xi) + self.imag_conv(xr)
        return torch.cat([real, imag], dim=1)

class SpectralMixer(nn.Module):
    """Learnable complex gain per frequency."""
    def __init__(self, channels, size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, channels, size, size) * 0.01)

    def forward(self, x):
        return x * self.weight

class CrossAttentionBlock(nn.Module):
    """Cross-attn from solution (query) to coefficient (key/value)."""
    def __init__(self, ch, heads=4):
        super().__init__()
        self.key_proj = nn.Conv2d(2, ch, 1)   # <-- project 2-channel coeff to ch
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, coeff, sol):
        B,C,H,W = sol.shape
        coeff_proj = self.key_proj(coeff)
        q = sol.flatten(2).permute(0,2,1)        # (B,HW,ch)
        k = coeff_proj.flatten(2).permute(0,2,1) # (B,HW,ch)
        v = coeff_proj.flatten(2).permute(0,2,1)
        out,_ = self.attn(q,k,v)
        out = out.permute(0,2,1).reshape(B,C,H,W)
        return out + sol

class ComplexUNetBlock(nn.Module):
    """
    Complex-aware residual block with optional FiLM conditioning and attention.
    """
    def __init__(self, in_ch, out_ch, emb_ch, up=False, down=False, attention=False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.emb_ch = emb_ch
        self.attention = attention

        self.norm0 = nn.GroupNorm(8, in_ch)
        self.conv0 = ComplexConv2d(in_ch, out_ch, 3, padding=1)

        # FiLM: output 2*channels for scale/shift
        self.film = nn.Linear(emb_ch, out_ch * 2)

        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv1 = ComplexConv2d(out_ch, out_ch, 3, padding=1)

        # Skip connection
        if in_ch != out_ch:
            self.skip = ComplexConv2d(in_ch, out_ch, 1, padding=0)
        else:
            self.skip = nn.Identity()

        # Optional attention
        if attention:
            self.attn_norm = nn.GroupNorm(8, out_ch)
            self.attn = nn.MultiheadAttention(out_ch, num_heads=4, batch_first=True)
        else:
            self.attn = None

    def forward(self, x, emb):
        orig = x
        h = self.conv0(F.silu(self.norm0(x)))

        # FiLM conditioning
        gamma_beta = self.film(emb).unsqueeze(-1).unsqueeze(-1)  # [B, 2*out_ch, 1, 1]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        h = F.silu(gamma * self.norm1(h) + beta)

        h = self.conv1(h)
        h = h + self.skip(orig)  # Residual

        # Attention if requested
        if self.attn is not None:
            B, C, H, W = h.shape
            h_flat = h.flatten(2).permute(0,2,1)  # (B, HW, C)
            attn_out, _ = self.attn(h_flat, h_flat, h_flat)
            h = attn_out.permute(0,2,1).reshape(B,C,H,W)
        return h

# ---------- SongUNet adaptation ----------

@persistence.persistent_class
class SongUNetFourier(nn.Module):
    def __init__(self,
                 img_resolution   = 32,
                 in_channels      = 4,
                 out_channels     = 4,
                 model_channels   = 128,
                 channel_mult     = [1,2,2,2],
                 num_blocks       = 2,
                 attn_resolutions = [8],
                 dropout          = 0.1,
                 **kwargs):
        super().__init__()
        
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        
        # ---- Noise embedding (FiLM) ----
        self.map_noise = nn.Sequential(
            nn.Linear(1, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels)
        )
        
        # ---- Encoder ----
        self.enc = nn.ModuleList()
        ch = in_channels
        self.skip_channels = []
        for mult in channel_mult:
            out_ch = model_channels * mult
            self.enc.append(ComplexUNetBlock(ch, out_ch, model_channels))
            self.skip_channels.append(out_ch)
            ch = out_ch
        
        # ---- Bottleneck cross-attention ----
        self.cross_attn = CrossAttentionBlock(ch)
        
        # ---- Decoder ----
        self.dec = nn.ModuleList()
        reversed_skips = list(reversed(self.skip_channels))
        decoder_ch = ch
        for skip_ch in reversed_skips:
            in_ch = decoder_ch + skip_ch
            out_ch = skip_ch
            self.dec.append(ComplexUNetBlock(in_ch, out_ch, model_channels))
            decoder_ch = out_ch
        
        # ---- Output ----
        self.out_conv = ComplexConv2d(decoder_ch, out_channels, 3, padding=1)
    
    def forward(self, x, noise_labels=None, class_labels=None, augment_labels=None):
        # Split coeff / solution
        coeff, sol = torch.chunk(x, 2, dim=1)
        h = torch.cat([coeff, sol], dim=1)
        
        # Noise embedding -> FiLM
        noise_emb = self.map_noise(noise_labels.view(noise_labels.shape[0], -1))  # [B, model_channels]
        
        # ---- Encoder ----
        skips = []
        for block in self.enc:
            h = block(h, noise_emb)
            skips.append(h)
        
        # ---- Bottleneck cross-attention ----
        h = self.cross_attn(coeff, h)
        
        # ---- Decoder ----
        for block, skip in zip(self.dec, reversed(skips)):
            if h.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=h.shape[-2:], mode='bilinear', align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h, noise_emb)
        
        # ---- Output ----
        out = self.out_conv(h)
        return out




#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                 # Image resolution.
        img_channels,                   # Number of color channels.
        label_dim       = 0,            # Number of class labels, 0 = unconditional.
        use_fp16        = False,        # Execute the underlying model at FP16 precision?
        beta_d          = 19.9,         # Extent of the noise level schedule.
        beta_min        = 0.1,          # Initial slope of the noise level schedule.
        M               = 1000,         # Original number of timesteps in the DDPM formulation.
        epsilon_t       = 1e-5,         # Minimum t-value used during training.
        model_type      = 'SongUNet',   # Class name of the underlying model.
        **model_kwargs,                 # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VEPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                 # Image resolution.
        img_channels,                   # Number of color channels.
        label_dim       = 0,            # Number of class labels, 0 = unconditional.
        use_fp16        = False,        # Execute the underlying model at FP16 precision?
        sigma_min       = 0.02,         # Minimum supported noise level.
        sigma_max       = 100,          # Maximum supported noise level.
        model_type      = 'SongUNet',   # Class name of the underlying model.
        **model_kwargs,                 # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models".

@persistence.persistent_class
class iDDPMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        C_1             = 0.001,            # Timestep adjustment at low noise levels.
        C_2             = 0.008,            # Timestep adjustment at high noise levels.
        M               = 1000,             # Original number of timesteps in the DDPM formulation.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels*2, label_dim=label_dim, **model_kwargs)

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x[:, :self.img_channels].to(torch.float32)
        return D_x

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)

#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMPrecond(torch.nn.Module):
    _MODEL_TYPE_ALIASES = {
        'SongUNet': 'ViT3D',   # legacy
    }
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        # Resolve legacy model_type strings
        model_type = self._MODEL_TYPE_ALIASES.get(model_type, model_type)
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------

from typing import Dict, Tuple, List, Optional
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import Mlp
from torch import nn


approx_gelu = lambda: nn.GELU(approximate="tanh") 



class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5, bias=False, device=None):
        """Root Mean Square Layer Normalization
        :param d: model size
        :param eps:  epsilon value, default 1e-5
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.bias = bias
        self.device = device

        self.scale = nn.Parameter(torch.ones(d, device=device))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d, device=device))
            self.register_parameter("offset", self.offset)

    def _layer_impl(self, x, scale, offset, eps):
        x_dtype = x.dtype
        with torch.autocast("cuda", enabled=False):
            x = x.to(torch.float32)
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d

            rms_x = norm_x * d_x ** (-1.0 / 2)
            x_normed = x / (rms_x + eps)
        x_normed = x_normed.to(x_dtype)

        if self.bias:
            output = scale * x_normed + offset
        else:
            output = scale * x_normed
        output = output.to(x_dtype)
        return output

    def forward(self, x):
        offset = None
        if self.bias:
            offset = self.offset
        output = self._layer_impl(x, self.scale, offset, self.eps)
        return output

    def reset_parameters(self):
        torch.nn.init.ones_(self.scale)  # type: ignore
        if self.bias:
            torch.nn.init.zeros_(self.offset)


def modulate(norm_func, x, shift, scale, force_fp32=False):
    """Modulates the given activation using this shift and scale values
    :param norm_func: The normalization function to be used
    :param x: (batch_size, sequence_length, channels) The input tensor
    :param shift: (batch_size, [sequence_length], channels) The shift tensor. If None, no shift is applied
    :param scale: (batch_size, [sequence_length], channels) The scale tensor. 1 is summed to the scale tensor, so 0 should be returned for no scaling
    :param force_fp32: Whether to force the normalization to be computed in FP32
    """
    # Suppose x is (B, N, D), shift is (B, D), scale is (B, D)
    # Runs the normalization layer in full precision. This should happen already for nn.LayerNorm
    if force_fp32:
        with torch.autocast("cuda", enabled=False):
            x = norm_func(x.to(torch.float32))
    else:
        x = norm_func(x)

    #
    if scale.ndim == 2:
        scale = scale.unsqueeze(1)
    if shift.ndim == 2:
        shift = shift.unsqueeze(1)
    x = x * (scale + 1)
    if shift is not None:
        x = x + shift
    return x

def get_layernorm(
    hidden_size: torch.Tensor,
    eps: float,
    affine: bool,
    use_kernel: bool,
    parallel_type: str = "sp",
):
    return nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        kv_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        out_proj_bias: bool = True,
        norm_layer_eps: float = 1e-5,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = RMSNorm,
        enable_flashattn: bool = False,
        save_qkv: bool = False
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn = enable_flashattn

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(kv_dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.dim, norm_layer_eps) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.dim, norm_layer_eps) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=out_proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
        
        self.attn_map_copy = {}
        self.save_qkv = save_qkv

    def forward(
        self,
        q_input: torch.Tensor,
        kv_input: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.q_proj(q_input)
        kv = self.kv_proj(kv_input)
        k, v = kv.chunk(2, dim=-1)
        # Appleis qk normalization before separating heads as it is faster
        q = self.q_norm(q)
        k = self.k_norm(k)
        

        q, k, v = map(  # noqa C417
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q, k, v),
        )

        # Maybe save QKV projections for visualization
        if self.save_qkv:
            self.attn_map_copy["q"] = q.clone().permute(0, 2, 1, 3)
            self.attn_map_copy["k"] = k.clone().permute(0, 2, 1, 3)
            self.attn_map_copy["v"] = v.clone().permute(0, 2, 1, 3)
            self.attn_map_copy["b"] = None # no attn bias

        if attn_mask is not None:
            # do not use flash attention if attn_mask is provided
            if attn_mask.ndim == 3:
                # If attn_mask is B x N x N, it seems that it is required to be B x 1 x N x N (for head dim)
                attn_mask = attn_mask.unsqueeze(1)
            x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )  # Scale is automatically computed by the torch implementation
            
            
        elif self.enable_flashattn:
            # Executes attention, forcing the FlashAttention 2 Implementation
            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False,
                ):
                # with nn.attention.sdpa_kernel(
                #     nn.attention.SDPBackend.FLASH_ATTENTION,
                # ):
                    x = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        dropout_p=self.attn_drop.p if self.training else 0.0,
                    )  # Scale is automatically computed by the torch implementation
            except RuntimeError:
                # Fallback to default SDPA if FlashAttention is not available
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )

        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            # Disables autocast for the softmax. Should happen already for softmax
            with torch.autocast("cuda", enabled=False):
                attn = attn.to(torch.float32)
                attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v


        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



def group_tokens_2d(x, thw_shape, window_size):
    """Given the input tokens of shape B x N x Z, group tokens by window_size, producing B x T/zt x H/zh x W/zw x zt x zh x zw x C, 
    where T is the temporal dimension, H/w is the height divided by window size, W/C is the width divided by window size.

    Args:
        x (_type_): _input tokens of shape B x N x C, where N is the number of tokens and C is the number of channels.
        thw_shape (_type_): T, H, W dimensions of the patchified input
        window_size (_type_): The size of the window to group tokens by. Can be integer (for 2D) or tuple (zt, zh, zw) for 3D.
    Returns:
        _type_: _reshaped tokens of shape B x T/zt x H/zh x W/zw x zt x zh x zw x C
    """
    assert len(thw_shape) == 3, "thw_shape must be a tuple of length 3 (T, H, W), got {}".format(thw_shape)
    assert x.ndim == 3, "Input tokens must be of shape B x N x C, got {}".format(x.shape)
    
    # Handle both integer (backward compatibility) and tuple window_size
    if isinstance(window_size, (int, float)):
        assert int(window_size) == window_size, "window_size must be an integer, got {}".format(window_size)
        zt, zh, zw = 1, int(window_size), int(window_size)
    else:
        assert len(window_size) == 3, "window_size must be a tuple of length 3 (zt, zh, zw), got {}".format(window_size)
        zt, zh, zw = window_size
    
    t, h, w = thw_shape
    C = x.shape[-1]
    tg, hg, wg = t // zt, h // zh, w // zw
    
    assert t % zt == 0, f"Temporal dimension {t} must be divisible by temporal window size {zt}"
    assert h % zh == 0, f"Height dimension {h} must be divisible by height window size {zh}"
    assert w % zw == 0, f"Width dimension {w} must be divisible by width window size {zw}"
    
    x = x.reshape(shape=(x.shape[0], t, h, w, C))
    x = x.reshape(shape=(x.shape[0], tg, zt, hg, zh, wg, zw, C))
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)  # B x T/zt x H/zh x W/zw x zt x zh x zw x C 
    return x


def group_tokens_2d_flatten(x, thw_shape, window_size):
    """Given the input tokens of shape B x N x Z, group tokens by window and flatten, producing B x T/zt x H/zh x W/zw x zt * zh * zw * C.

    Args:
        x (_type_): _input tokens of shape B x N x C, where N is the number of tokens and C is the number of channels.
        thw_shape (_type_): T, H, W dimensions of the patchified input
        window_size (_type_): The size of the window to group tokens by. Can be integer (for 2D) or tuple (zt, zh, zw) for 3D.
    Returns:
        _type_: _reshaped tokens of shape B x T/zt x H/zh x W/zw x zt * zh * zw * C
    """
    
    x = group_tokens_2d(x, thw_shape, window_size) # B x T/zt x H/zh x W/zw x zt x zh x zw x C
    x = rearrange(x, 'b t h w z0 z1 z2 c -> b t h w (z0 z1 z2 c)')  # B x T/zt x H/zh x W/zw x zt * zh * zw * C
    return x

def rearrange_tokens_by_group(x, thw_shape, window_size):
    """ Given the input tokens of shape B x N x C, rearrange to shape (B * T/zt * H/zh * W/zw, zt * zh * zw, C),

    Args:
        x (_type_): _input tokens of shape B x N x C, where N is the number of tokens and C is the number of channels.
        thw_shape (_type_): T, H, W dimensions of the patchified input
        window_size (_type_): The size of the window to group tokens by. Can be integer (for 2D) or tuple (zt, zh, zw) for 3D.

    Returns:
        _type_: _reshaped tokens of shape (B * T/zt * H/zh * W/zw, zt * zh * zw, C)
    """
    x = group_tokens_2d(x, thw_shape, window_size)  # B x T/zt x H/zh x W/zw x zt x zh x zw x C    
    x = rearrange(x, 'b t h w z0 z1 z2 c -> b t h w (z0 z1 z2) c')  # B x T/zt x H/zh x W/zw x zt * zh * zw x C
    x = rearrange(x, 'b t h w g c -> (b t h w) g c')  # (B * tg * hg * wg, zt * zh * zw, C)
    return x

        
def rearrange_tokens_by_batch(x, thw_shape, window_size):
    """ Given the input tokens of shape (B * T/zt * H/zh * W/zw, zt * zh * zw, C) rearrange to shape (B, N, C)
    """
    t, h, w = thw_shape
    
    # Handle both integer (backward compatibility) and tuple window_size
    if isinstance(window_size, (int, float)):
        zt, zh, zw = 1, int(window_size), int(window_size)
    else:
        zt, zh, zw = window_size
    
    tg, hg, wg = t // zt, h // zh, w // zw
    B = x.shape[0] // (tg * hg * wg)
    assert x.shape[0] % (tg * hg * wg) == 0, "The number of tokens must be divisible by the number of groups, got {} tokens and {} groups".format(x.shape[0], tg * hg * wg)
    assert x.ndim == 3, "Input tokens must be of shape (B * T/zt * H/zh * W/zw, zt * zh * zw, C), got {}".format(x.shape)
    assert x.shape[1] == zt * zh * zw, "Input tokens must be of shape (B * T/zt * H/zh * W/zw, zt * zh * zw, C), got {}".format(x.shape)
    
    x = rearrange(x, '(b t h w) g c -> b t h w g c', b=B, t=tg, h=hg, w=wg)  # (B, T/zt, H/zh, W/zw, zt * zh * zw, C)
    x = rearrange(x, 'b t h w (z0 z1 z2) c -> b t h w z0 z1 z2 c', z0=zt, z1=zh, z2=zw)
    x = ungroup_tokens_2d(x, thw_shape, window_size)  # (B, N, C)
    return x

def ungroup_tokens_2d(x, thw_shape, window_size):
    
    """Given the input tokens of shape B x T/zt x H/zh x W/zw x zt x zh x zw x C, rearrange to shape B x N x C, 
    where N is the number of tokens and C is the number of channels.

    Args:
        x (_type_): _input tokens of shape B x T/zt x H/zh x W/zw x zt x zh x zw x C
        thw_shape (_type_): T, H, W dimensions of the patchified input
        window_size (_type_): The size of the window to group tokens by. Can be integer (for 2D) or tuple (zt, zh, zw) for 3D.
    Returns:
        _type_: _reshaped tokens of shape B x N x C, where N is the number of tokens and C is the number of channels.
    """
    assert len(thw_shape) == 3, "thw_shape must be a tuple of length 3 (T, H, W), got {}".format(thw_shape)
    assert x.ndim == 8, "Input tokens must be of shape B x T/zt x H/zh x W/zw x zt x zh x zw x C, got {}".format(x.shape)
    
    # Handle both integer (backward compatibility) and tuple window_size
    if isinstance(window_size, (int, float)):
        zt, zh, zw = 1, int(window_size), int(window_size)
    else:
        assert len(window_size) == 3, "window_size must be a tuple of length 3 (zt, zh, zw), got {}".format(window_size)
        zt, zh, zw = window_size
    
    assert (x.shape[4] == zt and x.shape[5] == zh and x.shape[6] == zw), "Input tokens window dimensions must match window_size, got {} vs ({}, {}, {})".format(x.shape[4:7], zt, zh, zw)
    
    t, h, w = thw_shape
    C = x.shape[-1]
    tg, hg, wg = t // zt, h // zh, w // zw
    
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7)  # B x T/zt x zt x H/zh x zh x W/zw x zw x C
    x = rearrange(x, 'b t z0 h z1 w z2 c -> b (t z0) (h z1) (w z2) c')
    x = rearrange(x, 'b t h w c -> b (t h w) c')  # B x N x C
    return x


def ungroup_tokens_2d_unflatten(x, thw_shape, window_size):
    """Given the input tokens of shape B x T/zt x H/zh x W/zw x zt * zh * zw * C, rearrange to shape B x N x C, 
    where N is the number of tokens and C is the number of channels.""" 
    
    # Handle both integer (backward compatibility) and tuple window_size
    if isinstance(window_size, (int, float)):
        zt, zh, zw = 1, int(window_size), int(window_size)
    else:
        zt, zh, zw = window_size
    
    x = rearrange(x, 'b t h w (z0 z1 z2 c) -> b t h w z0 z1 z2 c', z0=zt, z1=zh, z2=zw)  # B x T/zt x H/zh x W/zw x zt x zh x zw x C
    x = ungroup_tokens_2d(x, thw_shape, window_size)  # B x N x C
    return x   



class SequentialInputTokenEditor(nn.Module):
    """
    Sequential token editor that applies a sequence of token adapters
    """
    def __init__(self, config):
        super().__init__()
        
        self.adapters = nn.ModuleList()
        for adapter_config in config["adapters"]:
            adapter_target = adapter_config['target']
            self.adapters.append(
                adapter_target(adapter_config)
            )
        
        self.enable_profiling = config.get("enable_profiling", False)
    def forward(self, x, thw_shape, block_kwargs={}, **kwargs):
        # TODO: improve the implementation once previous experiments are done
        for idx, adapter in enumerate(self.adapters):
            x, thw_shape, block_kwargs = adapter(x, thw_shape, block_kwargs=block_kwargs, **kwargs)
        
        return x, thw_shape, block_kwargs
    
    def reset_parameters(self):
        for adapter in self.adapters:
            if hasattr(adapter, "reset_parameters"):
                adapter.reset_parameters()


class SequentialOutputTokenEditor(nn.Module):
    """
    Sequential token editor that applies a sequence of token adapters
    """
    def __init__(self, config):
        super().__init__()
        
        self.adapters = nn.ModuleList()
        for adapter_config in config["adapters"]:
            adapter_target = adapter_config['target']
            self.adapters.append(
                adapter_target(adapter_config)
            )
        self.enable_profiling = config.get("enable_profiling", False)
    def forward(self, x, thw_shape, block_kwargs={}, **kwargs):
        for idx, adapter in enumerate(self.adapters):
            x, thw_shape, block_kwargs = adapter(x, thw_shape, block_kwargs=block_kwargs, **kwargs)
    
        return x, thw_shape, block_kwargs
    
    def reset_parameters(self):
        for adapter in self.adapters:
            if hasattr(adapter, "reset_parameters"):
                adapter.reset_parameters()
    



class LatentTokensGroupAdapter(nn.Module):
    """
    process the input tokens for each window to copy the noise. Tokens are aggregated channel-wise.
    """
    def __init__(self, config):
        super().__init__()
        self.window_size = config["window_size"]
        self.patch_channels = config["patch_channels"]
        self.use_learnable_positional_encoding = config.get(
            "use_learnable_positional_encoding",
            True,
        )# default to True: when using RoPE, input patches have no positional information and laernable pos emb is needed
        self.use_latent_learnable_positional_encoding = config.get(
            "use_latent_learnable_positional_encoding",
            True,
        )# when not using RoPE, latent tokens have no positional information and learnable pos emb is needed
        
        self.fit_layer_config = config.get("fit_layer_config", {})
        
        # default is to use latent tokens equal to the patch tokens and drop later. Such design introduces a small overhead in multibudget settings.
        if isinstance(self.window_size, (int, float)):
            window_tokens = int(self.window_size) ** 2  # For backward compatibility: 1 * z * z
        else:
            zt, zh, zw = self.window_size
            window_tokens = zt * zh * zw  # For 3D: zt * zh * zw
        self.num_tokens = window_tokens
            
        
        hidden_size = self.patch_channels
        
        
        assert self.num_tokens == window_tokens, f"The case where num_tokens != window_tokens is not supported yet. Expected {window_tokens}, got {self.num_tokens}. Please make sure to adjust the RoPE accordingly"
            
        self.learned_query_embed = nn.Parameter(torch.zeros(self.num_tokens,  hidden_size), requires_grad=True)
        if self.use_learnable_positional_encoding:
            self.pos_embed = nn.Parameter(
                torch.randn((self.num_tokens, hidden_size)) * 0.02,
            )
        if self.use_latent_learnable_positional_encoding:
            max_tokens = 64 * 64 # maximum number of tokens in a single image, assuming 512x512 input size
            self.latent_pos_embed = nn.Parameter(
                torch.randn((max_tokens, hidden_size)) * 0.02,
            )
        
        fit_layer_target = self.fit_layer_config.get("target")
        self.fit_layer = fit_layer_target(self.fit_layer_config)

        
    def forward(self, patches, thw_shape, block_kwargs={}, **kwargs):
        # patches: (B, N, C)
        B, N, C = patches.shape
        
        patches = rearrange_tokens_by_group(patches, thw_shape, self.window_size)  # (B * tg * hg * wg, 1 * z * z, C)
        
        # add positional encoding if needed
        if self.use_learnable_positional_encoding:
            pos_embed = self.pos_embed[None, :patches.shape[1], :]
            patches = patches + pos_embed.to(patches.dtype)
        
        
        # prepare latent tokens
        latent_tokens = self.learned_query_embed[None, :, :]  # (1, 1 * z * z, C)
        latent_tokens = latent_tokens.repeat(patches.shape[0], 1, 1) # (B * tg * hg * wg, 1 * z * z, C)
        latent_tokens = latent_tokens.to(patches.dtype)
            
        latent_tokens = rearrange_tokens_by_batch(latent_tokens, thw_shape, self.window_size)  # (B, N, C)
        patches = rearrange_tokens_by_batch(patches, thw_shape, self.window_size)  # (B, N, C)

        block_kwargs['patches'] = patches
        latent_tokens, thw_shape, block_kwargs = self.fit_layer(latent_tokens, thw_shape, block_kwargs=block_kwargs) 
        
        
        # apply positional encoding if needed
        if self.use_latent_learnable_positional_encoding:
            latent_tokens = latent_tokens + self.latent_pos_embed[None, :latent_tokens.shape[1], :]
            
        return latent_tokens, thw_shape, block_kwargs
    
    def reset_parameters(self):
        """
        Reset the parameters of the model.
        """
        nn.init.normal_(self.learned_query_embed, std=0.02)  
        
        if self.use_learnable_positional_encoding:
            nn.init.normal_(self.pos_embed, std=0.02)
        if self.use_latent_learnable_positional_encoding:
            nn.init.normal_(self.latent_pos_embed, std=0.02)


 
class FiTLayer(nn.Module):
    """
    Mimic FiT design for the read layers 
    """
    def __init__(self, config):
        super().__init__()
        
        self.window_size = config["window_size"]
        self.patch_channels = config["patch_channels"]
        self.num_heads = config["num_heads"]
        self.qkv_bias = config.get("qkv_bias", True)
        
        if isinstance(self.window_size, (int, float)):
            window_tokens = int(self.window_size) ** 2
        else:
            zt, zh, zw = self.window_size
            window_tokens = zt * zh * zw
        
        self.depth = config.get("depth", 1) # increasing read depth does not help much
        self.mlp_ratio = config.get("mlp_ratio", 1.0) # increasing mlp_ratio is effective 
        self.use_read_residual = config.get("use_read_residual", True)
        self.enable_flashattn = config.get("enable_flashattn", True) # disable flash attention in case of group_size=1
        self.save_ca_qkv = config.get("save_ca_qkv", False) # legacy attention visualization option.
        hidden_size = self.patch_channels
        self.num_tokens = window_tokens

        assert self.num_tokens == window_tokens, f"The case where num_tokens != window_tokens is not supported yet. Expected {window_tokens}, got {self.num_tokens}. Please make sure to adjust the RoPE accordingly"
        
        # CA for the read operation
        self.latents_attend_to_patches = []
        self.read_ff = []
        for l in range(self.depth):
            attn_layer =  CrossAttention(
                dim=hidden_size,
                kv_dim=hidden_size,
                num_heads=self.num_heads,
                qkv_bias=self.qkv_bias,
                qk_norm=True,
                enable_flashattn=self.enable_flashattn,
                save_qkv=self.save_ca_qkv
            )
            self.latents_attend_to_patches.append(attn_layer)
            
            ff_layer = Mlp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * self.mlp_ratio),
                out_features=self.patch_channels,
                act_layer=approx_gelu,
                bias=False,
            )
            self.read_ff.append(ff_layer)
        self.read_ff = nn.ModuleList(self.read_ff)
        self.latents_attend_to_patches = nn.ModuleList(self.latents_attend_to_patches)

    def forward(self, latents, thw_shape, block_kwargs={}, **kwargs):
        # latents: (B, N, C) # patches: (B, N, C)
        patches = block_kwargs.get("patches", None)
        assert patches is not None, "patches must be provided in block_kwargs"
        
        B, N, C = patches.shape
        
        patches = rearrange_tokens_by_group(patches, thw_shape, self.window_size)  # (B * tg * hg * wg, 1 * z * z, C)
        latents = rearrange_tokens_by_group(latents, thw_shape, self.window_size)  # (B * tg * hg * wg, 1 * z * z, C)
        
        # read from patches
        for attn_layer, ff in zip(self.latents_attend_to_patches, self.read_ff):
            latents = attn_layer(latents, patches) + latents     
            latents = ff(latents) + latents
        
        # project the output
        latent_tokens = rearrange_tokens_by_batch(latents, thw_shape, self.window_size) # (B, N, C)
        patches = rearrange_tokens_by_batch(patches, thw_shape, self.window_size) # (B, N, C)
        
        # update patches and store them for the write layer
        block_kwargs["patches"] = patches
        
        assert latent_tokens.shape == (B, N, C), "Output shape does not match input shape, expected {}, got {}".format((B, N, C), latent_tokens.shape)
        
        return latent_tokens, thw_shape, block_kwargs
    
    def reset_parameters(self):
        """
        Reset the parameters of the model.
        """
        pass


class FiTLayerWModulation(FiTLayer):
    """
    FiT layer that applies a modulation layer to the input tokens before processing.
    """
    
    def __init__(self, config): 
        super().__init__(config)
        enable_layernorm_kernel = config.get("enable_layernorm_kernel", False)

        assert self.depth == 1, "FiTLayerWModulation is implemented for depth == 1 only"

        self.patches_norm = get_layernorm(
            self.patch_channels,
            eps=1e-6,
            affine=False,
            use_kernel=enable_layernorm_kernel,
        )
        
        self.latent_norm = get_layernorm(
            self.patch_channels,
            eps=1e-6,
            affine=False,
            use_kernel=enable_layernorm_kernel,
        )
        
        
    def forward(self, latents, thw_shape, block_kwargs={}, **kwargs):
        # x: (B, N, C) # patches: (B, N, C)
        patches = block_kwargs.get("patches", None)
        modulation_chunks = block_kwargs.get("modulation_chunks", None)
        
        assert patches is not None, "patches must be provided in block_kwargs"
        assert modulation_chunks is not None, "modulation_chunks must be provided in block_kwargs"
        
        B, N, C = patches.shape
        (
            shift_patches,
            scale_patches,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            _,
        ) = modulation_chunks
        
        if len(gate_attn.shape) == 2:
            gate_attn = gate_attn.unsqueeze(1) # B x 1 x D
            gate_mlp = gate_mlp.unsqueeze(1)
            shift_patches = shift_patches.unsqueeze(1)  # (B, 1, C)
            scale_patches = scale_patches.unsqueeze(1)  # (B, 1, C)
            shift_mlp = shift_mlp.unsqueeze(1)  # (B, 1, C)
            scale_mlp = scale_mlp.unsqueeze(1)
        
        # TODO: @Moayed inefficient implementation for now
        gate_attn = gate_attn.expand(-1, patches.shape[1], -1)  # (B, N, C)
        gate_mlp = gate_mlp.expand(-1, patches.shape[1], -1)  # (B, N, C)
        shift_patches = shift_patches.expand(-1, patches.shape[1], -1) # (B, N, C)
        scale_patches = scale_patches.expand(-1, patches.shape[1], -1) # (B, N, C)
        shift_mlp = shift_mlp.expand(-1, patches.shape[1], -1) # (B, N, C)
        scale_mlp = scale_mlp.expand(-1, patches.shape[1], -1) # (B, N, C)
        
        # rearrange by group 
        # (B, N, C) -> (B * tg * hg * wg, 1 * z * z, C)
        gate_attn = rearrange_tokens_by_group(gate_attn, thw_shape, self.window_size)  # (B * tg * hg * wg, 1 * z * z, C)
        gate_mlp = rearrange_tokens_by_group(gate_mlp, thw_shape, self.window_size)  # (B * tg * hg * wg, 1 * z * z, C)
        shift_patches = rearrange_tokens_by_group(shift_patches, thw_shape, self.window_size)  # (B * tg * hg * wg, 1 * z * z, C)
        scale_patches = rearrange_tokens_by_group(scale_patches, thw_shape, self.window_size)  # (B * tg * hg * wg, 1 * z * z, C)
        shift_mlp = rearrange_tokens_by_group(shift_mlp, thw_shape, self.window_size)  # (B * tg * hg * wg, 1 * z * z, C)
        scale_mlp = rearrange_tokens_by_group(scale_mlp, thw_shape, self.window_size)  # (B * tg * hg * wg, 1 * z * z, C)  
        
        
        patches = rearrange_tokens_by_group(patches, thw_shape, self.window_size)  # (B * tg * hg * wg, 1 * z * z, C)
        latents = rearrange_tokens_by_group(latents, thw_shape, self.window_size)  # (B * tg * hg * wg, 1 * z * z, C)
        
        
        # read from patches
        for attn_layer, ff in zip(self.latents_attend_to_patches, self.read_ff):
            # apply modulation (rearrange could be more efficient than replicating the modulation)
            modulated_patches = modulate(self.patches_norm, patches, shift_patches, scale_patches)    
            attn_output = gate_attn * attn_layer(latents, modulated_patches)  # (B * tg * hg * wg, 1 * z * z, C)
            latents = attn_output + latents
            
            
            modulated_latents = modulate(self.latent_norm, latents, shift_mlp, scale_mlp)
            mlp_output = gate_mlp * ff(modulated_latents)
            latents = mlp_output + latents
        
        latent_tokens = rearrange_tokens_by_batch(latents, thw_shape, self.window_size) # (B, N, C)
        patches = rearrange_tokens_by_batch(patches, thw_shape, self.window_size) # (B, N, C)
        
        # update patches
        block_kwargs["patches"] = patches
        
        assert latent_tokens.shape == (B, N, C), "Output shape does not match input shape, expected {}, got {}".format((B, N, C), latent_tokens.shape)
        
        return latent_tokens, thw_shape, block_kwargs
    
    
    
class FiTLayerWModulationLayer(FiTLayerWModulation):
    def __init__(self, config):
        super().__init__(config)
        hidden_size = self.patch_channels
        modulation_linear = nn.Linear(
                hidden_size,
                7 * hidden_size,
                bias=True,
            )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), modulation_linear)

    def forward(self, latents, thw_shape, block_kwargs={}, **kwargs):
        assert 'modulation_chunks' not in block_kwargs, "FiTLayerWModulationLayer assumes that modulation_chunks is not be provided in block_kwargs. There is potentially a bug"
        assert 'modulation' in block_kwargs, "modulation condition must be provided in block_kwargs"
        modulation = block_kwargs['modulation']
        block_kwargs['modulation_chunks'] = self.adaLN_modulation(modulation).chunk(7, dim=1)
        ret = super().forward(latents, thw_shape, block_kwargs, **kwargs)
        
        # remove modulation_chunks to avoid affecting other layers
        del block_kwargs['modulation_chunks']
        
        return ret
    




class FiTWriteLayer(nn.Module):
    """
    Mimic FiT design but with self-attention instead of Cross-attention.
    This layer is used to write the latent tokens back to the input tokens.
    """
    def __init__(self, config):
        super().__init__()
        
        self.window_size = config["window_size"]
        self.patch_channels = config["patch_channels"]
        self.num_heads = config["num_heads"]
        self.qkv_bias = config.get("qkv_bias", True)
        self.depth = config.get("depth", 1)
        self.mlp_ratio = config.get("mlp_ratio", 1.0) # increasing mlp_ratio is effective
        self.use_learnable_positional_encoding = config.get(
            "use_learnable_positional_encoding",
            False,
        ) # in default settings where patches carried by the residuals are not manipulated, they will have the pos emb from the read layer and this is not needed.
        
        
        if isinstance(self.window_size, (int, float)):
            self.num_tokens = int(self.window_size) ** 2
        else:
            zt, zh, zw = self.window_size
            self.num_tokens = zt * zh * zw

        self.enable_flashattn = config.get("enable_flashattn", True)
        hidden_size = self.patch_channels
        


        # CA for the read operation
        self.patches_attend_to_latents = []
        self.write_ff = []
        for l in range(self.depth):
            attn_layer =  CrossAttention(
                dim=hidden_size,
                kv_dim=hidden_size,
                num_heads=self.num_heads,
                qkv_bias=self.qkv_bias,
                qk_norm=True,
                enable_flashattn=self.enable_flashattn,
            )
            self.patches_attend_to_latents.append(attn_layer)
            
            ff_layer = Mlp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * self.mlp_ratio),
                out_features=self.patch_channels,
                act_layer=approx_gelu,
                bias=False,
            )
            self.write_ff.append(ff_layer)
        self.write_ff = nn.ModuleList(self.write_ff)
        self.patches_attend_to_latents = nn.ModuleList(self.patches_attend_to_latents)
        
        
        if self.use_learnable_positional_encoding:
            self.pos_embed = nn.Parameter(
                torch.randn((self.num_tokens, hidden_size)) * 0.02,
            )
        
        
    def forward(self, latents, thw_shape, block_kwargs={}, **kwargs):
        # x: (B, N, C) # patches: (B, N, C)
        input_keep_mask = block_kwargs.get("input_keep_mask", None)
        patches = block_kwargs.get("patches", None)
        
        assert patches is not None, "patches must be provided in block_kwargs and passed from the read layer"
        
        B, N, C = patches.shape
        
        
        patches = rearrange_tokens_by_group(patches, thw_shape, self.window_size)  # (B * tg * hg * wg, 1 * z * z, C)
        latents = rearrange_tokens_by_group(latents, thw_shape, self.window_size)  # (B * tg * hg * wg, 1 * z * z, C)
        
        # add positional encoding if needed
        if self.use_learnable_positional_encoding:
            pos_embed = self.pos_embed[None, :patches.shape[1], :]
            patches = patches + pos_embed
            
        # read from latents
        for attn_layer, ff in zip(self.patches_attend_to_latents, self.write_ff):
            patches = attn_layer(
                patches,
                latents,
            ) + patches  
            patches = ff(patches) + patches
        
        latents = rearrange_tokens_by_batch(latents, thw_shape, self.window_size) # (B, N, C)
        patches = rearrange_tokens_by_batch(patches, thw_shape, self.window_size)
        
        # update patches and latents is not needed in case of a single read/write
        # block_kwargs["patches"] = patches
        # block_kwargs['latents'] = latents
        
        assert patches.shape == (B, N, C), "Output shape does not match input shape, expected {}, got {}".format((B, N, C), patches.shape)
        
        return patches, thw_shape, block_kwargs
    
    def reset_parameters(self):
        """
        Reset the parameters of the model.
        """
        if self.use_learnable_positional_encoding:
            nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
            

    

class InputTokensGatherMaskedTokens(nn.Module):
    """
    Gather the masked tokens from the input tokens and discard the rest.
    @moayed:  TODO: current implementation is not effcienct in the gather operation. Improve it later.
    """
    def __init__(self, config={}):
        super().__init__()
        
    def forward(self, x, thw_shape, block_kwargs={}, **kwargs):
        precomputed_freqs_cis = block_kwargs.get('precomputed_freqs_cis')
        input_keep_mask = block_kwargs.get('input_keep_mask')
        attn_mask = block_kwargs.get('attn_mask', None)
        
        # assert precomputed_freqs_cis is not None, "precomputed_freqs_cis must be provided in block_kwargs"
        
        
        if input_keep_mask is None:
            # if no mask is provided, return the input as is
            return x, thw_shape, block_kwargs
        
        B, N, C = x.shape
        
        assert torch.all(input_keep_mask== input_keep_mask[0]), f"Keep mask does not match first batch element mask: {input_keep_mask[:, :, 0].sum(-1)} != {input_keep_mask[0, :, 0].sum()}"
        
        keep_indices = torch.nonzero(input_keep_mask[0, :, 0], as_tuple=False).squeeze(-1)
        
        x_masked = torch.stack([x[b, keep_indices, :] for b in range(B)], dim=0)  # (B, K, C)
        if precomputed_freqs_cis is not None:
            if precomputed_freqs_cis.dim() == 2:
                precomputed_freqs_cis_masked = precomputed_freqs_cis[keep_indices, :]
            else:
                precomputed_freqs_cis_masked = torch.stack([precomputed_freqs_cis[b, keep_indices, :] for b in range(B)], dim=0)

        else:
            precomputed_freqs_cis_masked = None
            
        
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # if attn_mask is 2D, expand it to match the batch size
                attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1)
            
            attn_mask_masked = torch.stack([
                attn_mask[b, keep_indices][:, keep_indices] for b in range(B)
            ])
        else:
            attn_mask_masked = None
            
            
            
        
        x_masked = x_masked.contiguous()
        precomputed_freqs_cis_masked = precomputed_freqs_cis_masked.contiguous() if precomputed_freqs_cis_masked is not None else None
        attn_mask = attn_mask.contiguous() if attn_mask is not None else None
        
        
        block_kwargs['precomputed_freqs_cis'] = precomputed_freqs_cis_masked
        block_kwargs['attn_mask'] = attn_mask_masked
        
        
        # store original tokens to restore later
        block_kwargs['original_tokens'] = x.clone()
        block_kwargs['original_precomputed_freqs_cis'] = precomputed_freqs_cis.clone() if precomputed_freqs_cis is not None else None
        block_kwargs['original_attn_mask'] = attn_mask.clone() if attn_mask is not None else None
        block_kwargs['keep_indices'] = keep_indices
        
        return x_masked, thw_shape, block_kwargs

        
class OutputTokensRestoreMaskedTokens(nn.Module):  
    
    def __init__(self, config={}):
        super().__init__() 
        self.patch_channels = config.get("patch_channels", None)
    
    def forward(self, x, thw_shape, block_kwargs={}, **kwargs):
        
        B, N, C = x.shape
        original_tokens = block_kwargs.get('original_tokens', None)
        original_precomputed_freqs_cis = block_kwargs.get('original_precomputed_freqs_cis', None)
        original_attn_mask = block_kwargs.get('original_attn_mask', None)
        keep_indices = block_kwargs.get('keep_indices', None)

        # only update the tokens and restore everything else (such as RoPE)
        for b in range(B):
            original_tokens[b, keep_indices, :] = x[b]
        
        original_tokens = original_tokens.contiguous()
        block_kwargs['precomputed_freqs_cis'] = original_precomputed_freqs_cis
        block_kwargs['attn_mask'] = original_attn_mask
        
        return original_tokens, thw_shape, block_kwargs
        


import copy 

class RoPEInputTokensEditorWRoPEPerGroup(nn.Module):
    """ Uses 3D rope"""
    def __init__(self, config, **kwargs):
        super().__init__()
        self.window_size = config["window_size"]
        self.use_learnable_positional_encoding = config.get(
            "use_learnable_positional_encoding",
            False,
        )
        
        self.patch_channels = config["patch_channels"]
        self.num_heads = config["num_heads"]
        self.head_size = self.patch_channels // self.num_heads
        
        rope_encodings_precomputer = config.get("rope_encodings_precomputer")

        if rope_encodings_precomputer is not None:
            rope_encodings_precomputer = copy.deepcopy(
                rope_encodings_precomputer,
            )  # Avoids propagation of in place operations
            rope_encodings_precomputer_target = rope_encodings_precomputer["target"]
            del rope_encodings_precomputer["target"]
            self.rope_encodings_precomputer = rope_encodings_precomputer_target(
                rope_encodings_precomputer,
            )
            
            
        if self.use_learnable_positional_encoding:
            self.pos_embed = nn.Parameter(
                torch.randn((self.window_size**2, self.patch_channels)) * 0.02,
            )
        
    def forward(self, x, thw_shape, block_kwargs={}, **kwargs):
        precomputed_freqs_cis = block_kwargs.get('precomputed_freqs_cis', None)
        if precomputed_freqs_cis is None:
            # if no PE is provided, return the input as is
            return x, thw_shape, block_kwargs
        
        
        # grouped_precomputed_freqs_cis = group_tokens_2d(precomputed_freqs_cis, thw_shape, self.window_size)  # 1 x T x H/z x W/z x 1 x z x z x C    
        # w_shape = grouped_precomputed_freqs_cis.shape[4:-1]  # (1, z, z)
        # grouped_precomputed_freqs_cis = grouped_precomputed_freqs_cis[:, :, :, :, 0:1, 0:1, 0:1, :].expand(-1, -1, -1, -1, *w_shape, -1)  # (1, T, H/z, W/z, 1, z, z, C)
        # unified_precomputed_freqs_cis = ungroup_tokens_2d(grouped_precomputed_freqs_cis, thw_shape, self.window_size)  # (1, N, C)
        # unified_precomputed_freqs_cis = unified_precomputed_freqs_cis.squeeze(0)  # (N, C)
        
        
        
        # test another implementation 
        precomputed_freqs_cis = precomputed_freqs_cis.unsqueeze(0)  # (1, N, D)
        grouped_precomputed_freqs_cis = rearrange_tokens_by_group(precomputed_freqs_cis, thw_shape, self.window_size)  # (B * tg * hg * wg, 1 * z * z, C)
        grouped_precomputed_freqs_cis = grouped_precomputed_freqs_cis[:, 0:1, :].expand(-1, self.window_size**2, -1)  # (B * tg * hg * wg, 1 * z * z, C)
        unified_precomputed_freqs_cis = rearrange_tokens_by_batch(grouped_precomputed_freqs_cis, thw_shape, self.window_size)  # (B, N, C)
        unified_precomputed_freqs_cis = unified_precomputed_freqs_cis.squeeze(0)  # (N, C)
        block_kwargs['precomputed_freqs_cis'] = unified_precomputed_freqs_cis
        
        # assert torch.allclose(unified_precomputed_freqs_cis, unified_precomputed_freqs_cis_2), "The two implementations of RoPE are not equivalent"
        
        
        if self.rope_encodings_precomputer is not None:
            # make coords
            t, height, width = thw_shape
            height = height // self.window_size
            width = width // self.window_size
            
            group_size = self.window_size ** 2
            device = x.device
            # g_coords = torch.ones(group_size, device=device).float() # N
            # h_coords = torch.arange(height, device=device).float() # N (all ones)
            # w_coords = torch.arange(width, device=device).float() # N (all ones)

            unified_precomputed_freqs_cis = (
                    self.rope_encodings_precomputer.precompute_freqs_cis_3d(
                        self.head_size,
                        group_size,
                        height,
                        width,
                        device=x.device
                    )
                ) # N x C
            # unified_precomputed_freqs_cis arrange as g x h x w
            unified_precomputed_freqs_cis = unified_precomputed_freqs_cis.reshape(group_size, height*width, -1)
            unified_precomputed_freqs_cis = unified_precomputed_freqs_cis.permute(1, 0, 2)  # (H * W, G, C)
            unified_precomputed_freqs_cis = rearrange_tokens_by_batch(unified_precomputed_freqs_cis, thw_shape, self.window_size) # (1, N, C)
            unified_precomputed_freqs_cis = unified_precomputed_freqs_cis.squeeze(0)  # (N, C)
            block_kwargs['precomputed_freqs_cis'] = unified_precomputed_freqs_cis
            
        
        if self.use_learnable_positional_encoding:
            x = rearrange_tokens_by_group(x, thw_shape, self.window_size)  # (B * tg * hg * wg, 1 * z * z, C)
            x = x + self.pos_embed[None, :x.shape[1], :]  # (B * tg * hg * wg, 1 * z * z, C)
            x = rearrange_tokens_by_batch(x, thw_shape, self.window_size)  # (B, N, C)
        
        return x, thw_shape, block_kwargs

    def reset_parameters(self):
        """
        Reset the parameters of the model.
        """
        if self.use_learnable_positional_encoding:
            nn.init.normal_(self.pos_embed, std=0.02)
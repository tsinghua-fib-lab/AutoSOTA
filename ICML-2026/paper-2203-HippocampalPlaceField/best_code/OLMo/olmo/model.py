from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from typing import List, Union

from .aliases import PathOrStr
from .beam_search import BeamSearch, Constraint, FinalSequenceScorer, Sampler
from .config import (
    ActivationCheckpointingStrategy,
    ActivationType,
    BlockType,
    CheckpointType,
    FSDPWrapStrategy,
    InitFnType,
    LayerNormType,
    ModelConfig,
    ShardedCheckpointerType,
    TrainConfig,
)
from .exceptions import OLMoConfigurationError
from .initialization import init_normal
from .torch_util import ensure_finite_, get_cumulative_document_lengths

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")

__all__ = [
    "LayerNormBase",
    "LayerNorm",
    "RMSLayerNorm",
    "RotaryEmbedding",
    "Activation",
    "GELU",
    "ReLU",
    "SwiGLU",
    "OLMoBlock",
    "OLMoSequentialBlock",
    "OLMo",
    "OLMoOutput",
    "OLMoGenerateOutput",
]


log = logging.getLogger(__name__)


def activation_checkpoint_function(cfg: ModelConfig):
    preserve_rng_state = not (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and (cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


def should_checkpoint_block(strategy: Optional[ActivationCheckpointingStrategy], block_idx: int) -> bool:
    if strategy is None:
        return False
    elif (
        (strategy == ActivationCheckpointingStrategy.whole_layer)
        or (strategy == ActivationCheckpointingStrategy.one_in_two and block_idx % 2 == 0)
        or (strategy == ActivationCheckpointingStrategy.one_in_three and block_idx % 3 == 0)
        or (strategy == ActivationCheckpointingStrategy.one_in_four and block_idx % 4 == 0)
        or (strategy == ActivationCheckpointingStrategy.one_in_eight and block_idx % 8 == 0)
        or (strategy == ActivationCheckpointingStrategy.two_in_three and block_idx % 3 != 0)
        or (strategy == ActivationCheckpointingStrategy.three_in_four and block_idx % 4 != 0)
    ):
        return True
    else:
        return False


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def _non_meta_init_device(config: ModelConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
    ):
        super().__init__()
        self.config = config
        self.eps = config.layer_norm_eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine)
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            )
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x

# Add helper functions at module level or inside RotaryEmbedding as staticmethods
def _yarn_ramp(r: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Compute γ(r) as in YaRN Eq.18."""
    return torch.clamp((r - alpha) / (beta - alpha), 0.0, 1.0)

def _yarn_get_mscale(scale: float) -> float:
    """Compute attention magnitude scaling factor (Eq.22)."""
    if scale <= 1.0:
        return 1.0
    return 0.1 * math.log(scale) + 1.0

class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    Supports YaRN dynamic interpolation for length extrapolation.
    """

    def __init__(self, config: ModelConfig, cache: BufferCache):  
        super().__init__()
        self.config = config
        self._cache = cache
        self.base_inv_freq = self._compute_base_inv_freq(_non_meta_init_device(config))
        self.inv_freq = self.get_inv_freq(_non_meta_init_device(config))
        # Warm up cache.
        self.get_rotary_embedding(config.max_sequence_length, _non_meta_init_device(config))
    
    def _compute_base_inv_freq(self, device: torch.device) -> torch.Tensor:
        dim = self.config.d_model // self.config.n_heads
        i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)

        if self.config.uniform_frequency:
            inv_freq = 2.0 * torch.pi * i / self.config.max_sequence_length
        else:
            inv_freq = 1.0 / (self.config.rope_theta ** (i / dim))

        if self.config.fope and not self.config.use_place_cells:
            inv_freq[inv_freq < 2 * torch.pi / self.config.max_sequence_length] = 0.0
        return inv_freq

    def get_inv_freq(self, device: torch.device):
        inv_freq = self._compute_base_inv_freq(device)

        # --- YaRN integration ---
        if getattr(self.config, "yarn_enabled", False):
            base_len = self.config.yarn_max_position_embeddings
            target_len = self.config.yarn_target_max_position_embeddings or self.config.max_sequence_length
            scale = target_len / base_len

            if scale > 1.0:
                # Compute r(d) = L / λ_d = L * inv_freq / (2π) * 2π → L * inv_freq
                # But paper uses r = L / λ = L * θ / (2π) ≈ L * inv_freq (since θ ∝ 1/λ)
                r = base_len * inv_freq  # shape: (dim//2,)

                alpha = self.config.yarn_beta_slow
                beta = self.config.yarn_beta_fast
                gamma = _yarn_ramp(r, alpha, beta)  # shape: (dim//2,)

                # h(θ) = (1 - γ) * (θ / s) + γ * θ ⇒ inv_freq' = inv_freq / [(1-γ)/s + γ]
                interpolation_factor = (1 - gamma) / scale + gamma
                inv_freq = inv_freq / interpolation_factor

                # Store mscale for later use
                self._mscale = _yarn_get_mscale(scale)
            else:
                self._mscale = 1.0
        else:
            self._mscale = 1.0

        return inv_freq
    
    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # Skip cache if dynamic scaling is enabled
        if not getattr(self.config, "yarn_dynamic_scaling", False):
            if (
                (pos_sin := self._cache.get("rope_pos_sin")) is not None
                and (pos_cos := self._cache.get("rope_pos_cos")) is not None
                and pos_sin.shape[-2] >= seq_len
                and pos_cos.shape[-2] >= seq_len
            ):
                if pos_sin.device != device:
                    pos_sin = pos_sin.to(device)
                    self._cache["rope_pos_sin"] = pos_sin
                if pos_cos.device != device:
                    pos_cos = pos_cos.to(device)
                    self._cache["rope_pos_cos"] = pos_cos
                return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads

            # Reconstruct inv_freq with dynamic scale if needed
            if getattr(self.config, "yarn_dynamic_scaling", False) and getattr(self.config, "yarn_enabled", False):
                base_len = self.config.yarn_max_position_embeddings
                scale = max(1.0, seq_len / base_len)
                # Recompute inv_freq with this scale
                i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                inv_freq = 1.0 / (self.config.rope_theta ** (i / dim))
                if scale > 1.0:
                    r = base_len * inv_freq
                    alpha = self.config.yarn_beta_slow
                    beta = self.config.yarn_beta_fast
                    gamma = _yarn_ramp(r, alpha, beta)
                    interpolation_factor = (1 - gamma) / scale + gamma
                    inv_freq = inv_freq / interpolation_factor
                self._mscale = _yarn_get_mscale(scale)
            else:
                inv_freq = self.inv_freq.to(device)

            seq = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", seq, inv_freq)
            if self.config.fope:
                positions = freqs
            else:
                positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]

        if not getattr(self.config, "yarn_dynamic_scaling", False):
            self._cache["rope_pos_sin"] = pos_sin
            self._cache["rope_pos_cos"] = pos_cos

        return pos_sin, pos_cos
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)


    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        rotated = (t * pos_cos) + (self.rotate_half(t) * pos_sin)
        if hasattr(self, '_mscale') and self._mscale != 1.0:
            rotated = rotated * self._mscale
        return rotated.to(t.dtype)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]  # could be different if layer_past not None
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)




class GridEmbedding(RotaryEmbedding):
    def __init__(self, config: ModelConfig, cache: BufferCache, sigma):
        super().__init__(config, cache)
        # --- Start of modifications ---
        if isinstance(sigma, float):
            # If a single sigma is provided, create a list to use the same value for all heads.
            self.sigma = [sigma] * self.config.n_heads

        with torch.no_grad():
            # This class should only be used with standard RoPE, not FoPE.
            assert not self.config.fope, "ScaledRotaryEmbedding is not compatible with fope=True"

            device = _non_meta_init_device(config)
            inv_freq = self.get_inv_freq(device) # Shape: (dim / 2)

            sigmas_tensor = torch.tensor(self.sigma, device=device, dtype=torch.float).view(self.config.n_heads, 1)

            if hasattr(self.config, 'decay_func'):
                if self.config.decay_func == 'gaussian':
                    scale = torch.exp(-sigmas_tensor**2 * inv_freq.view(1, -1)**2/2)*inv_freq.view(1, -1)
                elif self.config.decay_func == 'exp':
                    #print('using exponential decay function')
                    scale = (1/sigmas_tensor)**2/((1/sigmas_tensor)**2+inv_freq.view(1, -1)**2)*inv_freq.view(1, -1)
                elif self.config.decay_func == 'power':
                    scale = torch.exp(-sigmas_tensor*inv_freq.view(1, -1))*inv_freq.view(1, -1)
            else:
                scale = torch.exp(-sigmas_tensor**2 * inv_freq.view(1, -1)**2/2)*inv_freq.view(1, -1)
            #print(scale)
            scale = torch.sqrt(scale)
            # In standard RoPE, the head dimension is composed of pairs of sin/cos
            # waves of the same frequency. We duplicate the scaling factor to match.
            # Resulting `scale_full` has shape (n_heads, dim)
            scale_full = torch.cat((scale, scale), dim=-1)

            # Normalize the scale factor to preserve variance. This is a global correction.
            correction_factor = torch.rsqrt(torch.mean(scale_full**2))
            scale_full = scale_full * correction_factor

            # Register as a buffer so it moves to the correct device with the model.
            self.register_buffer('scale_factor', scale_full)
            #print(self.scale_factor)
        # --- End of modifications ---

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # Skip cache if dynamic scaling is enabled
        if not getattr(self.config, "yarn_dynamic_scaling", False):
            if (
                (pos_sin := self._cache.get("rope_pos_sin")) is not None
                and (pos_cos := self._cache.get("rope_pos_cos")) is not None
                and pos_sin.shape[-2] >= seq_len
                and pos_cos.shape[-2] >= seq_len
            ):
                if pos_sin.device != device:
                    pos_sin = pos_sin.to(device)
                    self._cache["rope_pos_sin"] = pos_sin
                if pos_cos.device != device:
                    pos_cos = pos_cos.to(device)
                    self._cache["rope_pos_cos"] = pos_cos
                return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads

            # Reconstruct inv_freq with dynamic scale if needed
            if getattr(self.config, "yarn_dynamic_scaling", False) and getattr(self.config, "yarn_enabled", False):
                base_len = self.config.yarn_max_position_embeddings
                scale = max(1.0, seq_len / base_len)
                # Recompute inv_freq with this scale
                i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                inv_freq = 1.0 / (self.config.rope_theta ** (i / dim))
                if scale > 1.0:
                    r = base_len * inv_freq
                    alpha = self.config.yarn_beta_slow
                    beta = self.config.yarn_beta_fast
                    gamma = _yarn_ramp(r, alpha, beta)
                    interpolation_factor = (1 - gamma) / scale + gamma
                    inv_freq = inv_freq / interpolation_factor
                self._mscale = _yarn_get_mscale(scale)
            else:
                inv_freq = self.inv_freq.to(device)

        seq = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", seq, inv_freq)  # (seq_len, dim/2)

        if self.config.fope:
            positions = freqs
        else:
            positions = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)

        # zeros mask
        half_dim = positions.shape[-1] // 2
        zeros_half = torch.zeros(seq_len, half_dim, device=device)

        pos_sin = torch.cat([
            zeros_half,
            freqs.sin()
        ], dim=-1)[None, None, :, :]

        pos_cos = torch.cat([
            freqs.cos(),
            zeros_half 
        ], dim=-1)[None, None, :, :]

        if not getattr(self.config, "yarn_dynamic_scaling", False):
            self._cache["rope_pos_sin"] = pos_sin
            self._cache["rope_pos_cos"] = pos_cos

        return pos_sin, pos_cos    

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Applies scaling to the input tensor `t` (q or k) before applying the rotation.
        """
        # --- Start of modifications ---

        # t has shape (B, nh, T, hs)
        # self.scale_factor has shape (nh, hs)
        # We reshape scale_factor to (1, nh, 1, hs) for broadcasting.
        # t = t.view(B, nh, T, 2, hs//2)
        # t1, t2 = t.unbind(dim=-2)
        # t = torch.cat(())

        t_scaled = t * self.scale_factor.view(1, self.config.n_heads, 1, -1)
        # --- End of modifications ---
        return t_scaled * pos_cos + t_scaled * pos_sin
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]  # could be different if layer_past not None
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)
    


class ScaledRotaryEmbedding(RotaryEmbedding):
    """
    Scaled RoPE with Optional Layer-wise Control.
    """
    def __init__(self, config: ModelConfig, cache: BufferCache, sigma: float = 1.0, layer_index: Optional[int] = None):
        super().__init__(config, cache)
        
        self._sigma_input = sigma
        self._layer_index = layer_index
        self.is_learnable = getattr(config, "learnable_sigma", False)
        self.use_scaling = True
        
        self._update_scale_factor(_non_meta_init_device(config))

    def _update_scale_factor(self, device: torch.device):
        use_scaling = True
        scaling_threshold = getattr(self.config, "rope_scaling_threshold", -1)
        
        if scaling_threshold >= 0 and self._layer_index is not None:
            if self._layer_index <= scaling_threshold:
                use_scaling = False
        self.use_scaling = use_scaling

        dim = self.config.d_model // self.config.n_heads
        
        if not use_scaling:
            scale_full = torch.ones(self.config.n_heads, dim, device=device)
            if hasattr(self, 'scale_factor'):
                self.scale_factor.copy_(scale_full)
            else:
                self.register_buffer('scale_factor', scale_full)
            return 

        if self.is_learnable:
            if not hasattr(self, "sigma_param"):
                initial_sigmas = torch.ones(self.config.n_heads, device=device) * float(self._sigma_input)
                self.sigma_param = nn.Parameter(initial_sigmas)
            return

        if isinstance(self._sigma_input, float):
            sigmas = [self._sigma_input] * self.config.n_heads
        else:
            sigmas = self._sigma_input
            
        if len(sigmas) != self.config.n_heads:
            raise ValueError(f"Sigma count ({len(sigmas)}) must match head count ({self.config.n_heads})")

        with torch.no_grad():
            sigmas_tensor = torch.tensor(sigmas, device=device, dtype=torch.float).view(self.config.n_heads, 1)
            freqs = self.base_inv_freq.to(device).view(1, -1)
            
            decay_func = getattr(self.config, 'decay_func', 'gaussian')
            
            if decay_func == 'gaussian':
                scale = torch.exp(-sigmas_tensor**2 * freqs**2/2) * freqs
            elif decay_func == 'exp':
                scale = (1/sigmas_tensor)**2 / ((1/sigmas_tensor)**2 + freqs**2) * freqs
            elif decay_func == 'power':
                scale = torch.exp(-sigmas_tensor * freqs) * freqs
            elif decay_func == 'segmented':
                order = getattr(self.config, 'decay_order', 8)
                scale = (1.0 / (1.0 + (sigmas_tensor * freqs) ** order)) * freqs
            else:
                scale = torch.exp(-sigmas_tensor**2 * freqs**2/2) * freqs
            
            scale = torch.sqrt(scale)
            scale_full = torch.cat((scale, scale), dim=-1)

            correction_factor = torch.rsqrt(torch.mean(scale_full**2))
            scale_full = scale_full * correction_factor

            if hasattr(self, 'scale_factor'):
                self.scale_factor.copy_(scale_full)
            else:
                self.register_buffer('scale_factor', scale_full)

    def _compute_scale_from_sigmas(self, sigmas: torch.Tensor, device: torch.device) -> torch.Tensor:
        sigmas = sigmas.view(self.config.n_heads, 1)
        freqs = self.base_inv_freq.to(device).view(1, -1)
        decay_func = getattr(self.config, 'decay_func', 'gaussian')

        if decay_func == 'gaussian':
            scale = torch.exp(-sigmas**2 * freqs**2 / 2) * freqs
        elif decay_func == 'exp':
            scale = (1 / sigmas) ** 2 / ((1 / sigmas) ** 2 + freqs**2) * freqs
        elif decay_func == 'power':
            scale = torch.exp(-sigmas * freqs) * freqs
        elif decay_func == 'segmented':
            order = getattr(self.config, 'decay_order', 8)
            scale = (1.0 / (1.0 + (sigmas * freqs) ** order)) * freqs
        else:
            scale = torch.exp(-sigmas**2 * freqs**2 / 2) * freqs

        scale = torch.clamp(scale, min=1e-10)
        scale = torch.sqrt(scale)
        scale_full = torch.cat((scale, scale), dim=-1)
        mean_square = torch.mean(scale_full**2, dim=-1, keepdim=True)
        mean_square = torch.clamp(mean_square, min=1e-10)
        correction_factor = torch.rsqrt(mean_square)
        scale_full = scale_full * correction_factor
        scale_full = torch.nan_to_num(scale_full, nan=1.0, posinf=1.0, neginf=1.0)
        return scale_full

    def _get_scale_factor(self, device: torch.device) -> torch.Tensor:
        dim = self.config.d_model // self.config.n_heads
        if not self.use_scaling:
            return torch.ones(1, self.config.n_heads, 1, dim, device=device)
        if self.is_learnable:
            current_sigmas = torch.clamp(self.sigma_param, min=1e-3).to(device)
            scale_full = self._compute_scale_from_sigmas(current_sigmas, device)
            return scale_full.view(1, self.config.n_heads, 1, -1)
        return self.scale_factor.view(1, self.config.n_heads, 1, dim).to(device)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_scaled = t * self._get_scale_factor(t.device)
        return super().apply_rotary_pos_emb(pos_sin, pos_cos, t_scaled)

    def get_sigma_values(self) -> Optional[torch.Tensor]:
        if self.is_learnable:
            return torch.clamp(self.sigma_param.detach(), min=1e-3)
        return None


class ScaledRotaryEmbedding1(RotaryEmbedding):
    """
    Scaled RoPE with Optional Layer-wise Control.
    Modes:
    1. Uniform Scaling: All layers use the same sigma (if threshold < 0).
    2. Bio-Gradient: Shallow layers use Standard RoPE, Deep layers use Scaled RoPE (if threshold >= 0).
    """

    def __init__(self, config: ModelConfig, cache: BufferCache, sigma: float = 1.0, layer_index: Optional[int] = None):
        """
        Args:
            sigma (float): The scaling factor.
            layer_index (int, optional): The current layer index.
        """
        super().__init__(config, cache)
        
        use_scaling = True
        
        scaling_threshold = getattr(config, "rope_scaling_threshold", -1)
        
        if scaling_threshold >= 0 and layer_index is not None:
            if layer_index <= scaling_threshold:
                use_scaling = False
                print(f"Layer {layer_index}: Scaling DISABLED (Gradient Mode)")
        
        dim = self.config.d_model // self.config.n_heads
        
        if not use_scaling:
            scale_full = torch.ones(self.config.n_heads, dim)
            self.register_buffer('scale_factor', scale_full)
            return
        
        
        if isinstance(sigma, float):
            sigmas = [sigma] * self.config.n_heads
        else:
            sigmas = sigma
            
        if len(sigmas) != self.config.n_heads:
             raise ValueError(f"Sigma count ({len(sigmas)}) must match head count ({self.config.n_heads})")

        with torch.no_grad():
            device = _non_meta_init_device(config)
            inv_freq = self.get_inv_freq(device) 
            
            sigmas_tensor = torch.tensor(sigmas, device=device, dtype=torch.float).view(self.config.n_heads, 1)
            freqs = inv_freq.view(1, -1) 
            
            decay_func = getattr(self.config, 'decay_func', 'gaussian')
            
            if decay_func == 'gaussian':
                scale = torch.exp(-sigmas_tensor**2 * freqs**2/2) * freqs
            elif decay_func == 'exp':
                scale = (1/sigmas_tensor)**2 / ((1/sigmas_tensor)**2 + freqs**2) * freqs
            elif decay_func == 'power':
                scale = torch.exp(-sigmas_tensor * freqs) * freqs
            elif decay_func == 'segmented':
                order = getattr(self.config, 'decay_order', 8)
                scale = (1.0 / (1.0 + (sigmas_tensor * freqs) ** order)) * freqs
            else:
                scale = torch.exp(-sigmas_tensor**2 * freqs**2/2) * freqs
            
            scale = torch.sqrt(scale)
            scale_full = torch.cat((scale, scale), dim=-1)

            correction_factor = torch.rsqrt(torch.mean(scale_full**2))
            scale_full = scale_full * correction_factor

            self.register_buffer('scale_factor', scale_full)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_scaled = t * self.scale_factor.view(1, self.config.n_heads, 1, -1)
        return super().apply_rotary_pos_emb(pos_sin, pos_cos, t_scaled)
    



class ScaledRotaryEmbedding0(RotaryEmbedding):
    """
    A rotary embedding that rescales the query and key tensors based on their
    frequency before applying the rotation. A Gaussian window is applied to the
    frequencies, effectively acting as a low-pass filter, with a potentially
    different sigma (place field size) for each head.
    """
    def __init__(self, config: ModelConfig, cache: BufferCache, sigma: Union[float, List[float]] = 1.0):
        """
        Initializes the ScaledRotaryEmbedding.

        Args:
            config (ModelConfig): The model configuration.
            cache (BufferCache): The buffer cache.
            sigma (Union[float, List[float]], optional): 
                The standard deviation of the Gaussian used for rescaling.
                Can be a single float to apply to all heads, or a list of floats
                with one value per head. Defaults to 1.0.
        """
        self.sigma_vertical = getattr(config, "sigma_vertical", False)
        super().__init__(config, cache)
        

        self.sigma = None
        self.n_omega_intervals = None
        self.dim_intervalsize = None
        dim = self.config.d_model // self.config.n_heads
        self.n_omegas = dim // 2
        self.omega_interval_indices = None

        if self.sigma_vertical:
            if isinstance(sigma, float):
                self.sigma = [sigma]
                self.n_omega_intervals = 1
                self.dim_intervalsize = self.config.n_heads
            else:
                self.n_omega_intervals = len(sigma)
                self.dim_intervalsize = self.n_omegas // self.n_omega_intervals
                self.sigma= sigma

            if self.n_omega_intervals > self.n_omegas:
                raise ValueError(
                    f"The number of intervals ({self.n_omega_intervals}) must match the number of omegas({self.num_omegas})"
                )
            
            self.omega_interval_indices = torch.linspace(
                0, self.n_omega_intervals - 1, self.n_omegas, dtype=torch.long).to(_non_meta_init_device(config))
        else:
            if isinstance(sigma, float):
                self.sigma = [sigma] * self.config.n_heads
            else:
                self.sigma = sigma
            if len(self.sigma) != self.config.n_heads:
                raise ValueError(
                    f"The number of sigmas ({len(self.sigma)}) must match the number of heads ({self.config.n_heads})."
                )
        # if isinstance(sigma, float):
            # If a single sigma is provided, create a list to use the same value for all heads.
        #    self.sigma = [sigma] * self.config.n_heads
        # else:
        #    self.sigma = sigma

        # Validate that the number of sigmas matches the number of heads.
        # if len(self.sigma) != self.config.n_heads:
        #     raise ValueError(
        #        f"The number of sigmas ({len(self.sigma)}) must match the number of heads ({self.config.n_heads})."
        #    )

        with torch.no_grad():
            # This class should only be used with standard RoPE, not FoPE.
            assert not self.config.fope, "ScaledRotaryEmbedding is not compatible with fope=True"

            device = _non_meta_init_device(config)
            inv_freq = self.get_inv_freq(device) # Shape: (dim / 2)

            # Create frequency tensor, handling potential division by zero.
            # freq = torch.zeros_like(inv_freq)
            # non_zero_mask = inv_freq != 0.0
            # freq[non_zero_mask] = torch.reciprocal(inv_freq[non_zero_mask]) # Shape: (dim / 2)

            # Create a tensor of sigmas for broadcasting. Shape: (n_heads, 1)

            if self.sigma_vertical:
                T_sigma = torch.tensor(self.sigma, device=device, dtype=torch.float)
                sigma_per_omega = T_sigma[self.omega_interval_indices]
                sigmas_tensor = sigma_per_omega.unsqueeze(0).repeat(self.config.n_heads, 1)
            else:
                sigmas_tensor = torch.tensor(self.sigma, device=device, dtype=torch.float).view(self.config.n_heads, 1)

            # Calculate scale for each head by broadcasting sigmas against frequencies.
            # sigmas_tensor**2 has shape (n_heads, 1)
            # freq.view(1, -1)**2 has shape (1, dim / 2)
            # Resulting `scale` has shape (n_heads, dim / 2)
            #print(inv_freq)
            if hasattr(self.config, 'decay_func'):
                freqs = inv_freq.view(1, -1) # theta

                if self.config.decay_func == 'gaussian':
                    scale = torch.exp(-sigmas_tensor**2 * freqs**2/2)*freqs
                elif self.config.decay_func == 'exp':
                    #print('using exponential decay function')
                    scale = (1/sigmas_tensor)**2/((1/sigmas_tensor)**2+freqs**2)*freqs
                elif self.config.decay_func == 'power':
                    scale = torch.exp(-sigmas_tensor*freqs)*freqs
                elif self.config.decay_func == 'segmented':
                    order = getattr(self.config, 'decay_order', 8)
                    
                    filter_profile = 1.0 / (1.0 + (sigmas_tensor * freqs) ** order)
                    
                    scale = filter_profile * freqs
            else:
                scale = torch.exp(-sigmas_tensor**2 * freqs**2/2)*freqs
            #print(scale)
            scale = torch.sqrt(scale)
            # In standard RoPE, the head dimension is composed of pairs of sin/cos
            # waves of the same frequency. We duplicate the scaling factor to match.
            # Resulting `scale_full` has shape (n_heads, dim)
            scale_full = torch.cat((scale, scale), dim=-1)

            # Normalize the scale factor to preserve variance. This is a global correction.
            correction_factor = torch.rsqrt(torch.mean(scale_full**2))
            scale_full = scale_full * correction_factor

            # Register as a buffer so it moves to the correct device with the model.
            self.register_buffer('scale_factor', scale_full)
            #print(self.scale_factor)
        
    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Applies scaling to the input tensor `t` (q or k) before applying the rotation.
        """
        t_scaled = t * self.scale_factor.view(1, self.config.n_heads, 1, -1)
        
        return super().apply_rotary_pos_emb(pos_sin, pos_cos, t_scaled)

"""
class ScaledRoPE(RotaryEmbedding):
    def __init__(self, config: ModelConfig, cache: BufferCache): # , lambda_val: float = 10.0, sigma: float = 20.0):
        super().__init__(config,cache)
        self.config = config
        #self.__cache = cache
        self.lambda_val = self.config.sin_lambda  # for w_sin
        self.sigma = self.config.cos_sigma            # for w_cos
        self.d_head = self.config.d_model // self.config.n_heads
        self.d_half = self.d_head // 2
        self.device = _non_meta_init_device(config)

        self._precompute_weights(self.device)

    def _precompute_weights(self, device: torch.device):
        # 1.  ω_d inv_freq
        inv_freq = self.get_inv_freq(device)  # shape (d_half,)
        
        # 2. w_sin = (ω_d * λ)^2 / (1 + (ω_d * λ)^2)
        omega_lambda = inv_freq * self.lambda_val
        w_sin = (omega_lambda ** 2) / (1 + omega_lambda ** 2)  # shape (d_half,)
        
        # 3. w_cos = exp(-(σ * ω_d)^2 / 2)
        sigma_omega = self.sigma * inv_freq
        w_cos = torch.exp(-(sigma_omega ** 2) / 2)  # shape (d_half,)
        
        
        self.register_buffer("w_sin", w_sin)
        self.register_buffer("w_cos", w_cos)

    # calculate the omegas
    def get_inv_freq(self, device):
        dim = self.config.d_model // self.config.n_heads
        inv_freq = 1.0 / (
            self.config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
        )
        # if self.config.fope is True and self.config.use_place_cells is False:
        # Clip frequencies under the floor frequency to zero 
        #    inv_freq[inv_freq < 2 * torch.pi / self.config.max_sequence_length] = 0.0
        return inv_freq

    def split_half(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return x1, x2

    def get_relative_rotary_embedding(self, q_len: int, k_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        q_pos = torch.arange(q_len, device=device, dtype=torch.float)  # (q_len,)
        k_pos = torch.arange(k_len, device=device, dtype=torch.float)  # (k_len,)
        relative_diffs = q_pos.view(-1, 1) - k_pos.view(1, -1)  # (q_len, k_len)
        relative_freqs = einsum("ij, d -> ijd", relative_diffs, self.inv_freq)  # (q_len, k_len, d_half)
        pos_sin = relative_freqs.sin()[None, None, :, :, :]  # (1, 1, q_len, k_len, d_half)
        pos_cos = relative_freqs.cos()[None, None, :, :, :]  # (1, 1, q_len, k_len, d_half)
        return pos_sin, pos_cos

    def apply_scaled_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t1, t2 = self.split_half(t)  # (B, nh, T, d_half)
        
        t1_w = t1.unsqueeze(-2) * self.w_cos.view(1, 1, 1, 1, -1)  # (B, nh, T_q, 1, d_half)
        t2_w = t2.unsqueeze(-2) * self.w_sin.view(1, 1, 1, 1, -1)  # (B, nh, T_q, 1, d_half)

        t_rot1 = (t1_w * pos_cos) + (-t2_w * pos_sin) # (B, nh, T_q, k_len, d_half)
        t_rot2 = (t2_w * pos_cos) + (t1_w * pos_sin)
        t_rot = torch.cat([t_rot1, t_rot2], dim =-1)
        return t_rot.to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_ = q.float()
            k_ = k
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            q_len, k_len = q_.shape[-2], k_.shape[-2]
            
            pos_sin, pos_cos = self.get_relative_rotary_embedding(q_len, k_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            
            # rotate and scale q
            q_rot = self.apply_scaled_rotary_pos_emb(pos_sin, pos_cos, q_)
            k_rot = k_
        
        q_rot = q_rot.view(q.shape[0], q.shape[1], q_len, k_len, self.d_head)
        return q_rot.type_as(q), k_rot.type_as(k)
"""





class ScaledRoPE(nn.Module):
    def __init__(self, config, cache):
        super().__init__()
        self.config = config
        self._cache = cache  # OLMo's BufferCache
        self.d_head = config.d_model // config.n_heads
        self.d_half = self.d_head // 2
        self.lambda_val = getattr(config, 'sin_lambda', 1.0)
        self.sigma = getattr(config, 'cos_sigma', 1.0)
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)
        self.device = _non_meta_init_device(config)
        self._precompute_weights(self.device)

    def _precompute_weights(self, device: torch.device):
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.d_half, device=device, dtype=torch.float) / self.d_half)
        )
        omega_lambda = inv_freq * self.lambda_val
        w_sin = (omega_lambda ** 2) / (1 + omega_lambda ** 2)
        w_cos = torch.exp(-(self.sigma * inv_freq) ** 2 / 2)
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("w_sin", w_sin)
        self.register_buffer("w_cos", w_cos)

    def _get_rotary_embedding(self, seq_len: int, device: torch.device):
        cache_key_sin = "scaled_rope_pos_sin"
        cache_key_cos = "scaled_rope_pos_cos"
        if (
            (pos_sin := self._cache.get(cache_key_sin)) is not None
            and (pos_cos := self._cache.get(cache_key_cos)) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self._cache[cache_key_sin] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self._cache[cache_key_cos] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]


        with torch.autocast(device.type, enabled=False):
            pos = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("i,d->id", pos, self.inv_freq)  # (seq_len, d_half)
            positions = torch.cat((freqs, freqs), dim=-1)  # (seq_len, d_head)
            pos_sin = positions.sin()[None, None, :, :]
            pos_cos = positions.cos()[None, None, :, :]


        self._cache[cache_key_sin] = pos_sin
        self._cache[cache_key_cos] = pos_cos
        return pos_sin, pos_cos

    def _apply_scaling_and_rotation(self, x: torch.Tensor, cos_emb: torch.Tensor, sin_emb: torch.Tensor) -> torch.Tensor:
        B, nh, T, d_head = x.shape
        d_half = d_head // 2

        x1 = x[..., :d_half]   # (B, nh, T, d_half)
        x2 = x[..., d_half:]   # (B, nh, T, d_half)

        # Apply scaling weights
        x1 = x1 * self.w_cos  # (d_half,) broadcasts1613
        x2 = x2 * self.w_sin

        # Expand embeddings to (1, 1, T, d_head)
        cos_emb = cos_emb[:, :, :T, :]
        sin_emb = sin_emb[:, :, :T, :]

        # Split embeddings
        cos1, cos2 = cos_emb[..., :d_half], cos_emb[..., d_half:]
        sin1, sin2 = sin_emb[..., :d_half], sin_emb[..., d_half:]

        # Standard RoPE rotation
        x_rot1 = x1 * cos1 - x2 * sin1
        x_rot2 = x2 * cos2 + x1 * sin2

        return torch.cat([x_rot1, x_rot2], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            T_q = q_.shape[-2]
            T_k = k_.shape[-2]
            max_len = max(T_q, T_k)

            pos_sin_full, pos_cos_full = self._get_rotary_embedding(max_len, q_.device)
            pos_sin_full = pos_sin_full.type_as(q_)
            pos_cos_full = pos_cos_full.type_as(q_)

            # Slice for query (causal case)
            pos_sin_q = pos_sin_full[:, :, max_len - T_q :, :]
            pos_cos_q = pos_cos_full[:, :, max_len - T_q :, :]
            pos_sin_k = pos_sin_full[:, :, :T_k, :]
            pos_cos_k = pos_cos_full[:, :, :T_k, :]

            # Apply scaled RoPE to both Q and K
            q_rot = self._apply_scaling_and_rotation(q_, pos_cos_q, pos_sin_q)
            k_rot = self._apply_scaling_and_rotation(k_, pos_cos_k, pos_sin_k)

        return q_rot.type_as(q), k_rot.type_as(k)

class DiagPositionEmbedding(nn.Module):
    def __init__(self, config, cache):
        super().__init__()
        self.dim = config.d_model // config.n_heads
        self.base = getattr(config, "rope_theta", 10000.0)
        self.max_seq_len = getattr(config, "max_sequence_length", 8192)


        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_len):
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]  # (1,1,T,hd)
            self._sin_cached = emb.sin()[None, None, :, :]
        elif self._cos_cached is None or self._cos_cached.shape[-2] < seq_len:
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

    def apply_rotary_pos_emb(self, x, cos, sin):
        # x: (B, nh, T, hd)
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat(
            [
                x1 * cos - x2 * sin,
                x2 * cos + x1 * sin,
            ],
            dim=-1,
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k: (B, nh, T, hd)
        seq_len = q.shape[-2]
        self._update_cos_sin_tables(q, seq_len)

        cos = self._cos_cached[:, :, :seq_len, :]
        sin = self._sin_cached[:, :, :seq_len, :]

        q_embed = self.apply_rotary_pos_emb(q, cos, sin)
        k_embed = self.apply_rotary_pos_emb(k, cos, sin)
        return q_embed, k_embed


class FourierEmbedding(RotaryEmbedding):
    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__(config, cache)
        #print(self.inv_freq)
        # The paper sets the number of frequency components (D) as a hyperparameter. 
        # Here we follow the pseudocode's logic. 
        self.input_dim = self.inv_freq.size(-1)
        self.head_dim = config.d_model // config.n_heads
        self.output_dim = self.input_dim if self.input_dim <= self.head_dim // 4 else self.head_dim // 4
        device = _non_meta_init_device(config)
        # Initialize coefficient matrices for sine and cosine components 
        self.sin_coef = nn.Parameter(
            torch.randn(self.config.n_heads, self.input_dim, self.output_dim, device=device),
            requires_grad=False
        )
        self.cos_coef = nn.Parameter(
            torch.randn(self.config.n_heads, self.input_dim, self.output_dim, device = device),
            requires_grad=False
        )

        # Initialize with Xavier normal and add identity matrix as in the paper's code 
        torch.nn.init.xavier_normal_(self.sin_coef, gain=self.config.rope_fourier_init_norm_gain)
        torch.nn.init.xavier_normal_(self.cos_coef, gain=self.config.rope_fourier_init_norm_gain)
        self.sin_coef += torch.eye(self.input_dim, self.output_dim, device=self.sin_coef.device)
        self.cos_coef += torch.eye(self.input_dim, self.output_dim, device=self.cos_coef.device)


    # Override the application of the embedding
    def apply_rotary_pos_emb(self, pos_sin, pos_cos, t):
        # This implements the Fourier Series construction from the pseudocode 
        # It maps the coefficients of all frequencies to a Fourier Series for each dimension 
        fourier_sin = torch.einsum("bhtD,hDd->bhtd", pos_sin, self.sin_coef / self.sin_coef.sum(dim=-1, keepdim=True))
        fourier_cos = torch.einsum("bhtD,hDd->bhtd", pos_cos, self.cos_coef / self.cos_coef.sum(dim=-1, keepdim=True))

        # Pad the tensors if necessary 
        fourier_sin = F.pad(input=fourier_sin, pad=(0, self.head_dim // 2 - fourier_sin.size(-1)), mode="constant", value=0)
        fourier_cos = F.pad(input=fourier_cos, pad=(0, self.head_dim // 2 - fourier_cos.size(-1)), mode="constant", value=0)

        # The pseudocode concatenates twice, likely a typo. The logic is to match the head dimension.
        # We will follow the logic of applying the rotation.
        fourier_sin = torch.cat((fourier_sin, fourier_sin), dim=-1)
        fourier_cos = torch.cat((fourier_cos, fourier_cos), dim=-1)

        # Apply the final rotation 
        return ((t * fourier_cos) + (self.rotate_half(t) * fourier_sin)).to(t.dtype)
class PlaceCellEmbedding(FourierEmbedding):
    """
    A positional embedding that rescales the Fourier embedding's sine
    components based on their frequency, inspired by place cells.
    A Gaussian window is applied to the frequency coefficients.

    This is achieved by multiplying the sine coefficients by a scaling factor
    derived from a Gaussian function of the inverse frequencies.
    """
    def __init__(self, config: ModelConfig, cache: BufferCache, sigma: float = 1.0):
        """
        Initializes the PlaceCellEmbedding.

        Args:
            config (ModelConfig): The model configuration.
            cache (BufferCache): The buffer cache.
            sigma (float, optional): The standard deviation of the Gaussian
                                     used for rescaling. Defaults to 1.0.
        """
        # Initialize the parent FourierEmbedding class
        super().__init__(config, cache)
        self.sigma = sigma

        # Rescale the sine coefficients according to the frequency.
        # This is done by applying a Gaussian function to the frequencies.
        # Frequencies that are further away from zero are dampened.
        #print(inv_freq)
        with torch.no_grad():
            inv_freq =  self.get_inv_freq(_non_meta_init_device(config))
            # scale = torch.zeros_like(inv_freq) + 1e-10
            # non_zero_mask = inv_freq != 0.0
            # freq = torch.reciprocal(inv_freq[non_zero_mask])
            #scale[non_zero_mask] = torch.exp(-self.sigma**2 * freq**2)
            scale = torch.exp(-self.sigma**2*inv_freq**2/2)*inv_freq*self.input_dim**2*math.log(10000)
            print(scale)
            # Calculate the frequency scaling factor
            # self.inv_freq is a Tensor of shape (input_dim,)
            correction_factor = torch.rsqrt(torch.mean(scale**2))
            scale = scale*correction_factor

            # self.sin_coef is a Parameter of shape (n_heads, input_dim, output_dim)
            # We broadcast freq_scale to (1, input_dim, 1) to multiply it
            # with the sin_coef parameter in-place.
            self.sin_coef *= scale[None, :, None]
            #print(self.sin_coef.shape)
            self.cos_coef *= scale[None, :, None]


class DiagonalPositionEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.d_model
        base = getattr(config, "rope_theta", 10000.0)
        theta = base ** (-2.0 * torch.arange(0, self.dim).float() / self.dim)
        self.register_buffer("theta", theta, persistent=True)

    def compute_cos_modulation(
        self,
        seq_len_q: int,
        seq_len_k: int,
        device: torch.device,
        dtype: torch.dtype,
        offset_q: int = 0,
        offset_k: int = 0,
    ) -> torch.Tensor:
        """
        Compute cos((m - n) * theta_d) for all m in [offset_q, offset_q + seq_len_q),
                                            n in [offset_k, offset_k + seq_len_k)
        Returns:
            cos_mod: (seq_len_q, seq_len_k, dim)
        """
        # Position indices
        pos_q = torch.arange(offset_q, offset_q + seq_len_q, device=device, dtype=torch.float32)
        pos_k = torch.arange(offset_k, offset_k + seq_len_k, device=device, dtype=torch.float32)
        # (seq_len_q, seq_len_k, 1)
        pos_diff = pos_q[:, None, None] - pos_k[None, :, None]
        # (1, 1, dim)
        theta = self.theta[None, None, :].to(device=device, dtype=torch.float32)
        # (seq_len_q, seq_len_k, dim)
        cos_mod = torch.cos(pos_diff * theta)
        return cos_mod.to(dtype=dtype)


class NoPE(nn.Module):
    """
    No Positional Encoding.
    Just returns q and k as is.
    """
    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return q, k


class XPosEmbedding(RotaryEmbedding):
    """
    XPos implementation inheriting from RotaryEmbedding.
    Includes numerical stability fix for bfloat16 and long sequences.
    """
    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__(config, cache)
        self.symbolic_scale_base = 512
        
        dim = config.d_model // config.n_heads
        
        min_decay = 0.95 
        max_decay = 1.0
        
        indices = torch.arange(0, dim, 2, dtype=torch.float32)
        scale = min_decay + (max_decay - min_decay) * (indices / dim)
        
        self.register_buffer("scale", scale)

    def get_scale(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(seq_len, device=device, dtype=self.scale.dtype)
        power = t[:, None]
        
        scale_val = self.scale.to(device) ** power
        
        scale_val = torch.cat([scale_val, scale_val], dim=-1)
        
        return scale_val[None, None, :, :]

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]
            
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            
            scale = self.get_scale(key_len, q_.device).type_as(q_)

            scale_q = scale[:, :, -query_len:, :]
            sin_q = pos_sin[:, :, -query_len:, :]
            cos_q = pos_cos[:, :, -query_len:, :]
            
            q_emb = (q_ * cos_q) + (self.rotate_half(q_) * sin_q)
            q_emb = q_emb * scale_q
            
            scale_k_slice = scale[:, :, :key_len, :]
            scale_k_safe = torch.clamp(scale_k_slice, min=1e-6)
            scale_k = 1.0 / scale_k_safe
            
            sin_k = pos_sin[:, :, :key_len, :]
            cos_k = pos_cos[:, :, :key_len, :]
            
            k_emb = (k_ * cos_k) + (self.rotate_half(k_) * sin_k)
            k_emb = k_emb * scale_k
            
            return q_emb.type_as(q), k_emb.type_as(k)



class Activation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> Activation:
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


def alibi_attention_bias(seq_len: int, config: ModelConfig, device: torch.device) -> torch.FloatTensor:
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, 1, seq_len)

    # shape: (1, 1, seq_len, seq_len)
    alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, seq_len, 1)
    alibi_bias.abs_().mul_(-1)

    # shape: (n_heads,)
    m = torch.arange(1, config.n_heads + 1, dtype=torch.float, device=device)
    m.mul_(config.alibi_bias_max / config.n_heads)

    # shape: (1, n_heads, seq_len, seq_len)
    return alibi_bias * (1.0 / (2 ** m.view(1, config.n_heads, 1, 1)))  # type: ignore


class OLMoBlock(nn.Module):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0

        self._activation_checkpoint_fn: Optional[Callable] = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout)

        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            assert config.effective_n_kv_heads is not None
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)

        # Make sure QKV clip coefficient is positive, otherwise it's not well-defined.
        if config.clip_qkv is not None:
            assert config.clip_qkv > 0

        # Activation function.
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        self.attn_out = nn.Linear(
            config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
        )

        # Feed-forward output projection.
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore


        # Rotary embeddings, Grid embedding
        if getattr(self.config, 'use_diag_pe', False):
            self.pos_enc = DiagonalPositionEncoding(config)
            log.info("Using Diagonal Modulated Position Encoding (DMPE): score = sum_d q_d k_d cos((m-n)theta_d)")
        elif self.config.grid:
            if hasattr(self.config, 'grid_sigma') and self.config.grid_sigma is not None:
                sigmas = self.config.grid_sigma
                log.info(f"Using Grid with per-head sigmas: {sigmas}")
            else:
                # Fallback: use a scalar sigma (e.g., from config or default)
                sigmas = getattr(self.config, 'grid_sigma', 30.0)
                log.info(f"Using Grid with default value 30.0")
            self.rotary_emb = GridEmbedding(config, self.__cache, sigmas)
        elif self.config.fope:
            if getattr(self.config, 'use_place_cells', False):
                # Make sure you have added the PlaceCellEmbedding class to this file
                # or have imported it.
                self.rotary_emb = PlaceCellEmbedding(
                    config, self.__cache, sigma=self.config.place_cell_sigma
                )
            else:
                self.rotary_emb = FourierEmbedding(config, self.__cache)
        elif self.config.rope:
            use_scaled = getattr(self.config, 'use_scaled_rope1', False)
            
            raw_sigma = getattr(self.config, 'scaled_rope_sigmas', None)
            if raw_sigma is None:
                raw_sigma = self.config.scaled_rope_sigma
            
            current_layer_sigma = raw_sigma
            
            if use_scaled and isinstance(raw_sigma, list) and len(raw_sigma) == config.n_layers:
                current_layer_sigma = raw_sigma[self.layer_id]

            if use_scaled and current_layer_sigma is not None:
                self.rotary_emb = ScaledRotaryEmbedding(
                    config, 
                    self.__cache, 
                    sigma=current_layer_sigma, 
                    layer_index=self.layer_id
                )
            elif getattr(self.config, 'use_scaled_rope2', False):
                self.rotary_emb = ScaledRoPE(config)
            else:
                self.rotary_emb = RotaryEmbedding(config, self.__cache)
        
        elif getattr(self.config, 'nope', False):
            self.rotary_emb = NoPE(config, self.__cache)
            # log.info("Using NoPE (No Positional Encoding)")

        elif getattr(self.config, 'xpos', False):
            self.rotary_emb = XPosEmbedding(config, self.__cache)
            # log.info("Using XPos (Extrapolatable Position Embedding)")


        self.flash_attn_func = None
        self.flash_attn_varlen_func = None
        if config.flash_attention:
            try:
                from flash_attn import (  # type: ignore
                    flash_attn_func,
                    flash_attn_varlen_func,
                )

                self.flash_attn_func = flash_attn_func
                self.flash_attn_varlen_func = flash_attn_varlen_func
                print(">>> SUCCESS: Flash Attention library loaded successfully!")
            except ModuleNotFoundError:
                print(">>> WARNING: config.flash_attention=True but 'flash-attn' library not found! Falling back to slow attention.")
                pass

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()

        if self.config.init_fn == InitFnType.normal:
            attn_out_std = ff_out_std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor

        elif self.config.init_fn == InitFnType.mitchell:
            attn_out_std = 1 / (math.sqrt(2 * self.config.d_model * (self.layer_id + 1)))
            ff_out_std = 1 / (math.sqrt(2 * self.ff_out.in_features * (self.layer_id + 1)))
            cutoff_factor = self.config.init_cutoff_factor or 3.0

        elif self.config.init_fn == InitFnType.full_megatron:
            attn_out_std = ff_out_std = self.config.init_std / math.sqrt(2.0 * self.config.n_layers)
            cutoff_factor = self.config.init_cutoff_factor or 3.0

        else:
            raise NotImplementedError(self.config.init_fn)

        init_normal(self.attn_out, std=attn_out_std, init_cutoff_factor=cutoff_factor)
        init_normal(self.ff_out, std=ff_out_std, init_cutoff_factor=cutoff_factor)

    def set_activation_checkpointing(
        self, strategy: Optional[ActivationCheckpointingStrategy], checkpoint_func: Optional[Callable] = None
    ):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = checkpoint_func or activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
        """
        if max_doc_len is not None and cu_doc_lens is not None:
            assert self.flash_attn_varlen_func is not None, "flash-attn is required for document masking"
            assert attn_mask is None, "attn-mask is currently not supported with document masking"
            B, T, D = q.size(0), q.size(2), q.size(3)
            r = self.flash_attn_varlen_func(
                q.transpose(1, 2).view(B * T, -1, D),
                k.transpose(1, 2).view(B * T, -1, D),
                v.transpose(1, 2).view(B * T, -1, D),
                cu_doc_lens,
                cu_doc_lens,
                max_doc_len,
                max_doc_len,
                dropout_p=dropout_p,
                causal=is_causal,
            )
            return r.view(B, T, -1, D).transpose(1, 2)
        elif self.flash_attn_func is not None and attn_mask is None:
            r = self.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=is_causal
            )
            return r.transpose(1, 2)
        else:
            # torch's sdpa doesn't support GQA, so we're doing this
            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T_q, C = q.size()  # T_q = current query length
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Reshape for multi-head: (B, nh, T, hs)
        q = q.view(B, T_q, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        k = k.view(B, T_q, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        v = v.view(B, T_q, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        # Handle past key/values for caching
        offset_k = 0
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
            offset_k = past_key.size(-2)  # number of cached tokens

        present = (k, v) if use_cache else None
        T_k = k.size(-2)  # total key length (including cache)

        if getattr(self.config, 'use_diag_pe', False):
            assert max_doc_len is None and cu_doc_lens is None, "Document masking not supported in DMPE yet"
            
            # Expand k/v heads if needed (GQA)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

            # Compute cos((m - n) * theta_d) for all (m,n,d)
            cos_mod = self.pos_enc.compute_cos_modulation(
                seq_len_q=T_q,
                seq_len_k=T_k,
                device=q.device,
                dtype=q.dtype,
                offset_q=offset_k,
                offset_k=0,
            )  # (T_q, T_k, D_full)

            head_dim = q.size(-1)
            cos_mod = cos_mod.unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, T_k, D_full)
            cos_mod = cos_mod.view(1, 1, T_q, T_k, self.config.n_heads, head_dim)  # (1,1,Tq,Tk,nh,hd)
            cos_mod = cos_mod.permute(0, 4, 2, 3, 1, 5)  # (1, nh, Tq, Tk, 1, hd)
            cos_mod = cos_mod.squeeze(-2)  # (1, nh, Tq, Tk, hd)

            q_exp = q.unsqueeze(3)  # (B, nh, T_q, 1, hd)
            k_exp = k.unsqueeze(2)  # (B, nh, 1, T_k, hd)
            qk_prod = q_exp * k_exp  # (B, nh, T_q, T_k, hd)

            scores = torch.sum(qk_prod * cos_mod, dim=-1)  # (B, nh, T_q, T_k)
            scores = scores / math.sqrt(head_dim)

            if attention_bias is not None:
                bias_slice = attention_bias[:, :, offset_k:offset_k + T_q, :T_k]
                scores = scores + self._cast_attn_bias(bias_slice, dtype)

            if attention_bias is None:
                causal_mask = torch.triu(
                    torch.full_like(scores, float("-inf")), diagonal=1 + (T_k - T_q)
                )
                scores = scores + causal_mask

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(
                attn_weights,
                p=self.config.attention_dropout if self.training else 0.0,
                training=self.training,
            )

            att = torch.matmul(attn_weights, v)

        else:
            # ----------------------------
            # Original RoPE / other paths
            # ----------------------------
            if hasattr(self, 'rotary_emb') and self.rotary_emb is not None:
                q, k = self.rotary_emb(q, k)

            local_window = getattr(self.config, "local_window_size", -1)
            num_local_layers = getattr(self.config, "num_local_layers", 0)
            
            is_local_layer = (local_window > 0) and (self.layer_id < num_local_layers)

            if is_local_layer:
                q_idx = torch.arange(T_q, device=q.device).view(-1, 1)
                k_idx = torch.arange(T_k, device=k.device).view(1, -1)
                
                dist_mask = (q_idx + offset_k - k_idx) < local_window
                
                causal_mask = k_idx <= (q_idx + offset_k)
                
                valid_mask = dist_mask & causal_mask

                local_bias = torch.zeros(1, 1, T_q, T_k, device=q.device, dtype=dtype)
                local_bias.masked_fill_(~valid_mask, torch.finfo(dtype).min)

                if attention_bias is not None:
                    bias_slice = attention_bias[:, :, offset_k:offset_k + T_q, :T_k]
                    bias_slice = self._cast_attn_bias(bias_slice, dtype)
                    attention_bias = bias_slice + local_bias
                else:
                    attention_bias = local_bias
            else:
                if attention_bias is not None:
                    bias_slice = attention_bias[:, :, offset_k:offset_k + T_q, :T_k]
                    attention_bias = self._cast_attn_bias(bias_slice, dtype)

            att = self._scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_bias,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                is_causal=(attention_bias is None),
                max_doc_len=max_doc_len,
                cu_doc_lens=cu_doc_lens,
            )

        # Reassemble heads
        att = att.transpose(1, 2).contiguous().view(B, T_q, C)
        return self.attn_out(att), present
    
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: ModelConfig, cache: BufferCache) -> OLMoBlock:
        if config.block_type == BlockType.sequential:
            return OLMoSequentialBlock(layer_id, config, cache)
        elif config.block_type == BlockType.llama:
            return OLMoLlamaBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class OLMoSequentialBlock(OLMoBlock):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). To compute it as ``LN(MLP(x + LN(Attention(x))))``,
    use the flag `norm_after`.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Attention input projection. Projects x -> (q, k, v)

        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim,
        )
        self.att_proj = nn.Linear(
            config.d_model, sum(self.fused_dims), bias=config.include_bias, device=config.init_device
        )
        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

        # Layer norms.
        self.attn_norm = LayerNorm.build(config, size=config.d_model)
        self.ff_norm = LayerNorm.build(config, size=config.d_model)

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.

        if self.config.init_fn == InitFnType.normal:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor
        elif self.config.init_fn == InitFnType.mitchell:
            std = 1 / math.sqrt(self.config.d_model)
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == InitFnType.full_megatron:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            raise NotImplementedError(self.config.init_fn)

        init_normal(self.att_proj, std, cutoff_factor)
        init_normal(self.ff_proj, std, cutoff_factor)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)

        # apply norm before
        if not self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                h = self._activation_checkpoint_fn(self.attn_norm, x)
            else:
                h = self.attn_norm(x)
        else:
            h = x

        qkv = self.att_proj(h)

        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        q, k, v = qkv.split(self.fused_dims, dim=-1)

        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention,
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                max_doc_len=max_doc_len,
                cu_doc_lens=cu_doc_lens,
            )
        else:
            att, cache = self.attention(
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                max_doc_len=max_doc_len,
                cu_doc_lens=cu_doc_lens,
            )

        if self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                att = self._activation_checkpoint_fn(self.attn_norm, att)
            else:
                att = self.attn_norm(att)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x

        if not self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
            else:
                x = self.ff_norm(x)

        x = self.ff_proj(x)

        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = self.ff_out(x)

        if self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
            else:
                x = self.ff_norm(x)

        x = self.dropout(x)
        x = og_x + x

        return x, cache


class OLMoLlamaBlock(OLMoBlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `OLMoSequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        if config.multi_query_attention:
            q_proj_out_dim = config.d_model
            k_proj_out_dim = config.d_model // config.n_heads
            v_proj_out_dim = config.d_model // config.n_heads
        else:
            q_proj_out_dim = config.d_model
            k_proj_out_dim = config.d_model
            v_proj_out_dim = config.d_model
        self.q_proj = nn.Linear(
            config.d_model, q_proj_out_dim, bias=config.include_bias, device=config.init_device
        )
        self.k_proj = nn.Linear(
            config.d_model, k_proj_out_dim, bias=config.include_bias, device=config.init_device
        )
        self.v_proj = nn.Linear(
            config.d_model, v_proj_out_dim, bias=config.include_bias, device=config.init_device
        )

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.

        if self.config.init_fn == InitFnType.normal:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor
        elif self.config.init_fn == InitFnType.mitchell:
            std = 1 / math.sqrt(self.config.d_model)
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == InitFnType.full_megatron:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            raise NotImplementedError(self.config.init_fn)

        init_normal(self.q_proj, std, cutoff_factor)
        init_normal(self.k_proj, std, cutoff_factor)
        init_normal(self.v_proj, std, cutoff_factor)
        init_normal(self.ff_proj, std, cutoff_factor)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if max_doc_len is not None or cu_doc_lens is not None:
            raise NotImplementedError(
                f"attention document masking is not implemented for {self.__class__.__name__}"
            )

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if is_causal:
            assert attn_mask is None

            query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None
            attn_bias = get_causal_attention_bias(self.__cache, key_len, q.device)[:, :, :query_len, :key_len]
        elif attn_mask is not None:
            attn_bias = attn_mask.to(q.dtype)
        else:
            attn_bias = torch.zeros_like(attn_weights)

        attn_weights += attn_bias
        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(q.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout_p)
        return torch.matmul(attn_weights, v)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        if self.config.clip_qkv is not None:
            q.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            k.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            v.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        # Get attention scores.
        att, cache = self.attention(
            q,
            k,
            v,
            attention_bias,
            layer_past=layer_past,
            use_cache=use_cache,
            max_doc_len=max_doc_len,
            cu_doc_lens=cu_doc_lens,
        )

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x = self.ff_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache


class OLMoOutput(NamedTuple):
    logits: torch.FloatTensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """

    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    """
    Attention keys and values from each block.
    """

    hidden_states: Optional[Tuple[torch.Tensor, ...]]
    """
    Hidden states from each block.
    """


class OLMoGenerateOutput(NamedTuple):
    token_ids: torch.LongTensor
    """
    The generated token IDs, a tensor of shape `(batch_size, beam_size, max_steps)`.
    These do *not* include the original input IDs.
    """

    scores: torch.FloatTensor
    """
    The scores of the generated sequences, a tensor of shape `(batch_size, beam_size)`.
    """


class OLMoBlockGroup(nn.ModuleList):
    def __init__(self, config: ModelConfig, layer_offset: int, modules: Optional[Iterable[nn.Module]] = None):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset
        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layers_past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        for block_idx, block in enumerate(self):
            layer_past = None if layers_past is None else layers_past[block_idx]
            block_idx += self.layer_offset
            if should_checkpoint_block(self.activation_checkpointing_strategy, block_idx):
                # shape: (batch_size, seq_len, d_model)
                x, cache = self._activation_checkpoint_fn(  # type: ignore
                    block,
                    x,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                    max_doc_len=max_doc_len,
                    cu_doc_lens=cu_doc_lens,
                )
            else:
                # shape: (batch_size, seq_len, d_model)
                x, cache = block(
                    x,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                    max_doc_len=max_doc_len,
                    cu_doc_lens=cu_doc_lens,
                )
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
        return x, attn_key_values

    def reset_parameters(self):
        for block in self:
            block.reset_parameters()

    def set_activation_checkpointing(
        self, strategy: Optional[ActivationCheckpointingStrategy], checkpoint_func: Optional[Callable] = None
    ):
        self.activation_checkpointing_strategy = strategy
        for block in self:
            block.set_activation_checkpointing(strategy, checkpoint_func=checkpoint_func)


class OLMo(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

        # Validate config.
        if self.config.alibi and self.config.flash_attention:
            raise OLMoConfigurationError("ALiBi is currently not supported with FlashAttention")

        if self.config.alibi and self.config.rope:
            raise OLMoConfigurationError("ALiBi and RoPE are mutually exclusive")

        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise OLMoConfigurationError("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise OLMoConfigurationError("n layers must be divisible by block group size")
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )

        blocks = [OLMoBlock.build(i, config, self.__cache) for i in range(config.n_layers)]
        if self.config.block_group_size > 1:
            block_groups = [
                OLMoBlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )
        if config.embedding_layer_norm:
            self.transformer.update({"emb_norm": LayerNorm.build(config)})

        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None
        self.__num_bck_flops: Optional[int] = None

        # Warm up cache.
        if self.config.alibi:
            get_causal_attention_bias(self.__cache, config.max_sequence_length, _non_meta_init_device(config))
            self.get_alibi_attention_bias(config.max_sequence_length, _non_meta_init_device(config))

    def set_activation_checkpointing(
        self, strategy: Optional[ActivationCheckpointingStrategy], checkpoint_func: Optional[Callable] = None
    ):
        self.activation_checkpointing_strategy = strategy
        if self.config.block_group_size != 1:
            for block_group in self.transformer.block_groups:
                block_group.set_activation_checkpointing(strategy, checkpoint_func=checkpoint_func)
        else:
            for block in self.transformer.blocks:
                block.set_activation_checkpointing(strategy, checkpoint_func=checkpoint_func)

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def reset_parameters(self):
        log.info("Initializing model parameters...")
        # Top-level embeddings / linear layers.

        if self.config.init_fn == InitFnType.normal:
            # Note: We may potentially want to multiply the std by a factor of sqrt(d) in case of `scale_logits`
            # and `weight_tying`. However, we are currently not using either, and may need to rethink the init logic
            # if/when we do want it.
            wte_std = self.config.emb_init_std or self.config.init_std
            wte_cutoff_factor = self.config.init_cutoff_factor
        elif self.config.init_fn == InitFnType.mitchell:
            wte_std = self.config.emb_init_std or 1.0 / math.sqrt(self.config.d_model)
            wte_cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == InitFnType.full_megatron:
            wte_std = self.config.init_std
            if self.config.emb_init_std is not None:
                wte_std = self.config.emb_init_std
            elif self.config.scale_emb_init:
                wte_std *= math.sqrt(self.config.d_model)
            wte_cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            raise NotImplementedError(self.config.init_fn)

        init_normal(self.transformer.wte, std=wte_std, init_cutoff_factor=wte_cutoff_factor)

        if hasattr(self.transformer, "wpe"):
            if self.config.init_fn == InitFnType.normal:
                wpe_std = self.config.init_std
                wpe_cutoff_factor = self.config.init_cutoff_factor
            elif self.config.init_fn == InitFnType.mitchell:
                wpe_std = 1 / math.sqrt(self.config.d_model)
                wpe_cutoff_factor = self.config.init_cutoff_factor or 3.0
            elif self.config.init_fn == InitFnType.full_megatron:
                wpe_std = self.config.init_std
                wpe_cutoff_factor = self.config.init_cutoff_factor or 3.0
            else:
                raise NotImplementedError(self.config.init_fn)

            init_normal(self.transformer.wpe, std=wpe_std, init_cutoff_factor=wpe_cutoff_factor)

        # Top-level layer norm.
        self.transformer.ln_f.reset_parameters()  # type: ignore

        # Output weights.
        if hasattr(self.transformer, "ff_out"):
            if self.config.init_fn == InitFnType.normal:
                ff_out_std = self.config.init_std
                ff_out_cutoff_factor = self.config.init_cutoff_factor
            elif self.config.init_fn == InitFnType.mitchell:
                ff_out_std = 1 / math.sqrt(self.config.d_model)
                ff_out_cutoff_factor = self.config.init_cutoff_factor or 3.0
            elif self.config.init_fn == InitFnType.full_megatron:
                ff_out_std = 1 / math.sqrt(self.config.d_model)
                ff_out_cutoff_factor = self.config.init_cutoff_factor or 3.0
            else:
                raise NotImplementedError(self.config.init_fn)

            init_normal(self.transformer.ff_out, ff_out_std, ff_out_cutoff_factor)

        # Let the blocks handle themselves.
        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()

    def get_alibi_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (alibi_bias := self.__cache.get("alibi_attention_bias")) is not None and alibi_bias.shape[
            -1
        ] >= seq_len:
            if alibi_bias.device != device:
                alibi_bias = alibi_bias.to(device)
                self.__cache["alibi_attention_bias"] = alibi_bias
            return alibi_bias
        with torch.autocast(device.type, enabled=False):
            alibi_bias = alibi_attention_bias(seq_len, self.config, device)
        self.__cache["alibi_attention_bias"] = alibi_bias
        return alibi_bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        doc_lens: Optional[torch.Tensor] = None,
        max_doc_lens: Optional[Sequence[int]] = None,
    ) -> OLMoOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        :param doc_lens: Document lengths to use in attention for intra-document masking.
            Shape `(batch_size, max_docs)`.
        :param max_doc_lens: Maximum document length for each instance in the batch.
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        max_doc_len: Optional[int] = None
        cu_doc_lens: Optional[torch.Tensor] = None
        if doc_lens is not None and max_doc_lens is not None:
            max_doc_len = max(max_doc_lens)
            cu_doc_lens = get_cumulative_document_lengths(doc_lens)

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore

        # Apply embedding layer norm.
        if self.config.embedding_layer_norm:
            x = self.transformer.emb_norm(x)

        if not (self.config.alibi or self.config.rope):
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        # Apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None:
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

        # Merge attention mask with attention bias.
        if (
            attention_bias is not None
            or attention_mask is not None
            or self.config.alibi
            # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
            # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
            # scores correctly.
            or past_key_values is not None
        ):
            if attention_bias is None and self.config.alibi:
                attention_bias = get_causal_attention_bias(
                    self.__cache, past_length + seq_len, x.device
                ) + self.get_alibi_attention_bias(past_length + seq_len, x.device)
            elif attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
                # it can produce NaNs.
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        # decoder layers
        all_hidden_states = []

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                if should_checkpoint_block(self.activation_checkpointing_strategy, block_idx):
                    # shape: (batch_size, seq_len, d_model)
                    x, cache = self._activation_checkpoint_fn(
                        block,
                        x,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                        max_doc_len=max_doc_len,
                        cu_doc_lens=cu_doc_lens,
                    )
                else:
                    # shape: (batch_size, seq_len, d_model)
                    x, cache = block(
                        x,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                        max_doc_len=max_doc_len,
                        cu_doc_lens=cu_doc_lens,
                    )

                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                    ]
                )
                x, cache = block_group(
                    x,
                    attention_bias=attention_bias,
                    layers_past=layers_past,
                    use_cache=use_cache,
                    max_doc_len=max_doc_len,
                    cu_doc_lens=cu_doc_lens,
                )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        else:
            logits = self.transformer.ff_out(x)  # type: ignore
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        return OLMoOutput(
            logits=logits,
            attn_key_values=attn_key_values,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )

    def get_fsdp_wrap_policy(self, wrap_strategy: Optional[FSDPWrapStrategy] = None):
        if wrap_strategy is None:
            return None

        # The 'recurse' mode for the wrap function does not behave like you'd expect.
        # Even if we return False, it may still recurse because PyTorch does what it wants,
        # not what you want. This causes issues when, for example, we want to wrap 'ff_out' (a linear layer)
        # but not other linear layers within a block.
        # So we have to explicitly tell PyTorch which linear layers to wrap, and we also just
        # return True in 'recurse' mode for simplicity.
        size_based_module_to_wrap = {self.transformer.wte}
        if hasattr(self.transformer, "ff_out"):
            size_based_module_to_wrap.add(self.transformer.ff_out)

        if wrap_strategy == FSDPWrapStrategy.by_block:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlock)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_and_size:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, (OLMoBlock,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group:
            if self.config.block_group_size <= 1:
                raise OLMoConfigurationError(
                    "'by_block_group' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlockGroup)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group_and_size:
            if self.config.block_group_size <= 1:
                raise OLMoConfigurationError(
                    "'by_block_group_and_size' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, (OLMoBlockGroup,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.size_based:
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            return size_based_auto_wrap_policy
        elif wrap_strategy in {
            FSDPWrapStrategy.one_in_two,
            FSDPWrapStrategy.one_in_three,
            FSDPWrapStrategy.one_in_four,
            FSDPWrapStrategy.one_in_five,
        }:
            c = {
                FSDPWrapStrategy.one_in_two: 2,
                FSDPWrapStrategy.one_in_three: 3,
                FSDPWrapStrategy.one_in_four: 4,
                FSDPWrapStrategy.one_in_five: 5,
            }[wrap_strategy]

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlock) and module.layer_id % c == 0
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        else:
            raise NotImplementedError(wrap_strategy)

    def num_params(self, include_embedding: bool = True) -> int:
        """
        Get the total number of parameters.
        """
        params = (np for np in self.named_parameters())
        if not include_embedding:
            params = filter(  # type: ignore
                lambda np: ".wte." not in np[0] and ".wpe." not in np[0],
                params,
            )
        return sum(p.numel() for _, p in params)

    @property
    def num_fwd_flops(self):
        if self.__num_fwd_flops:
            return self.__num_fwd_flops

        # embedding table is just a lookup in the forward pass
        n_params = self.num_params(include_embedding=False)
        # the number of parameters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # this gets us FLOPs / token
        params_flops_per_token = 2 * n_params
        # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
        attn_flops_per_token = (
            self.config.n_layers * 2 * 2 * (self.config.d_model * self.config.max_sequence_length)
        )
        self.__num_fwd_flops = params_flops_per_token + attn_flops_per_token
        return self.__num_fwd_flops

    @property
    def num_bck_flops(self):
        if self.__num_bck_flops:
            return self.__num_bck_flops

        n_params = self.num_params()
        params_flops_per_token = 4 * n_params
        attn_flops_per_token = self.config.n_layers * 8 * (self.config.d_model * self.config.max_sequence_length)
        self.__num_bck_flops = params_flops_per_token + attn_flops_per_token
        return self.__num_bck_flops

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        max_steps: int = 10,
        beam_size: int = 1,
        per_node_beam_size: Optional[int] = None,
        sampler: Optional[Sampler] = None,
        min_steps: Optional[int] = None,
        final_sequence_scorer: Optional[FinalSequenceScorer] = None,
        constraints: Optional[List[Constraint]] = None,
    ) -> OLMoGenerateOutput:
        """
        Generate token IDs using beam search.

        Note that by default ``beam_size`` is set to 1, which is greedy decoding.

        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param attention_mask: A optional tensor of shape `(batch_size, seq_len)`, the same
            as for the forward method.
        :param attention_bias: A tensor of shape
            `(batch_size, 1, seq_len + tokens_to_generate, seq_len + tokens_to_generate)`,
            the same as for the forward method except only one shape is excepted here.

        For an explanation of the other arguments, see :class:`BeamSearch`.
        """
        beam_search = BeamSearch(
            self.config.eos_token_id,
            max_steps=max_steps,
            beam_size=beam_size,
            per_node_beam_size=per_node_beam_size,
            sampler=sampler,
            min_steps=min_steps,
            final_sequence_scorer=final_sequence_scorer,
            constraints=constraints,
        )

        # Validate inputs.
        batch_size, seq_len = input_ids.shape
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, seq_len)
        if attention_bias is not None:
            assert len(attention_bias.shape) == 4
            assert attention_bias.shape[:2] == (batch_size, 1)
            assert (
                seq_len + beam_search.max_steps
                <= attention_bias.shape[2]
                == attention_bias.shape[3]
                <= self.config.max_sequence_length
            )

        tokens_generated = 0

        def flatten_past_key_values(
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        ) -> Dict[str, torch.Tensor]:
            out = {}
            for i, (key, value) in enumerate(past_key_values):
                out[f"past_key_{i}"] = key
                out[f"past_value_{i}"] = value
            return out

        def unflatten_past_key_values(
            past_key_values: Dict[str, torch.Tensor],
        ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
            out = []
            for i in range(self.config.n_layers):
                past_key = past_key_values[f"past_key_{i}"]
                past_value = past_key_values[f"past_value_{i}"]
                out.append((past_key, past_value))
            return out

        def step(
            last_predictions: torch.Tensor, state: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            nonlocal tokens_generated

            attention_mask = state.get("attention_mask")
            attention_bias = state.get("attention_bias")

            if tokens_generated > 0:
                past_key_values = unflatten_past_key_values(state)
                input_ids = last_predictions.unsqueeze(1)
                if attention_mask is not None:
                    group_size = input_ids.shape[0]
                    attention_mask = torch.cat((attention_mask, attention_mask.new_ones((group_size, 1))), dim=-1)
            else:
                past_key_values = None
                input_ids = state["input_ids"]

            tokens_generated += 1

            # Run forward pass of model to get logits, then normalize to get log probs.
            output = self(
                input_ids,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                past_key_values=past_key_values,
                use_cache=True,
                last_logits_only=True,
            )
            log_probs = F.log_softmax(output.logits[:, -1, :], dim=-1)

            # Create new state.
            state = flatten_past_key_values(output.attn_key_values)
            if attention_mask is not None:
                state["attention_mask"] = attention_mask
            if attention_bias is not None:
                state["attention_bias"] = attention_bias

            return log_probs, state

        initial_preds = input_ids.new_zeros((batch_size,))  # This is arbitrary, we won't use this.
        state: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            state["attention_mask"] = attention_mask
        if attention_bias is not None:
            state["attention_bias"] = attention_bias
        with torch.no_grad():
            token_ids, scores = beam_search.search(initial_preds, state, step)

        return OLMoGenerateOutput(
            token_ids=token_ids,  # type: ignore[arg-type]
            scores=scores,  # type: ignore[arg-type]
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: PathOrStr, device: str = "cpu", checkpoint_type: Optional[CheckpointType] = None
    ) -> OLMo:
        """
        Load an OLMo model from a checkpoint.
        """
        from .util import resource_path

        # Guess checkpoint type.
        if checkpoint_type is None:
            try:
                if resource_path(checkpoint_dir, "model.pt").is_file():
                    checkpoint_type = CheckpointType.unsharded
                else:
                    checkpoint_type = CheckpointType.sharded
            except FileNotFoundError:
                checkpoint_type = CheckpointType.sharded

        # Load config.
        config_path = resource_path(checkpoint_dir, "config.yaml")
        model_config = ModelConfig.load(config_path, key="model", validate_paths=False)

        if checkpoint_type == CheckpointType.unsharded:
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            model = OLMo(model_config)

            # Load state dict directly to target device.
            state_dict_path = resource_path(checkpoint_dir, "model.pt")
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(model._make_state_dict_compatible(state_dict)[0])
            model = model.to(torch.device(device))
        else:
            train_config = TrainConfig.load(config_path)
            if train_config.sharded_checkpointer == ShardedCheckpointerType.olmo_core:
                from olmo_core.distributed.checkpoint import (  # type: ignore
                    load_model_and_optim_state,
                )

                model_config.init_device = device
                model = OLMo(model_config)
                load_model_and_optim_state(checkpoint_dir, model)
            else:
                # train_config.sharded_checkpointer == ShardedCheckpointerType.torch_new
                from .checkpoint import load_model_state

                # Initialize model on target device. In this case the state dict is loaded in-place
                # so it's not necessary to start on CPU if the target device is a GPU.
                model_config.init_device = device
                model = OLMo(model_config)

                # Load state dict in place.
                load_model_state(checkpoint_dir, model)

        return model.eval()

    def _make_state_dict_compatible(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Set[str]]]:
        """
        Handles some cases where the state dict is valid yet may need to be transformed in order to
        be loaded.

        This modifies the state dict in-place and also returns it, along with a mapping of original key
        names to new key names in cases where the keys were simply renamed. That mapping can be used
        to make a corresponding optimizer state dict compatible as well.
        """
        import re
        from fnmatch import fnmatch

        new_keys_to_og_keys: Dict[str, str] = {}

        # Remove "_fsdp_wrapped_module." prefix from all keys. We don't want this prefix when the model is
        # not wrapped in FSDP. And when the model is wrapped in FSDP, loading this state dict will still work
        # fine without the prefixes. This also simplifies the other steps below.
        for key in list(state_dict.keys()):
            state_dict[(new_key := key.replace("_fsdp_wrapped_module.", ""))] = state_dict.pop(key)
            new_keys_to_og_keys[new_key] = key

        # For backwards compatibility prior to fixing https://github.com/allenai/LLM/issues/222
        if self.config.block_type == BlockType.sequential:
            for key in list(state_dict.keys()):
                if fnmatch(key, "transformer.*.norm.weight"):
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("norm.weight", "attn_norm.weight"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    state_dict[(new_key := key.replace("norm.weight", "ff_norm.weight"))] = tensor.clone()
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]
                elif fnmatch(key, "transformer.*.norm.bias"):
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("norm.bias", "attn_norm.bias"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    state_dict[(new_key := key.replace("norm.bias", "ff_norm.bias"))] = tensor.clone()
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]

        # For loading a state dict that was saved with a different `block_group_size`.
        if "transformer.block_groups.0.0.attn_out.weight" in state_dict.keys():
            state_dict_block_group_size = len(
                [k for k in state_dict.keys() if fnmatch(k, "transformer.block_groups.0.*.attn_out.weight")]
            )
        else:
            state_dict_block_group_size = 1
        if self.config.block_group_size != state_dict_block_group_size:
            log.info(
                f"Regrouping state dict blocks from group size {state_dict_block_group_size} to "
                f"group size {self.config.block_group_size}"
            )
            # For simplicity we're first going to flatten out the block groups in the state dict (if necessary)
            # and then (re-)group them into the right block sizes.
            if state_dict_block_group_size > 1:
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.block_groups\.(\d+)\.(\d+)\..*", key)) is not None:
                        group_idx, group_block_idx = int(m.group(1)), int(m.group(2))
                        block_idx = (group_idx * state_dict_block_group_size) + group_block_idx
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"block_groups.{group_idx}.{group_block_idx}.", f"blocks.{block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

            if self.config.block_group_size > 1:
                # Group the state dict blocks into the right block size.
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.blocks\.(\d+)\..*", key)) is not None:
                        block_idx = int(m.group(1))
                        group_idx, group_block_idx = (
                            block_idx // self.config.block_group_size,
                            block_idx % self.config.block_group_size,
                        )
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"blocks.{block_idx}.", f"block_groups.{group_idx}.{group_block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

        og_keys_to_new: Dict[str, Set[str]] = defaultdict(set)
        for new_key, og_key in new_keys_to_og_keys.items():
            og_keys_to_new[og_key].add(new_key)

        return state_dict, og_keys_to_new

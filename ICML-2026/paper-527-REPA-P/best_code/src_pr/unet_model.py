import math
import torch
from torch import nn, einsum
from functools import partial
from einops import rearrange
from src_pr.einops_exts import rearrange_many
from src_pr.rotary_embedding_torch import RotaryEmbedding
import numpy as np

# helpers functions

def generalized_image_to_b_xy_c(tensor):
    """
    Transpose the tensor from [batch, channels, ..., pixel_x, pixel_y] to [batch, pixel_x*pixel_y, channels, ...]. We assume two pixel dimensions.
    """
    num_dims = len(tensor.shape) - 3  # Subtracting batch and pixel dimensions
    pattern = 'b ' + ' '.join([f'c{i}' for i in range(num_dims)]) + ' x y -> b (x y) ' + ' '.join([f'c{i}' for i in range(num_dims)])
    return rearrange(tensor, pattern)

def generalized_b_xy_c_to_image(tensor, pixels_x=None, pixels_y=None):
    """
    Transpose the tensor from [batch, pixel_x*pixel_y, channels, ...] to [batch, channels, ..., pixel_x, pixel_y] using einops.
    """
    if pixels_x is None or pixels_y is None:
        pixels_x = pixels_y = int(np.sqrt(tensor.shape[1]))
    num_dims = len(tensor.shape) - 2  # Subtracting batch and pixel dimensions (NOTE that we assume two pixel dimensions that are FLATTENED into one dimension)
    pattern = 'b (x y) ' + ' '.join([f'c{i}' for i in range(num_dims)]) + f' -> b ' + ' '.join([f'c{i}' for i in range(num_dims)]) + ' x y'
    return rearrange(tensor, pattern, x=pixels_x, y=pixels_y)

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def exists(x):
    return x is not None

def noop(*args, **kwargs):
    pass

def is_odd(n):
    return (n % 2) == 1

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

# relative positional bias

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# small helper modules

class UnsqueezeLastDim(nn.Module):
    def forward(self, x):
        return torch.unsqueeze(x, -1)

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim, padding_mode = 'zeros'):
    if padding_mode == 'zeros':
        return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1), padding_mode='zeros')
    elif padding_mode == 'circular':
        return CircularUpsample(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))
    else:
        raise ValueError('Unknown padding mode: {}'.format(padding_mode))

# WARNING: (Experimental) This is hard-coded for above kernel size, stride, and padding. Do not use for other cases.
# Use this for upsamling with circular padding in both pixel dimensions (Torch does not offer this natively).
class CircularUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super(CircularUpsample, self).__init__()
        assert kernel_size[0] == 1 and kernel_size[1] == 4 and kernel_size[2] == 4
        assert stride[0] == 1 and stride[1] == 2 and stride[2] == 2
        assert padding[0] == 0 and padding[1] == 1 and padding[2] == 1
        assert dilation == 1
        if not isinstance(dilation, tuple):
            dilation = (dilation, dilation, dilation)
        self.true_padding = (dilation[0] * (kernel_size[0] - 1) - padding[0],
                             dilation[1] * (kernel_size[1] - 1) - padding[1],
                             dilation[2] * (kernel_size[2] - 1) - padding[2])
        # this ensures that no padding is applied by the ConvTranspose3d layer since we manually apply it before
        self.removed_padding = (dilation[0] * (kernel_size[0] - 1) + stride[0] + padding[0] - 1,
                             dilation[1] * (kernel_size[1] - 1) + stride[1] + padding[1] - 1,
                             dilation[2] * (kernel_size[2] - 1) + stride[2] + padding[2] - 1)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding=self.removed_padding)

    def forward(self, x):
        true_padding_repeated = tuple(i for i in reversed(self.true_padding) for _ in range(2))
        x = nn.functional.pad(x, true_padding_repeated, mode = 'circular') # manually apply padding of 1 on all sides
        x = self.conv_transpose(x)
        return x

def Downsample(dim, padding_mode='zeros'):
    if padding_mode == 'zeros' or padding_mode == 'circular':
        return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1), padding_mode=padding_mode)
    else:
        raise ValueError('Unknown padding mode: {}'.format(padding_mode))

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, padding_mode = 'zeros', groups = 8):
        super().__init__()
        if padding_mode == 'zeros' or padding_mode == 'circular':
            self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1), padding_mode=padding_mode)
        else:
            raise ValueError('Unknown padding mode: {}'.format(padding_mode))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, padding_mode = 'zeros', groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, padding_mode = padding_mode, groups = groups)
        self.block2 = Block(dim_out, dim_out, padding_mode = padding_mode, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, cond_dim = 64):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias = False)
        self.to_k = nn.Linear(cond_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, hidden_dim, bias=False)        
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h = self.heads)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w) # added this (not included in original repo)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b = b)

# attention along space and time
class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None,
        cond_dim = 64,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)

        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_k = nn.Linear(cond_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, hidden_dim, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
        self,
        x,
        pos_bias = None,
    ):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split out heads
        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)
        if exists(self.rotary_emb):
            k = self.rotary_emb.rotate_queries_or_keys(k)
        # scale
        q = q * self.scale
        # rotate positions into queries and keys for time attention
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
        # similarity
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)
        # relative positional bias
        if exists(pos_bias):
            sim = sim + pos_bias
        # numerical stability
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        # aggregate values
        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

# convolutional encoder for 1D stress-strain response (only required for ablation study)
class SignalEmbedding(nn.Module):
    def __init__(self, cond_arch, init_channel, channel_upsamplings):
        super().__init__()
        if cond_arch == 'CNN':
            scale_factor = [init_channel, *map(lambda m: 1 * m, channel_upsamplings)]
            in_out_channels = list(zip(scale_factor[:-1], scale_factor[1:]))
            self.num_resolutions = len(in_out_channels)
            self.emb_model = self.generate_conv_embedding(in_out_channels)
        elif cond_arch == 'GRU':
            self.emb_model = nn.GRU(input_size = init_channel, hidden_size = channel_upsamplings[-1], num_layers = 3, batch_first=True)
        else:
            raise ValueError('Unknown architecture: {}'.format(cond_arch))

        self.cond_arch = cond_arch

    def Downsample1D(self, dim, dim_out = None):
        return nn.Conv1d(dim,default(dim_out, dim),kernel_size=4, stride=2, padding=1)

    def generate_conv_embedding(self, channel_upsamplings):
        embedding_modules = nn.ModuleList([])
        for idx, (ch_in, ch_out) in enumerate(channel_upsamplings):
            embedding_modules.append(self.Downsample1D(ch_in,ch_out))
            embedding_modules.append(nn.SiLU())
        return nn.Sequential(*embedding_modules)

    def forward(self, x):
        # add channel dimension for conv1d
        if len(x.shape) == 2 and self.cond_arch == 'CNN':
            x = x.unsqueeze(1)
            x = self.emb_model(x)
        elif len(x.shape) == 2 and self.cond_arch == 'GRU':
            x = x.unsqueeze(2)
            x, _ = self.emb_model(x)
        x = torch.squeeze(x)
        return x

class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 2,
        self_condition = False,
        attn_heads = 8,
        attn_dim_head = 32,
        init_dim = None,
        init_kernel_size = 7,
        use_sparse_linear_attn = True,
        resnet_groups = 8,
        cond_bias = False,
        cond_attention = 'none', # 'none', 'self-stacked', 'cross', 'self-cross/spatial'
        cond_attention_tokens = 6,
        cond_to_time = 'add',
        padding_mode = 'zeros',
        sigmoid_last_channel = False,
    ):
        super().__init__()
        self.input_channels = channels * (2 if self_condition else 1)
        self.self_condition = self_condition

        time_dim = dim * 4

        self.cond_bias = cond_bias
        self.cond_dim = time_dim
        self.cond_to_time = cond_to_time
        self.padding_mode = padding_mode

        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        # this reshapes a tensor of shape [first argument] to
        # [second argument], applies an attention layer and then transforms it back 
        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(dim, heads = attn_heads, dim_head = attn_dim_head, rotary_emb = rotary_emb, cond_dim = self.cond_dim))

        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32)

        # initial conv
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2

        if self.padding_mode == 'zeros' or self.padding_mode == 'circular':
            self.init_conv = nn.Conv3d(self.input_channels, init_dim, (1, init_kernel_size, init_kernel_size), padding = (0, init_padding, init_padding), padding_mode=self.padding_mode)
        else:
            raise ValueError('Unknown padding mode: {}'.format(self.padding_mode))
        
        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # CNN signal embedding for cond bias
        self.sign_emb_CNN = SignalEmbedding('CNN', init_channel=1, channel_upsamplings=(16, 32, 64, 128, self.cond_dim))

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # block type
        block_klass = partial(ResnetBlock, padding_mode = self.padding_mode, groups = resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim = time_dim + int(self.cond_dim or 0) if self.cond_to_time == 'concat' else self.cond_dim)

        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads = attn_heads, cond_dim = self.cond_dim))) if use_sparse_linear_attn else nn.Identity(),
                Downsample(dim_out, self.padding_mode) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads = attn_heads, cond_dim = self.cond_dim))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads = attn_heads, cond_dim = self.cond_dim))) if use_sparse_linear_attn else nn.Identity(),
                Upsample(dim_in, self.padding_mode) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

        # gradient embedding as in 'A physics-informed diffusion model for high-fidelity flow field reconstruction'
        self.emb_conv = nn.Sequential(
            torch.nn.Conv2d(channels, init_dim, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            torch.nn.Conv2d(init_dim, init_dim, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        )
        self.combine_conv = torch.nn.Conv2d(init_dim*2, init_dim, kernel_size=1, stride=1, padding=0)

        self.sigmoid_last_channel = sigmoid_last_channel

    def forward_with_guidance_scale(
        self,
        *args,
        **kwargs,
    ):
        guidance_scale = kwargs.pop('guidance_scale', 3.)
        logits = self.forward(*args, null_cond_prob = 0., **kwargs)
        if guidance_scale == 1:
            return logits
        null_logits = self.forward(*args, null_cond_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * guidance_scale

    def forward(
        self,
        x,
        time,
        x_self_cond = None,
        cond = None,
        null_cond_prob = 0.
    ):
        batch, device = x.shape[0], x.device

        # reshape x to video-like input (since this U-Net is designed for video)
        video_flag = False
        if len(x.shape) == 3:            
            x = generalized_b_xy_c_to_image(x)
            x = x.unsqueeze(2)
        elif len(x.shape) == 4:
            x = x.unsqueeze(2)
        elif len(x.shape) == 5:
            video_flag = True
        else:
            raise ValueError('Input must be image [BxCxPxP] or image sequence [BxCxFxPxP].')

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)

        # gradient conditioning, cond = dx
        if exists(cond):
            # classifier free guidance
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device=device)
            
            if len(cond.shape) == 3:
                label_mask_embed = rearrange(mask, 'b -> b 1 1')
                null_cond = torch.zeros_like(cond)
                cond = torch.where(label_mask_embed, null_cond, cond)
                cond = generalized_b_xy_c_to_image(cond)
                cond_emb = self.emb_conv(cond)
                cond = cond
            else:
                raise ValueError('Input must be [BxP*PxC].')
            
            x = torch.cat((x.squeeze(2), cond_emb), dim=1) # concatenate to channel dimension
            x = self.combine_conv(x).unsqueeze(2)

        r = x.clone()
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for block1, block2, spatial_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        x = self.final_conv(x)

        # reshape to image if we have image-like data as input
        if not video_flag:
            x = x.squeeze(2)

        if self.sigmoid_last_channel:
            # NOTE apply sigmoid on last channel of x to force E-field to be in [0,1]
            x[:, -1] = torch.sigmoid(x[:, -1])

        return x
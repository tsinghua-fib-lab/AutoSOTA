import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, t_emb):
        return self.net(t_emb)

class _ElementwiseAffineFlow(nn.Module):
    def __init__(self, data_dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(data_dim))
        self.shift = nn.Parameter(torch.zeros(data_dim))

    def forward(self, x, reverse=False):
        if reverse:
            y = (x - self.shift) * torch.exp(-self.log_scale)
            logdet = -self.log_scale.sum().expand(x.shape[0])
        else:
            y = x * torch.exp(self.log_scale) + self.shift
            logdet = self.log_scale.sum().expand(x.shape[0])
        return y, logdet


class _InvertibleLinearFlow(nn.Module):
    def __init__(self, data_dim):
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(data_dim, data_dim))
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        self.weight = nn.Parameter(q)

    def forward(self, x, reverse=False):
        if reverse:
            weight = torch.linalg.inv(self.weight)
        else:
            weight = self.weight
        y = x @ weight.T
        _, logabsdet = torch.linalg.slogdet(weight)
        return y, logabsdet.expand(x.shape[0])


class _AffineCouplingFlow(nn.Module):
    def __init__(self, data_dim, hidden_dim, mask):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * data_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, reverse=False):
        mask = self.mask.to(dtype=x.dtype, device=x.device)
        inv_mask = 1.0 - mask
        x_masked = x * mask
        shift, log_scale = self.net(x_masked).chunk(2, dim=-1)
        shift = shift * inv_mask
        log_scale = 2.0 * torch.tanh(log_scale) * inv_mask

        if reverse:
            y = x_masked + inv_mask * ((x - shift) * torch.exp(-log_scale))
            logdet = -log_scale.sum(dim=-1)
        else:
            y = x_masked + inv_mask * (x * torch.exp(log_scale) + shift)
            logdet = log_scale.sum(dim=-1)
        return y, logdet


class RealNVPFlow(nn.Module):
    """Small tabular RealNVP flow used by the normalizing-flow trainers."""

    def __init__(
        self,
        hps=None,
        data_dim=None,
        n_coupling_layers=6,
        hidden_dim=128,
        use_actnorm=False,
        use_invertible_linear=False,
    ):
        super().__init__()
        if hps is not None:
            data_dim = getattr(hps, "data_dim", data_dim)
            n_coupling_layers = getattr(hps, "n_coupling_layers", n_coupling_layers)
            hidden_dim = getattr(hps, "hidden_dim", hidden_dim)
        if data_dim is None:
            raise ValueError("RealNVPFlow requires data_dim")
        self.data_dim = int(data_dim)
        if self.data_dim < 1:
            raise ValueError("data_dim must be positive")

        layers = []
        if use_actnorm or self.data_dim == 1:
            layers.append(_ElementwiseAffineFlow(self.data_dim))
        if self.data_dim > 1:
            for i in range(int(n_coupling_layers)):
                if use_invertible_linear:
                    layers.append(_InvertibleLinearFlow(self.data_dim))
                if use_actnorm:
                    layers.append(_ElementwiseAffineFlow(self.data_dim))
                mask = (torch.arange(self.data_dim) % 2 == (i % 2)).float()
                layers.append(_AffineCouplingFlow(self.data_dim, int(hidden_dim), mask))
        self.layers = nn.ModuleList(layers)

    def _device(self):
        return next(self.parameters()).device

    def forward(self, x, reverse=False):
        x = x.view(x.shape[0], -1)
        logdet = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        layers = reversed(self.layers) if reverse else self.layers
        for layer in layers:
            x, layer_logdet = layer(x, reverse=reverse)
            logdet = logdet + layer_logdet
        return x, logdet

    def reverse(self, z):
        x, _ = self.forward(z, reverse=True)
        return x

    def sample(self, num_samples):
        z = torch.randn(int(num_samples), self.data_dim, device=self._device())
        return self.reverse(z)

    def log_prob(self, x):
        z, logdet = self.forward(x, reverse=False)
        log_base = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(dim=-1)
        return log_base + logdet


class _GlowActNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.bias = nn.Parameter(torch.zeros(1, int(num_channels), 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, int(num_channels), 1, 1))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def _initialize(self, x):
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        std = x.std(dim=(0, 2, 3), keepdim=True, unbiased=False).clamp_min(self.eps)
        self.bias.data.copy_(-mean)
        self.log_scale.data.copy_(torch.log(1.0 / std))
        self.initialized.fill_(True)

    def forward(self, x, reverse=False):
        if not bool(self.initialized.item()):
            if reverse:
                with torch.no_grad():
                    self.initialized.fill_(True)
            else:
                self._initialize(x)

        _, _, height, width = x.shape
        spatial_size = height * width
        if reverse:
            y = x * torch.exp(-self.log_scale) - self.bias
            logdet = -spatial_size * self.log_scale.view(-1).sum()
        else:
            y = (x + self.bias) * torch.exp(self.log_scale)
            logdet = spatial_size * self.log_scale.view(-1).sum()
        return y, logdet.expand(x.shape[0])


class _GlowInvertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        num_channels = int(num_channels)
        q, _ = torch.linalg.qr(torch.randn(num_channels, num_channels))
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        self.weight = nn.Parameter(q)

    def forward(self, x, reverse=False):
        _, _, height, width = x.shape
        if reverse:
            weight = torch.linalg.inv(self.weight)
        else:
            weight = self.weight
        y = F.conv2d(x, weight.view(weight.shape[0], weight.shape[1], 1, 1))
        _, logabsdet = torch.linalg.slogdet(weight)
        return y, (height * width * logabsdet).expand(x.shape[0])


class _GlowZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.log_scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x) * torch.exp(3.0 * self.log_scale)


class _GlowAffineCoupling(nn.Module):
    def __init__(self, num_channels, hidden_channels):
        super().__init__()
        self.num_channels = int(num_channels)
        self.identity = self.num_channels < 2
        if self.identity:
            self.net = None
            self.keep_channels = self.num_channels
            return

        self.keep_channels = self.num_channels // 2
        transform_channels = self.num_channels - self.keep_channels
        hidden_channels = int(hidden_channels)
        self.net = nn.Sequential(
            nn.Conv2d(self.keep_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            _GlowZeroConv2d(hidden_channels, 2 * transform_channels, 3),
        )

    def forward(self, x, reverse=False):
        if self.identity:
            return x, torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

        x_keep = x[:, : self.keep_channels]
        x_transform = x[:, self.keep_channels :]
        shift, log_scale = self.net(x_keep).chunk(2, dim=1)
        scale = torch.sigmoid(log_scale + 2.0).clamp_min(1e-6)
        logdet = torch.log(scale).view(x.shape[0], -1).sum(dim=1)

        if reverse:
            y_transform = x_transform / scale - shift
            logdet = -logdet
        else:
            y_transform = (x_transform + shift) * scale
        return torch.cat([x_keep, y_transform], dim=1), logdet


class _GlowSqueeze2d(nn.Module):
    def __init__(self, factor=2):
        super().__init__()
        self.factor = int(factor)

    def forward(self, x, reverse=False):
        factor = self.factor
        batch, channels, height, width = x.shape
        if reverse:
            if channels % (factor * factor) != 0:
                raise ValueError("Cannot unsqueeze Glow tensor with incompatible channels")
            x = x.view(batch, channels // (factor * factor), factor, factor, height, width)
            x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
            return x.view(batch, channels // (factor * factor), height * factor, width * factor)

        if height % factor != 0 or width % factor != 0:
            raise ValueError("Glow squeeze requires spatial dimensions divisible by factor")
        x = x.view(batch, channels, height // factor, factor, width // factor, factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(batch, channels * factor * factor, height // factor, width // factor)


class _GlowStep(nn.Module):
    def __init__(self, num_channels, hidden_channels):
        super().__init__()
        self.actnorm = _GlowActNorm2d(num_channels)
        self.invconv = _GlowInvertible1x1Conv(num_channels)
        self.coupling = _GlowAffineCoupling(num_channels, hidden_channels)

    def forward(self, x, reverse=False):
        logdet = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        if reverse:
            for layer in (self.coupling, self.invconv, self.actnorm):
                x, layer_logdet = layer(x, reverse=True)
                logdet = logdet + layer_logdet
        else:
            for layer in (self.actnorm, self.invconv, self.coupling):
                x, layer_logdet = layer(x, reverse=False)
                logdet = logdet + layer_logdet
        return x, logdet


class _GlowLevel(nn.Module):
    def __init__(
        self,
        num_channels,
        hidden_channels,
        n_flow_steps,
        do_squeeze,
        do_split,
        keep_channels,
    ):
        super().__init__()
        self.squeeze = _GlowSqueeze2d() if do_squeeze else None
        self.steps = nn.ModuleList(
            [_GlowStep(num_channels, hidden_channels) for _ in range(int(n_flow_steps))]
        )
        self.do_split = bool(do_split)
        self.keep_channels = int(keep_channels)


class GlowFlow(nn.Module):
    """Glow normalizing flow with ActNorm, invertible 1x1 convs, and affine coupling.

    Vector data is represented as a 1x1 image with C=data_dim channels. Larger
    image tensors use Glow's squeeze/split multiscale structure whenever the
    spatial dimensions permit it.
    """

    def __init__(
        self,
        hps=None,
        image_shape=None,
        data_dim=None,
        n_flow_steps=6,
        n_levels=3,
        hidden_channels=128,
    ):
        super().__init__()
        if hps is not None:
            image_shape = getattr(hps, "image_shape", image_shape)
            data_dim = getattr(hps, "data_dim", data_dim)
            n_flow_steps = getattr(
                hps, "n_flow_steps", getattr(hps, "n_coupling_layers", n_flow_steps)
            )
            n_levels = getattr(hps, "n_levels", n_levels)
            hidden_channels = getattr(
                hps, "hidden_channels", getattr(hps, "hidden_dim", hidden_channels)
            )

        if image_shape is None:
            if data_dim is None:
                raise ValueError("GlowFlow requires image_shape or data_dim")
            image_shape = (int(data_dim), 1, 1)
        if len(image_shape) != 3:
            raise ValueError("image_shape must be (channels, height, width)")

        self.image_shape = tuple(int(v) for v in image_shape)
        channels, height, width = self.image_shape
        if min(self.image_shape) < 1:
            raise ValueError("GlowFlow image dimensions must be positive")
        self.data_dim = channels * height * width

        levels = []
        latent_shapes = []
        c, h, w = channels, height, width
        for level_idx in range(int(n_levels)):
            do_squeeze = h > 1 and w > 1 and h % 2 == 0 and w % 2 == 0
            level_c = c * 4 if do_squeeze else c
            level_h = h // 2 if do_squeeze else h
            level_w = w // 2 if do_squeeze else w

            has_more_levels = level_idx < int(n_levels) - 1
            do_split = has_more_levels and level_c >= 2
            keep_channels = level_c // 2 if do_split else level_c
            split_channels = level_c - keep_channels
            levels.append(
                _GlowLevel(
                    level_c,
                    int(hidden_channels),
                    int(n_flow_steps),
                    do_squeeze,
                    do_split,
                    keep_channels,
                )
            )
            if do_split:
                latent_shapes.append((split_channels, level_h, level_w))
                c, h, w = keep_channels, level_h, level_w
            else:
                c, h, w = level_c, level_h, level_w
                if not do_squeeze:
                    break

        latent_shapes.append((c, h, w))
        self.levels = nn.ModuleList(levels)
        self.latent_shapes = latent_shapes
        self.latent_dim = sum(math.prod(shape) for shape in latent_shapes)

    def _device(self):
        return next(self.parameters()).device

    def _as_image(self, x):
        channels, height, width = self.image_shape
        if x.dim() == 2:
            if x.shape[1] != self.data_dim:
                raise ValueError(f"Expected flat Glow input with {self.data_dim} features")
            return x.view(x.shape[0], channels, height, width)
        if x.dim() == 4:
            if tuple(x.shape[1:]) == self.image_shape:
                return x
            if tuple(x.shape[1:]) == (height, width, channels):
                return x.permute(0, 3, 1, 2).contiguous()
            if x[0].numel() == self.data_dim:
                return x.reshape(x.shape[0], channels, height, width)
        raise ValueError(f"Expected Glow input shape (B,{channels},{height},{width})")

    def _flatten_latents(self, z_parts):
        return torch.cat([z.reshape(z.shape[0], -1) for z in z_parts], dim=1)

    def _unflatten_latents(self, z):
        if z.dim() > 2:
            z = z.reshape(z.shape[0], -1)
        if z.shape[1] != self.latent_dim:
            raise ValueError(f"Expected Glow latent with {self.latent_dim} features")

        parts = []
        offset = 0
        for shape in self.latent_shapes:
            numel = math.prod(shape)
            parts.append(z[:, offset : offset + numel].view(z.shape[0], *shape))
            offset += numel
        return parts

    def forward(self, x, reverse=False):
        if reverse:
            return self._reverse(x)

        x = self._as_image(x)
        logdet = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        z_parts = []
        for level in self.levels:
            if level.squeeze is not None:
                x = level.squeeze(x, reverse=False)
            for step in level.steps:
                x, layer_logdet = step(x, reverse=False)
                logdet = logdet + layer_logdet
            if level.do_split:
                z_parts.append(x[:, level.keep_channels :])
                x = x[:, : level.keep_channels]
        z_parts.append(x)
        return self._flatten_latents(z_parts), logdet

    def _reverse(self, z):
        z_parts = self._unflatten_latents(z)
        x = z_parts[-1]
        split_idx = len(z_parts) - 2
        logdet = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

        for level in reversed(self.levels):
            if level.do_split:
                x = torch.cat([x, z_parts[split_idx]], dim=1)
                split_idx -= 1
            for step in reversed(level.steps):
                x, layer_logdet = step(x, reverse=True)
                logdet = logdet + layer_logdet
            if level.squeeze is not None:
                x = level.squeeze(x, reverse=True)
        return x, logdet

    def reverse(self, z):
        x, _ = self.forward(z, reverse=True)
        return x

    def sample(self, num_samples):
        z = torch.randn(int(num_samples), self.latent_dim, device=self._device())
        return self.reverse(z)

    def log_prob(self, x):
        z, logdet = self.forward(x, reverse=False)
        log_base = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(dim=-1)
        return log_base + logdet


class GlowStyleFlow(GlowFlow):
    """Backward-compatible alias for the canonical Glow implementation."""

    def __init__(
        self,
        hps=None,
        data_dim=None,
        n_coupling_layers=6,
        hidden_dim=128,
        image_shape=None,
        n_levels=3,
    ):
        super().__init__(
            hps=hps,
            image_shape=image_shape,
            data_dim=data_dim,
            n_flow_steps=n_coupling_layers,
            n_levels=n_levels,
            hidden_channels=hidden_dim,
        )

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = F.silu(self.fc1(out))
        out = self.norm2(out)
        out = self.fc2(out)
        return residual + out

class MLPDenoiser(nn.Module):
    def __init__(
        self, input_dim=3, hidden_dim=128, time_embed_dim=32, time_concat=False
    ):
        super(MLPDenoiser, self).__init__()

        # If time_concat is True we avoid creating a TimeMLP and instead
        # concatenate the raw scalar timestep (shape (B,1)) directly.
        self.time_concat = bool(time_concat)
        self.time_embed_dim = 1 if self.time_concat else time_embed_dim

        # Time embedding layer (only created when not using time_concat)
        if not self.time_concat:
            # Default simple time embed (learned MLP). You can supply a custom
            # time embedding module (e.g. sinusoidal or gaussian fourier) by
            # setting `time_embed_module` when constructing the model.
            self.time_embed = nn.Sequential(
                nn.Linear(1, time_embed_dim),  # Map scalar time to higher dimension
                nn.SiLU(),  # Smooth activation
                nn.Linear(time_embed_dim, time_embed_dim),  # Further transform
            )
        else:
            # Placeholder attribute for code paths that check for `time_embed`
            self.time_embed = None

        # Optional external time embedding module (callable). If set,
        # `time_embed` will be ignored and the module will be used.
        self.time_embed_module = None

        # Main MLP layers
        print("Input dimension:", input_dim)
        print("Time embedding dimension:", self.time_embed_dim)

        self.fc1 = nn.Linear(input_dim + self.time_embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)  # Output has same dim as input

    def forward(self, x, t):
        """
        Args:
            x: Tensor of shape (batch_size, input_dim) or (batch_size, 1, 28, 28)
            t: Timestep tensor of shape (batch_size,) or (batch_size,1)
        """
        # Batch size
        B = x.shape[0]

        # 1) Flatten x to (B, D) without dropping dims
        x = x.view(B, -1)

        # 2) Ensure t is (B, 1) float
        if t.dim() == 1:
            t = t.unsqueeze(1)
        elif t.dim() > 2:
            t = t.view(B, 1)
        t = t.float()

        # 3) Time embedding -> (B, t_embed_dim)
        # If time_concat is enabled, use the raw scalar. Otherwise prefer an
        # external time_embed_module if set, falling back to the internal MLP.
        if getattr(self, "time_concat", False):
            # Ensure t is (B,1)
            if t.dim() == 1:
                t = t.unsqueeze(1)
            t_embed = t.float()
        else:
            if getattr(self, "time_embed_module", None) is not None:
                t_for_module = t.squeeze(-1) if t.dim() > 1 else t
                t_embed = self.time_embed_module(t_for_module.float())
            else:
                t_embed = self.time_embed(t)
        if t_embed.dim() == 1:
            t_embed = t_embed.view(B, -1)

        # 4) Concatenate -> (B, D + t_embed_dim)
        h = torch.cat([x, t_embed], dim=1)

        # 5) Fast sanity check: match fc1’s expected input size
        if h.size(1) != self.fc1.in_features:
            raise RuntimeError(
                f"Concat features {h.size(1)} != fc1.in_features {self.fc1.in_features} "
                f"(data {x.size(1)} + time_embed {t_embed.size(1)})"
            )

        # 6) MLP
        h = F.silu(self.fc1(h))
        h = F.silu(self.fc2(h))
        out = self.fc3(h)  # (B, D)

        # 7) Optional: ensure output matches data dim expected by fc3
        if out.size(1) != self.fc3.out_features:
            raise RuntimeError(
                f"Output features {out.size(1)} != fc3.out_features {self.fc3.out_features}"
            )

        return out

class ResBlock(nn.Module):
    """
    Residual block with time conditioning: add a per-channel bias from time embedding
    after the first activation (common DDPM trick).
    """
    def __init__(self, in_ch, out_ch, time_ch, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn1   = nn.GroupNorm(num_groups=groups, num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2   = nn.GroupNorm(num_groups=groups, num_channels=out_ch)
        self.time_proj = nn.Linear(time_ch, out_ch)  # produces channel-wise bias
        self.skip = (nn.Conv2d(in_ch, out_ch, kernel_size=1)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x, t_embed):
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.silu(h)

        # Add time bias
        tb = self.time_proj(t_embed)  # (B, out_ch)
        h = h + tb[:, :, None, None]

        h = self.conv2(h)
        h = self.gn2(h)
        h = F.silu(h)

        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_ch):
        super().__init__()
        self.block = ResBlock(in_ch, out_ch, time_ch)
        self.down  = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t_embed):
        x = self.block(x, t_embed)
        skip = x
        x = self.down(x)
        return x, skip


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ResBlock(out_ch + skip_ch, out_ch, time_ch)

    def forward(self, x, skip, t_embed):
        x = self.up(x)
        # Handle any odd-size mismatches (shouldn't happen for 28x28, but safe)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, t_embed)
        return x


class UNetDenoiser(nn.Module):
    """
    A compact UNet for MNIST:
      - Accepts either (B, 784) or (B, 1, 28, 28)
      - Time-conditioned via sinusoidal embeddings
      - Returns same shape as input
    """

    def __init__(
        self,
        in_channels=1,
        base_ch=32,
        time_embed_dim=128,
        img_size=(28, 28),
        time_mlp_width=128,
        groups=8,
        time_concat=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size  # (H, W)

        # Time embedding handling. If time_concat is True we will bypass
        # sinusoidal/TimeMLP and instead pass the raw scalar (B,1) to the
        # ResBlocks. To remain compatible with existing shapes we set the
        # internal time width to 1 when time_concat=True.
        self.time_concat = bool(time_concat)
        self.time_embed_dim = 1 if self.time_concat else time_embed_dim
        self.time_mlp_width = 1 if self.time_concat else time_mlp_width
        # Provide a small internal module that maps the scalar timestep
        # to the desired `time_embed_dim` before passing through the
        # TimeMLP. This keeps behavior consistent with the MLPDenoiser
        # and avoids shape-mismatch when the trainer passes raw scalar
        # timesteps (B,1).
        self.time_embed_module = None
        if not self.time_concat:
            self.time_embed_module = nn.Sequential(
                nn.Linear(1, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        self.time_mlp = (
            None
            if self.time_concat
            else TimeMLP(self.time_embed_dim, self.time_mlp_width)
        )

        # Encoder
        self.in_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)
        # Down / Up blocks
        self.down1 = Down(base_ch, base_ch * 2, self.time_mlp_width)  # 28x28 -> 14x14
        self.down2 = Down(base_ch * 2, base_ch * 4, self.time_mlp_width)  # 14x14 -> 7x7

        # Bottleneck
        self.mid = ResBlock(
            base_ch * 4, base_ch * 4, self.time_mlp_width, groups=groups
        )

        # Decoder
        self.up2 = Up(
            base_ch * 4, base_ch * 4, base_ch * 2, self.time_mlp_width
        )  # 7->14
        self.up1 = Up(base_ch * 2, base_ch * 2, base_ch, self.time_mlp_width)  # 14->28

        self.out_conv = nn.Conv2d(base_ch, in_channels, kernel_size=3, padding=1)

    def _reshape_in(self, x):
        """
        Canonicalize input to (B, C, H, W) for the UNet, but remember
        the original tail shape to restore the output afterwards.
        Accepts e.g.: (B, 784), (B, 784, 1), (B, 1, 784),
                      (B, 1, 28, 28), (B, 28, 28, 1), etc.
        """
        if x.dim() < 2:
            raise RuntimeError("Input must be batched (at least 2D).")
        B = x.size(0)
        orig_tail_shape = x.shape[1:]  # remember to restore
        # Squeeze all singleton dims after batch to simplify shape logic
        x = x.view(B, *orig_tail_shape)  # ensure contiguous view
        x = x.squeeze()  # squeeze *all* dims
        if x.dim() == 1:  # if squeeze killed batch (rare)
            x = x.unsqueeze(0)
        if x.size(0) != B:
            # ensure batch dimension still first
            x = x.view(B, -1)

        # If already image-like (B, C, H, W), we might just permute channels-last -> channels-first
        if x.dim() == 4:
            # channels-last? (B, H, W, C==1)
            if x.size(-1) == self.in_channels and x.size(1) != self.in_channels:
                x = x.permute(0, 3, 1, 2).contiguous()  # (B,H,W,C)->(B,C,H,W)
            # Now expected channels-first
            return x, orig_tail_shape, True

        # Otherwise, flatten and reshape to image
        C = self.in_channels
        H, W = self.img_size
        x = x.reshape(B, -1)
        D = x.size(1)
        expected = C * H * W

        if D != expected:
            # try to infer square image if img_size not matching the vector
            if D % C == 0:
                side = int(round((D // C) ** 0.5))
                if side * side * C == D:
                    H = W = side
                else:
                    raise RuntimeError(
                        f"Vector length {D} cannot be reshaped to an image with C={C}."
                    )
            else:
                raise RuntimeError(
                    f"Vector length {D} not divisible by C={C}; cannot infer (H,W)."
                )

        x_img = x.view(B, C, H, W).contiguous()
        return x_img, orig_tail_shape, False

    def forward(self, x, t):
        """
        x: any of (B, 784), (B, 784, 1), (B, 1, 784), (B, 1, 28, 28), (B, 28, 28, 1), ...
        t: (B,) or (B,1)
        """
        x_img, orig_tail_shape, already_image = self._reshape_in(x)

        t = t.float()
        if self.time_concat:
            # Use raw scalar as embedding (B,1)
            if t.dim() == 1:
                t_emb = t.unsqueeze(1)
            else:
                t_emb = t.view(t.size(0), 1)
        else:
            # Use learnable MLP for time embedding. If an internal
            # `time_embed_module` exists, first map the scalar timestep
            # to `time_embed_dim` and then pass through `time_mlp`.
            if getattr(self, "time_embed_module", None) is not None:
                # Ensure we pass a (B,1) tensor into the Linear layer
                t_for_mlp = t if t.dim() > 1 else t.unsqueeze(1)
                t_emb = self.time_embed_module(t_for_mlp.float())
                t_emb = self.time_mlp(t_emb)
            else:
                t_emb = self.time_mlp(t)

        h0 = self.in_conv(x_img)
        h1, s1 = self.down1(h0, t_emb)
        h2, s2 = self.down2(h1, t_emb)
        hmid = self.mid(h2, t_emb)
        u2 = self.up2(hmid, s2, t_emb)
        u1 = self.up1(u2, s1, t_emb)
        out = self.out_conv(u1)  # (B, C, H, W)

        # Restore original shape
        if already_image:
            # If original was channels-last, convert back
            if (
                len(orig_tail_shape) == 3
                and orig_tail_shape[-1] == self.in_channels
                and orig_tail_shape[0] != self.in_channels
            ):
                out = out.permute(0, 2, 3, 1).contiguous()  # (B,C,H,W)->(B,H,W,C)
            return out
        else:
            # Original was vector-ish (possibly with singleton tails like (B,784,1))
            B = out.size(0)
            flat = out.view(B, -1)
            return flat.view(B, *orig_tail_shape)

# --- 1D UNet for vector data ---
class UNetDenoiser1D(nn.Module):
    """
    A compact 1D UNet for vector data (e.g., protein coordinates):
      - Accepts (B, D) or (B, 1, D)
      - Time-conditioned via sinusoidal embeddings or raw scalar
      - Returns same shape as input
    """
    def __init__(self, input_dim=128, base_ch=32, time_embed_dim=128, time_concat=False):
        super().__init__()
        self.input_dim = input_dim
        self.base_ch = base_ch
        self.time_concat = bool(time_concat)
        self.time_embed_dim = 1 if self.time_concat else time_embed_dim
        # Internal scalar->embed module to map (B,1) -> (B,time_embed_dim)
        self.time_embed_module = None
        if not self.time_concat:
            self.time_embed_module = nn.Sequential(
                nn.Linear(1, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        self.time_mlp = (
            None if self.time_concat else TimeMLP(self.time_embed_dim, self.time_embed_dim)
        )

        # Encoder
        self.in_conv = nn.Conv1d(1, base_ch, kernel_size=3, stride=1, padding=1)
        self.down1 = nn.Conv1d(base_ch, base_ch * 2, kernel_size=3, stride=1, padding=1)
        self.down2 = nn.Conv1d(base_ch * 2, base_ch * 4, kernel_size=3, stride=1, padding=1)

        # Bottleneck
        self.mid = nn.Conv1d(base_ch * 4, base_ch * 4, kernel_size=3, stride=1, padding=1)

        # Decoder
        self.up2 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, kernel_size=3, stride=1, padding=1)
        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv1d(base_ch, 1, kernel_size=3, stride=1, padding=1)

        self.norm1 = nn.GroupNorm(4, base_ch * 2)
        self.norm2 = nn.GroupNorm(4, base_ch * 4)
        self.norm3 = nn.GroupNorm(4, base_ch * 2)
        self.norm4 = nn.GroupNorm(4, base_ch)

    def forward(self, x, t):
        # x: (B, D) or (B, 1, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
        B, C, D = x.shape
        t = t.float()
        if self.time_concat:
            # Use raw scalar as embedding (B,1) -> (B,1,1) then expand to (B,1,D)
            if t.dim() == 1:
                t_emb = t.unsqueeze(1)
            else:
                t_emb = t
            t_emb = t_emb.unsqueeze(2).expand(B, 1, D)
        else:
            # Use learnable MLP for time embedding. Map scalar to
            # `time_embed_dim` first if the internal module exists.
            if t.dim() == 1:
                t = t.unsqueeze(1)
            if getattr(self, "time_embed_module", None) is not None:
                t_for_module = t if t.dim() > 1 else t.unsqueeze(1)
                t_tmp = self.time_embed_module(t_for_module.float())
                t_emb = self.time_mlp(t_tmp)  # (B, time_embed_dim)
            else:
                t_emb = self.time_mlp(t.float())  # (B, time_embed_dim)

            if t_emb.shape[-1] == D:
                t_emb = t_emb.view(B, 1, D)
            else:
                t_emb = t_emb.unsqueeze(2).expand(B, t_emb.shape[1], D)
                t_emb = t_emb[:, :1, :]  # Only use first channel for addition
        # Encoder
        h0 = self.in_conv(x)
        h1 = F.silu(self.down1(h0 + t_emb))
        h1n = self.norm1(h1)
        h2 = F.silu(self.down2(h1n))
        h2n = self.norm2(h2)
        hmid = F.silu(self.mid(h2n))
        # Decoder
        u2 = F.silu(self.up2(hmid))
        u2 = u2 + h1n[..., :u2.shape[-1]]  # skip connection
        u2n = self.norm3(u2)
        u1 = F.silu(self.up1(u2n))
        u1 = u1 + h0[..., :u1.shape[-1]]  # skip connection
        u1n = self.norm4(u1)
        out = self.out_conv(u1n)
        out = out.squeeze(1) if out.shape[1] == 1 else out  # (B, D)
        return out

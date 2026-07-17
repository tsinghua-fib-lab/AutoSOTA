"""A simple MLP :class:`FlowNetMLP` with optional residual connections for velocity prediction."""

import copy

import torch
from torch import nn

ACTIVATIONS = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
    "softplus": nn.Softplus,
}


def get_activation(name: str) -> type[nn.Module]:
    """Get activation function class by name."""
    if name.lower() not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name.lower()]


class MLP(nn.Module):
    """
    MLP with optional residual connections.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_hidden_layers: Number of hidden layers
        activation_fn: Activation function name
        layernorm: Whether to use LayerNorm (pre-activation style)
        dropout: Dropout rate (0 = no dropout)
        residual_every: Add residual connection every N layers (0 = no residuals)
        zero_init: Whether to zero-initialize final layer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int,
        activation_fn: str = "silu",
        layernorm: bool = False,
        dropout: float = 0.0,
        residual_every: int = 0,
        zero_init: bool = False,
    ):
        super().__init__()
        self.residual_every = residual_every
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.layernorm = layernorm
        self.dropout_rate = dropout

        activation_cls = get_activation(activation_fn)

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Hidden layers with pre-activation design: LayerNorm → Activation → Linear → Dropout
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            layer_modules = []
            if layernorm:
                layer_modules.append(nn.LayerNorm(hidden_dim))
            layer_modules.append(activation_cls())
            layer_modules.append(nn.Linear(hidden_dim, hidden_dim))
            if dropout > 0:
                layer_modules.append(nn.Dropout(dropout))
            self.hidden_layers.append(nn.Sequential(*layer_modules))

        # Output projection with pre-activation
        output_modules = []
        if layernorm:
            output_modules.append(nn.LayerNorm(hidden_dim))
        output_modules.append(activation_cls())
        output_modules.append(nn.Linear(hidden_dim, output_dim))
        self.output_proj = nn.Sequential(*output_modules)

        if zero_init:
            nn.init.constant_(self.output_proj[-1].weight, 0)
            nn.init.constant_(self.output_proj[-1].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)

        if self.residual_every > 0:
            # Save state at start of each residual block
            h_residual = h
            for i, layer in enumerate(self.hidden_layers):
                h = layer(h)
                # Add residual at end of block (every N layers)
                if (i + 1) % self.residual_every == 0:
                    h = h + h_residual
                    h_residual = h  # Save for next block
        else:
            for layer in self.hidden_layers:
                h = layer(h)

        return self.output_proj(h)


class PositionalEmbedding(nn.Module):
    """
    Timestep embedding used in the DDPM++, ADM, and MeanFlow architectures.
    """

    def __init__(self, num_channels, max_positions=10000, endpoint=True):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.outer(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FlowNetMLP(nn.Module):
    """
    MLP velocity predictor with optional residual connections.

    Args:
        d: Data dimension
        x_emb_dim: Position embedding dimension
        t_emb_dim: Time embedding dimension
        x_hidden_layers: Number of hidden layers for x embedding (0 = linear)
        t_hidden_layers: Number of hidden layers for t embedding (0 = linear)
        num_hidden_layers: Number of hidden layers in main MLP
        hidden_dim: Hidden layer dimension
        zero_init: Whether to zero-initialize final layer
        layernorm: Whether to use LayerNorm
        activation_fn: Activation function name
        predict_log_var: Whether to predict log variance for weighted loss
        dropout: Dropout rate (0 = no dropout)
        residual_every: Add residual connection every N layers (0 = no residuals)
    """

    def __init__(
        self,
        d: int,
        x_emb_dim: int = 64,
        t_emb_dim: int = 64,
        x_hidden_layers: int = 0,
        t_hidden_layers: int = 0,
        num_hidden_layers: int = 3,
        hidden_dim: int = 256,
        zero_init: bool = False,
        layernorm: bool = False,
        activation_fn: str = "silu",
        predict_log_var: bool = False,
        dropout: float = 0.0,
        residual_every: int = 0,
    ):
        super().__init__()
        self.d = d
        self.x_emb_dim = x_emb_dim
        self.t_emb_dim = t_emb_dim
        self.x_hidden_layers = x_hidden_layers
        self.t_hidden_layers = t_hidden_layers
        self.predict_log_var = predict_log_var
        self.dropout_rate = dropout
        self.residual_every = residual_every
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.layernorm = layernorm
        self.activation_fn = activation_fn

        # Position embedding: Linear or MLP
        if x_hidden_layers > 0:
            self.x_emb = MLP(
                input_dim=d,
                hidden_dim=x_emb_dim,
                output_dim=x_emb_dim,
                num_hidden_layers=x_hidden_layers,
                activation_fn=activation_fn,
                layernorm=layernorm,
                zero_init=zero_init,
                dropout=dropout,
                residual_every=0,  # No residuals in embedding MLPs (small)
            )
        else:
            self.x_emb = nn.Linear(d, x_emb_dim)

        # Time embedding: PositionalEmbedding → Linear or MLP
        self.t_pos_emb = PositionalEmbedding(t_emb_dim)
        t_emb_input_dim = 2 * t_emb_dim  # t and dt concatenated
        t_emb_output_dim = 2 * t_emb_dim

        if t_hidden_layers > 0:
            self.t_emb = MLP(
                input_dim=t_emb_input_dim,
                hidden_dim=t_emb_output_dim,
                output_dim=t_emb_output_dim,
                num_hidden_layers=t_hidden_layers,
                activation_fn=activation_fn,
                layernorm=layernorm,
                zero_init=zero_init,
                dropout=dropout,
                residual_every=0,  # No residuals in embedding MLPs (small)
            )
        else:
            self.t_emb = nn.Linear(t_emb_input_dim, t_emb_output_dim)

        input_dim = x_emb_dim + 2 * t_emb_dim
        output_dim = d

        # Build MLP with optional residual connections and dropout
        self.v = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_hidden_layers=num_hidden_layers,
            activation_fn=activation_fn,
            layernorm=layernorm,
            zero_init=zero_init,
            dropout=dropout,
            residual_every=residual_every,
        )

        # EDM2-style: predict log variance for learned loss attenuation
        # Simple 1-hidden-layer MLP that takes only t_emb as input
        if predict_log_var:
            t_emb_dim_total = 2 * t_emb_dim  # t_emb and dt_emb concatenated
            self.log_var_mlp = nn.Sequential(
                nn.Linear(t_emb_dim_total, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, 1),
            )
            # Initialize to output log_var ≈ 0 (variance ≈ 1)
            nn.init.zeros_(self.log_var_mlp[-1].weight)
            nn.init.zeros_(self.log_var_mlp[-1].bias)

    def forward(self, x, t, dt):
        """
        Args:
            x (Tensor, shape ``(bs, *dim)``): positions at time t
            t (Tensor, shape ``(bs,)``): timepoints
            dt (Tensor, shape ``(bs,)``): time differences with respect to t

        Returns:
            v (Tensor, shape ``(bs, 2 * d)``): base velocity and correction to the base velocity
                ``v[:, :d]`` is the base velocity
                ``v[:, d:]`` is the correction velocity
        """
        x_emb = self.x_emb(x)
        t_emb = self.t_pos_emb(t.view(-1))
        dt_emb = self.t_pos_emb(dt.view(-1))
        t_emb = self.t_emb(torch.cat([t_emb, dt_emb], dim=1))
        emb = torch.cat([x_emb, t_emb], dim=1)
        v = self.v(emb)
        return v

    def get_log_var(self, t, dt):
        """
        Predict log variance for EDM2-style loss weighting :cite:p:`karras2024analyzing`.

        Args:
            t (Tensor, shape (bs,)): timepoints
            dt (Tensor, shape (bs,)): time differences

        Returns:
            log_var (Tensor, shape (bs, 1)): predicted log variance
        """
        if not self.predict_log_var:
            return None
        t_emb = self.t_pos_emb(t.view(-1))
        dt_emb = self.t_pos_emb(dt.view(-1))
        t_emb_cat = self.t_emb(torch.cat([t_emb, dt_emb], dim=1))
        return self.log_var_mlp(t_emb_cat)

    def __repr__(self) -> str:
        # Count parameters
        num_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        hidden_dim = self.hidden_dim
        num_hidden = self.num_hidden_layers

        # Architecture description
        if self.residual_every > 0:
            arch_desc = f"[{hidden_dim}] × {num_hidden} (residual every {self.residual_every})"
        else:
            arch_desc = f"[{hidden_dim}] × {num_hidden}"

        # Embedding descriptions
        if self.x_hidden_layers > 0:
            x_emb_desc = (
                f"MLP({self.d} → [{self.x_emb_dim}] × {self.x_hidden_layers} → {self.x_emb_dim})"
            )
        else:
            x_emb_desc = f"Linear({self.d} → {self.x_emb_dim})"

        if self.t_hidden_layers > 0:
            t_emb_desc = f"PositionalEmb({self.t_emb_dim}) + MLP({2*self.t_emb_dim} → [{2*self.t_emb_dim}] × {self.t_hidden_layers} → {2*self.t_emb_dim})"
        else:
            t_emb_desc = (
                f"PositionalEmb({self.t_emb_dim}) + Linear({2*self.t_emb_dim} → {2*self.t_emb_dim})"
            )

        lines = [
            "FlowNetMLP(",
            f"  input_dim={self.d}, x_emb_dim={self.x_emb_dim}, t_emb_dim={self.t_emb_dim}",
            f"  hidden_dim={hidden_dim}, num_hidden_layers={num_hidden}",
            f"  x_hidden_layers={self.x_hidden_layers}, t_hidden_layers={self.t_hidden_layers}",
            f"  dropout={self.dropout_rate}, residual_every={self.residual_every}",
            f"  layernorm={self.layernorm}, predict_log_var={self.predict_log_var}",
            "  ─────────────────────────────────────────",
            "  Architecture:",
            f"    x_emb: {x_emb_desc}",
            f"    t_emb: {t_emb_desc}",
            f"    MLP:   {self.x_emb_dim + 2*self.t_emb_dim} → {arch_desc} → {self.d}",
        ]
        if self.dropout_rate > 0:
            lines.append(f"    dropout: {self.dropout_rate} after each hidden layer")
        if self.predict_log_var:
            lines.append(f"    log_var_mlp: {2*self.t_emb_dim} → {hidden_dim//4} → 1")
        lines.extend(
            [
                "  ─────────────────────────────────────────",
                f"  Parameters: {num_params:,} total, {trainable_params:,} trainable",
                ")",
            ]
        )
        return "\n".join(lines)


def create_ema_flownet(flownet: nn.Module) -> nn.Module:
    """
    Create an EMA copy of a velocity network.

    Args:
        flownet: The velocity network to copy

    Returns:
        EMA copy with requires_grad=False
    """
    ema_flownet = copy.deepcopy(flownet)
    for p in ema_flownet.parameters():
        p.requires_grad = False
    return ema_flownet


__all__ = [
    "FlowNetMLP",
    "MLP",
    "PositionalEmbedding",
    "create_ema_flownet",
    "get_activation",
    "ACTIVATIONS",
]

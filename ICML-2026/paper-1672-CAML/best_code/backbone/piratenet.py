from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sin(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)


ACTIVATION_FN: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu,
    "gelu": F.gelu,
    "swish": F.silu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "sin": _sin,
}


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name in ACTIVATION_FN:
        return ACTIVATION_FN[name]
    raise NotImplementedError(f"Activation {name} not supported yet!")


def init_glorot_normal_(w: torch.Tensor) -> None:
    nn.init.xavier_normal_(w)


def init_zeros_(b: torch.Tensor) -> None:
    nn.init.zeros_(b)


class LinearWeightFact(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            mean: float = 0.0,
            stddev: float = 1.0,
            kernel_init: Callable[[torch.Tensor], None] = init_glorot_normal_,
            bias_init: Callable[[torch.Tensor], None] = init_zeros_,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.v = nn.Parameter(torch.empty(in_features, out_features))
        kernel_init(self.v)

        self.log_g = nn.Parameter(torch.empty(out_features))
        with torch.no_grad():
            self.log_g.normal_(mean=mean, std=stddev)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            bias_init(self.bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.exp(self.log_g)
        w = self.v * g
        y = x @ w
        if self.bias is not None:
            y = y + self.bias
        return y


class Dense(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            kernel_init: Callable[[torch.Tensor], None] = init_glorot_normal_,
            bias_init: Callable[[torch.Tensor], None] = init_zeros_,
            reparam: Optional[Dict] = None,
    ):
        super().__init__()
        self.reparam = reparam

        if reparam is None:
            self.linear = nn.Linear(in_features, out_features, bias=True)
            kernel_init(self.linear.weight)
            bias_init(self.linear.bias)
        else:
            if reparam.get("type", None) != "weight_fact":
                raise NotImplementedError(f"reparam type {reparam.get('type')} not supported.")
            self.linear = LinearWeightFact(
                in_features=in_features,
                out_features=out_features,
                bias=True,
                mean=float(reparam["mean"]),
                stddev=float(reparam["stddev"]),
                kernel_init=kernel_init,
                bias_init=bias_init,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class PeriodEmbs(nn.Module):
    def __init__(self, period: Tuple[float, ...], axis: Tuple[int, ...], trainable: Tuple[bool, ...]):
        super().__init__()
        if len(period) != len(axis) or len(axis) != len(trainable):
            raise ValueError("period, axis, trainable must have the same length.")

        self.axis = tuple(axis)

        self._period_params = nn.ParameterList()
        self._period_buffers: Dict[str, torch.Tensor] = {}

        self._period_is_trainable = list(trainable)
        for i, (p, is_tr) in enumerate(zip(period, trainable)):
            t = torch.tensor(float(p))
            if is_tr:
                self._period_params.append(nn.Parameter(t))
            else:
                name = f"period_{i}"
                self.register_buffer(name, t)
                self._period_buffers[name] = getattr(self, name)

    def _get_period_tensor(self, idx_in_axis_list: int) -> torch.Tensor:
        if self._period_is_trainable[idx_in_axis_list]:
            tr_positions = [i for i, t in enumerate(self._period_is_trainable) if t]
            param_index = tr_positions.index(idx_in_axis_list)
            return self._period_params[param_index]
        else:
            return self._period_buffers[f"period_{idx_in_axis_list}"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x_in = x.unsqueeze(0)
            squeeze_back = True
        else:
            x_in = x
            squeeze_back = False

        D = x_in.shape[-1]
        pieces = []
        for i in range(D):
            xi = x_in[..., i:i + 1]  # keep dim
            if i in self.axis:
                idx = self.axis.index(i)
                period = self._get_period_tensor(idx).to(xi.dtype).to(xi.device)
                pieces.append(torch.cos(period * xi))
                pieces.append(torch.sin(period * xi))
            else:
                pieces.append(xi)

        y = torch.cat(pieces, dim=-1)
        if squeeze_back:
            y = y.squeeze(0)
        return y


class FourierEmbs(nn.Module):
    def __init__(self, in_dim: int, embed_scale: float, embed_dim: int):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even (so embed_dim//2 frequencies).")
        self.embed_scale = float(embed_scale)
        self.embed_dim = int(embed_dim)

        # kernel: (in_dim, embed_dim//2)
        self.kernel = nn.Parameter(torch.empty(in_dim, embed_dim // 2))
        with torch.no_grad():
            self.kernel.normal_(mean=0.0, std=self.embed_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (..., in_dim)
        proj = x @ self.kernel  # (..., embed_dim//2)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class Embedding(nn.Module):
    def __init__(
            self,
            in_dim: int,
            periodicity: Optional[Dict] = None,
            fourier_emb: Optional[Dict] = None,
    ):
        super().__init__()
        self.period = None
        self.fourier = None

        cur_dim = in_dim

        if periodicity:
            self.period = PeriodEmbs(
                period=tuple(periodicity["period"]),
                axis=tuple(periodicity["axis"]),
                trainable=tuple(periodicity["trainable"]),
            )
            cur_dim = cur_dim + len(periodicity["axis"])

        if fourier_emb:
            self.fourier = FourierEmbs(
                in_dim=cur_dim,
                embed_scale=float(fourier_emb["embed_scale"]),
                embed_dim=int(fourier_emb["embed_dim"]),
            )
            cur_dim = int(fourier_emb["embed_dim"])

        self.out_dim = cur_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.period is not None:
            x = self.period(x)
        if self.fourier is not None:
            x = self.fourier(x)
        return x


class Mlp(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_layers: int = 4,
            hidden_dim: int = 256,
            out_dim: int = 1,
            activation: str = "tanh",
            periodicity: Optional[Dict] = None,
            fourier_emb: Optional[Dict] = None,
            reparam: Optional[Dict] = None,
            pi_init: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.activation_fn = get_activation(activation)

        self.embedding = Embedding(
            in_dim=input_dim,
            periodicity=periodicity,
            fourier_emb=fourier_emb,
        )
        emb_dim = self.embedding.out_dim

        self.layers = nn.ModuleList()
        cur = emb_dim
        for _ in range(num_layers):
            self.layers.append(Dense(cur, hidden_dim, reparam=reparam))
            cur = hidden_dim

        self.pi_init = None
        if pi_init is not None:
            self.pi_init = nn.Parameter(pi_init.clone().detach())
            self.out_dim = self.pi_init.shape[-1]
            self.head = None
        else:
            self.head = Dense(cur, out_dim, reparam=reparam)
            self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x)
        for layer in self.layers:
            x = self.activation_fn(layer(x))

        if self.pi_init is not None:
            y = x @ self.pi_init
        else:
            y = self.head(x)

        return x, y


class Bottleneck(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            activation: str,
            reparam: Optional[Dict],
    ):
        super().__init__()
        self.activation_fn = get_activation(activation)

        self.fc1 = Dense(output_dim, hidden_dim, reparam=reparam)
        self.fc2 = Dense(hidden_dim, hidden_dim, reparam=reparam)
        self.fc3 = Dense(hidden_dim, output_dim, reparam=reparam)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        x = self.fc3(x)
        x = x + identity
        x = self.activation_fn(x)
        return x


class PIBottleneck(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            activation: str,
            nonlinearity: float,
            reparam: Optional[Dict],
    ):
        super().__init__()
        self.activation_fn = get_activation(activation)

        self.fc1 = Dense(output_dim, hidden_dim, reparam=reparam)
        self.fc2 = Dense(hidden_dim, hidden_dim, reparam=reparam)
        self.fc3 = Dense(hidden_dim, output_dim, reparam=reparam)

        self.alpha = nn.Parameter(torch.tensor(float(nonlinearity)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        x = self.activation_fn(self.fc3(x))

        alpha = self.alpha
        x = alpha * x + (1.0 - alpha) * identity
        return x


class PIModifiedBottleneck(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            activation: str,
            nonlinearity: float,
            reparam: Optional[Dict],
    ):
        super().__init__()
        self.activation_fn = get_activation(activation)

        self.fc1 = Dense(output_dim, hidden_dim, reparam=reparam)
        self.fc2 = Dense(hidden_dim, hidden_dim, reparam=reparam)
        self.fc3 = Dense(hidden_dim, output_dim, reparam=reparam)

        self.alpha = nn.Parameter(torch.tensor(float(nonlinearity)))

    def forward(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.activation_fn(self.fc1(x))
        x = x * u + (1.0 - x) * v

        x = self.activation_fn(self.fc2(x))
        x = x * u + (1.0 - x) * v

        x = self.activation_fn(self.fc3(x))

        alpha = self.alpha
        x = alpha * x + (1.0 - alpha) * identity
        return x


class ResNet(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_layers: int = 2,
            hidden_dim: int = 256,
            out_dim: int = 1,
            activation: str = "tanh",
            periodicity: Optional[Dict] = None,
            fourier_emb: Optional[Dict] = None,
            reparam: Optional[Dict] = None,
            pi_init: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.activation_fn = get_activation(activation)

        self.embedding = Embedding(
            in_dim=input_dim,
            periodicity=periodicity,
            fourier_emb=fourier_emb,
        )
        emb_dim = self.embedding.out_dim

        self.blocks = nn.ModuleList([
            Bottleneck(
                hidden_dim=hidden_dim,
                output_dim=emb_dim,
                activation=activation,
                reparam=reparam,
            )
            for _ in range(num_layers)
        ])

        self.head = Dense(emb_dim, out_dim, reparam=reparam)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x)
        for blk in self.blocks:
            x = blk(x)
        y = self.head(x)
        return x, y


class PIResNet(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_layers: int = 2,
            hidden_dim: int = 256,
            out_dim: int = 1,
            activation: str = "tanh",
            nonlinearity: float = 0.0,
            periodicity: Optional[Dict] = None,
            fourier_emb: Optional[Dict] = None,
            reparam: Optional[Dict] = None,
            pi_init: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.activation_fn = get_activation(activation)

        self.embedding = Embedding(
            in_dim=input_dim,
            periodicity=periodicity,
            fourier_emb=fourier_emb,
        )
        emb_dim = self.embedding.out_dim

        self.blocks = nn.ModuleList([
            PIBottleneck(
                hidden_dim=hidden_dim,
                output_dim=emb_dim,
                activation=activation,
                nonlinearity=nonlinearity,
                reparam=reparam,
            )
            for _ in range(num_layers)
        ])

        self.pi_init = None
        if pi_init is not None:
            self.pi_init = nn.Parameter(pi_init.clone().detach())
            self.head = None
        else:
            self.head = Dense(emb_dim, out_dim, reparam=reparam)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x)
        for blk in self.blocks:
            x = blk(x)

        if self.pi_init is not None:
            y = x @ self.pi_init
        else:
            y = self.head(x)
        return x, y


class PirateNet(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_layers: int = 3,
            hidden_dim: int = 32,
            out_dim: int = 1,
            activation: str = "tanh",
            nonlinearity: float = 0.0,
            periodicity: Optional[Dict] = None,
            fourier_emb: Optional[Dict] = None,
            reparam: Optional[Dict] = None,
            pi_init: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.activation_fn = get_activation(activation)

        self.embedding = Embedding(
            in_dim=input_dim,
            periodicity=periodicity,
            fourier_emb=fourier_emb,
        )
        emb_dim = self.embedding.out_dim

        self.u_fc = Dense(emb_dim, hidden_dim, reparam=reparam)
        self.v_fc = Dense(emb_dim, hidden_dim, reparam=reparam)

        self.blocks = nn.ModuleList([
            PIModifiedBottleneck(
                hidden_dim=hidden_dim,
                output_dim=emb_dim,
                activation=activation,
                nonlinearity=nonlinearity,
                reparam=reparam,
            )
            for _ in range(num_layers)
        ])

        self.pi_init = None
        if pi_init is not None:
            self.pi_init = nn.Parameter(pi_init.clone().detach())
            self.head = None
        else:
            self.head = Dense(emb_dim, out_dim, reparam=reparam)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embs = self.embedding(x)
        x = embs

        u = self.activation_fn(self.u_fc(x))
        v = self.activation_fn(self.v_fc(x))

        for blk in self.blocks:
            x = blk(x, u, v)

        if self.pi_init is not None:
            y = x @ self.pi_init
        else:
            y = self.head(x)

        return y

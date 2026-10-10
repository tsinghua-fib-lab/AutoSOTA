"""Final v29 independent heads, extracted from the verified experiment.

Numerical operations and final fold settings are preserved. Unused historical
search variants and process-wide warning suppression have been removed.
"""

from __future__ import annotations

import math

from typing import Any

import torch

from torch import nn

METHOD_NAME = "released_fold_heads_variants_v29"

FEATURE_DIM = 2688

NUM_CLASSES = 3
NUM_FOLDS = 7

TEMP_MIN, TEMP_MAX = 0.5, 1.5

BIAS_BOUND = 0.4

KL_LAMBDA = 1.0

KL_TEMPERATURE = 2.0

TOTAL_STEPS = 3000  # 300 epochs x 10 batches

STAGE_DIMS = [768, 960, 960]

FOLD_SPECS = {
    0: ("stage", {"bound": 0.25, "freeze_temp": True, "use_bias": False}),
    1: ("bias_only", {"bound": 0.2}),
    2: ("stage", {"bound": 0.25, "freeze_temp": True, "bias_bound": 0.2}),
    3: ("stage", {"bound": 1.0, "kl_lambda": 2.0}),
    4: ("stage", {"bound": 1.5, "kl_lambda": 2.0}),
    5: ("bias_only", {"bound": 0.2}),
    6: ("stage", {"bound": 0.4, "freeze_temp": True, "bias_bound": 0.2}),
}

_OPTIMIZERS: dict[int, torch.optim.Optimizer] = {}


def _deterministic_linear(
    inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Elementwise linear map (no cuBLAS) so residual heads train bit-exactly.

    inputs: (N, D), weight: (D, O) -> (N, O) via broadcast-sum.
    """
    output = (inputs.unsqueeze(-1) * weight.unsqueeze(0)).sum(dim=1)
    if bias is not None:
        output = output + bias
    return output


def _fixed_seed_init(layer: nn.Linear, seed: int) -> None:
    """Deterministic weight init (fold-fixed seed, local generator).

    The protected driver builds all heads BEFORE seeding the training RNG, so
    the default global generator would give a different initialization per
    process.  A local generator keeps every fold bit-exactly reproducible.
    """
    generator = torch.Generator().manual_seed(int(seed))
    nn.init.xavier_uniform_(layer.weight, generator=generator)
    nn.init.zeros_(layer.bias)


class BiasOnlyHead(nn.Module):
    """Bias-only head: global class-prior correction (temperature frozen)."""

    def __init__(self, fold_index: int, bound: float = 0.2) -> None:
        super().__init__()
        self.fold_index = int(fold_index)
        self.bound = float(bound)
        self.bias = nn.Parameter(torch.zeros(NUM_CLASSES))

    def forward(
        self, features: torch.Tensor, released_logits: torch.Tensor
    ) -> torch.Tensor:
        return released_logits + self.bias

    def clamp_parameters(self) -> None:
        with torch.no_grad():
            self.bias.clamp_(-self.bound, self.bound)


class StageProbe(nn.Module):
    """Single-stage probe: LN -> d -> GELU -> 3, zero-init output."""

    def __init__(
        self, in_features: int, seed: int, bound: float = 1.0, bottleneck: int = 16
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.down = nn.Linear(in_features, bottleneck)
        self.up = nn.Linear(bottleneck, NUM_CLASSES)
        _fixed_seed_init(self.down, seed)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.bound = float(bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(
            _deterministic_linear(self.norm(x), self.down.weight.t(), self.down.bias)
        )
        raw = _deterministic_linear(hidden, self.up.weight.t(), self.up.bias)
        return self.bound * torch.tanh(raw / self.bound)


class StageHead(nn.Module):
    """Per-stage probe residuals + temperature/bias, KL-anchored."""

    def __init__(
        self,
        fold_index: int,
        bound: float = 1.0,
        freeze_temp: bool = False,
        bias_bound: float = BIAS_BOUND,
        probe_bottleneck: int = 16,
        kl_lambda: float = KL_LAMBDA,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.fold_index = int(fold_index)
        self.freeze_temp = bool(freeze_temp)
        self.bias_bound = float(bias_bound)
        self.kl_lambda = float(kl_lambda)
        self.use_bias = bool(use_bias)
        if self.freeze_temp:
            self.register_buffer("log_temperature", torch.tensor(0.0))
        else:
            self.log_temperature = nn.Parameter(torch.tensor(0.0))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(NUM_CLASSES))
        self.probe3 = StageProbe(
            STAGE_DIMS[0], 3000 + 10 * fold_index + 3, bound, probe_bottleneck
        )
        self.probe4 = StageProbe(
            STAGE_DIMS[1], 3000 + 10 * fold_index + 4, bound, probe_bottleneck
        )
        self.probe5 = StageProbe(
            STAGE_DIMS[2], 3000 + 10 * fold_index + 5, bound, probe_bottleneck
        )

    def forward(
        self, features: torch.Tensor, released_logits: torch.Tensor
    ) -> torch.Tensor:
        s3, s4, s5 = features[:, :768], features[:, 768:1728], features[:, 1728:]
        residual = self.probe3(s3) + self.probe4(s4) + self.probe5(s5)
        temperature = torch.exp(self.log_temperature)
        base = released_logits
        if self.use_bias:
            base = base + self.bias
        return base * temperature + residual

    def clamp_parameters(self) -> None:
        with torch.no_grad():
            if not self.freeze_temp:
                self.log_temperature.clamp_(math.log(TEMP_MIN), math.log(TEMP_MAX))
            if self.use_bias:
                self.bias.clamp_(-self.bias_bound, self.bias_bound)


def _validate_contract(features: torch.Tensor, released_logits: torch.Tensor) -> None:
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"invalid feature shape: {tuple(features.shape)}")
    if released_logits.shape != (features.shape[0], NUM_CLASSES):
        raise ValueError(
            f"invalid released-logit shape: {tuple(released_logits.shape)}"
        )


def make_optimizer(head: nn.Module) -> torch.optim.Optimizer | None:
    parameters = [
        parameter for parameter in head.parameters() if parameter.requires_grad
    ]
    if not parameters:
        return None
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-3)
    if not isinstance(head, BiasOnlyHead):
        _OPTIMIZERS[id(head)] = optimizer
    return optimizer


def _kl_anchor(logits: torch.Tensor, released_logits: torch.Tensor) -> torch.Tensor:
    teacher = torch.softmax(released_logits / KL_TEMPERATURE, dim=-1)
    student = torch.log_softmax(logits / KL_TEMPERATURE, dim=-1)
    return nn.functional.kl_div(student, teacher, reduction="batchmean")


def training_loss(
    head: nn.Module,
    features: torch.Tensor,
    released_logits: torch.Tensor,
    labels: torch.Tensor,
    fold_index: int,
) -> torch.Tensor:
    del fold_index
    _validate_contract(features, released_logits)
    logits = head(features, released_logits)
    if isinstance(head, BiasOnlyHead):
        return nn.functional.cross_entropy(logits, labels)
    kl_weight = getattr(head, "kl_lambda", KL_LAMBDA)
    return nn.functional.cross_entropy(logits, labels) + kl_weight * _kl_anchor(
        logits, released_logits
    )


def after_optimizer_step(head: nn.Module, fold_index: int) -> None:
    del fold_index
    head.clamp_parameters()
    optimizer = _OPTIMIZERS.get(id(head))
    if optimizer is None:
        return
    # Cosine decay to ~0 over TOTAL_STEPS keeps late epochs conservative.
    steps = int(optimizer.state.get("_steps", 0)) + 1
    optimizer.state["_steps"] = steps
    progress = min(1.0, steps / TOTAL_STEPS)
    lr = 3e-4 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr


def prediction_probabilities(
    head: nn.Module,
    features: torch.Tensor,
    released_logits: torch.Tensor,
    fold_index: int,
) -> torch.Tensor:
    del fold_index
    _validate_contract(features, released_logits)
    return torch.softmax(head(features, released_logits), dim=-1)


def candidate_metadata() -> dict[str, Any]:
    return {
        "method": METHOD_NAME,
        "parameter_sharing": False,
        "uses_cross_fold_inputs": False,
        "feature_dim": FEATURE_DIM,
    }


def build_fold_head(fold_index: int, feature_dim: int = FEATURE_DIM) -> nn.Module:
    """Build one final head without sharing tensors with other folds."""
    if fold_index not in FOLD_SPECS or feature_dim != FEATURE_DIM:
        raise ValueError("expected fold 0..6 and feature dimension 2688")
    kind, kwargs = FOLD_SPECS[fold_index]
    return (
        BiasOnlyHead(fold_index, **kwargs)
        if kind == "bias_only"
        else StageHead(fold_index, **kwargs)
    )

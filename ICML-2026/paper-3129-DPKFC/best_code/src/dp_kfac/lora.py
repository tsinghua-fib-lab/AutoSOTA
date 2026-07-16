"""Manual LoRA implementation compatible with Opacus GradSampleModule.

Uses plain nn.Linear layers for the low-rank adapters so that Opacus can
compute per-sample gradients without any custom hooks.
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Linear layer with frozen base weights and trainable low-rank adapters.

    Computes: y = W_frozen @ x + (B @ A) @ x * (alpha / r)

    Both lora_A and lora_B are plain nn.Linear so Opacus treats them normally.
    The frozen base weight is applied in a no_grad block.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Frozen base weight (not a parameter — just a buffer)
        self.register_buffer("base_weight", base_linear.weight.detach().clone())
        if base_linear.bias is not None:
            self.register_buffer("base_bias", base_linear.bias.detach().clone())
        else:
            self.base_bias = None

        # Trainable low-rank adapters
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        # Initialize: A ~ N(0, 1/r), B = 0 so initial output = base output
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen base forward
        with torch.no_grad():
            base_out = nn.functional.linear(x, self.base_weight, self.base_bias)

        # LoRA forward (trainable)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

        return base_out + lora_out


def apply_lora(
    model: nn.Module,
    target_modules: List[str],
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> Tuple[nn.Module, int]:
    """Replace target Linear layers with LoRALinear.

    Args:
        model: The model to modify (backbone).
        target_modules: List of substrings to match module names against.
            E.g., ["qkv", "proj", "fc"] will match any module whose name
            contains one of these strings.
        r: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: Dropout on LoRA input.

    Returns:
        (model, num_replaced): Modified model and count of replaced layers.
    """
    num_replaced = 0
    replacements = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue
        replacements[name] = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)

    # Apply replacements
    for name, new_module in replacements.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
        num_replaced += 1

    # Freeze everything except LoRA parameters
    for param_name, param in model.named_parameters():
        if "lora_A" in param_name or "lora_B" in param_name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model, num_replaced


def count_lora_params(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

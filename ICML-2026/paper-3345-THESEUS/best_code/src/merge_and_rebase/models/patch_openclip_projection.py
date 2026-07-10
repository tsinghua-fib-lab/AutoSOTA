from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import torch.nn as nn


def has_lora_compatible_proj_surface(visual: nn.Module) -> bool:
    return isinstance(getattr(visual, "lin_proj", None), nn.Linear)


def _has_openclip_forward_surface(visual: nn.Module) -> bool:
    return callable(getattr(visual, "_embeds", None)) and callable(getattr(visual, "_pool", None))


def _has_openai_clip_forward_surface(visual: nn.Module) -> bool:
    return all(
        hasattr(visual, name)
        for name in ("conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer", "ln_post")
    )


def patch_openclip_visual_proj(visual: nn.Module) -> int:
    if has_lora_compatible_proj_surface(visual):
        return 0

    proj = getattr(visual, "proj", None)
    if not isinstance(proj, torch.Tensor) or proj.ndim != 2:
        return 0

    in_features = int(proj.shape[0])
    out_features = int(proj.shape[1])
    lin_proj = nn.Linear(in_features, out_features, bias=False)
    lin_proj = lin_proj.to(device=proj.device, dtype=proj.dtype)
    with torch.no_grad():
        lin_proj.weight.copy_(proj.T)
    lin_proj.weight.requires_grad_(bool(getattr(proj, "requires_grad", False)))
    visual.lin_proj = lin_proj
    visual.register_parameter("proj", None)

    if _has_openclip_forward_surface(visual):
        def _forward_with_linear_proj(self, x: torch.Tensor):
            x = self._embeds(x)
            x = self.transformer(x)
            pooled, tokens = self._pool(x)
            pooled = self.lin_proj(pooled)
            if bool(getattr(self, "output_tokens", False)):
                return pooled, tokens
            return pooled
    elif _has_openai_clip_forward_surface(visual):
        def _forward_with_linear_proj(self, x: torch.Tensor):
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            cls = self.class_embedding.to(x.dtype)
            cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls, x], dim=1)
            x = x + self.positional_embedding.to(x.dtype)
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:, 0, :])
            return self.lin_proj(x)
    else:
        raise RuntimeError("Could not patch visual.proj: unsupported visual forward surface.")

    visual.forward = MethodType(_forward_with_linear_proj, visual)  # type: ignore[method-assign]
    setattr(visual, "_peft_patched_proj", True)
    return 1


def restore_openclip_proj_keyspace(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(sd)
    for key in list(out.keys()):
        if not key.endswith("lin_proj.weight"):
            continue
        prefix = key[: -len("lin_proj.weight")]
        out[f"{prefix}proj"] = out[key].T.contiguous()
        out.pop(key, None)
    return out

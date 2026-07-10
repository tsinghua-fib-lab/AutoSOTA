from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

from ..base import TensorDict
from ..registry import register

logger = logging.getLogger(__name__)

VALID_MASK_MODES = ("normal", "force")
VALID_VOTE_MODES = ("mean", "majority", "max")

# Re-export the type alias for convenience
GradRecipe = Callable[[nn.Module, Any], tuple[torch.Tensor, list[tuple[str, nn.Parameter]]]]


# ---------------------------------------------------------------------------
# Gradient sign computation (model-agnostic)
# ---------------------------------------------------------------------------


def compute_gradient_signs(
    model: nn.Module,
    dataloader,
    *,
    recipe: GradRecipe,
    device: str = "cuda",
    vote: str = "mean",
) -> TensorDict:
    """
    Compute the signed gradients of a loss w.r.t. model parameters.

    All model-specific logic (forward pass, loss function, which
    parameters to track) is encapsulated in ``recipe``.

    Parameters
    ----------
    model : Any ``nn.Module``.
    dataloader : Iterable of batches (format depends on recipe).
    recipe : ``(model, batch) -> (scalar_loss, [(name, param), ...])``
        See ``models.grad_recipes`` for concrete implementations.
    device : Torch device string.
    vote : ``"mean"`` (sign of avg gradient) or ``"majority"``/``"max"`` (majority vote).

    Returns
    -------
    dict[str, Tensor] : ``{param_name: sign_tensor}`` for every tracked
        parameter.
    """
    if vote not in VALID_VOTE_MODES:
        raise ValueError(f"vote must be one of {VALID_VOTE_MODES}, got '{vote}'")

    vote_mode = "majority" if vote == "max" else vote

    model.eval()
    model.zero_grad(set_to_none=True)
    model.to(device)

    total_steps = max(1, len(dataloader))
    scale = 1.0 / float(total_steps)

    # We discover trainable_params on the first batch and reuse the list.
    trainable_params: list[tuple[str, nn.Parameter]] | None = None
    params_only: list[nn.Parameter] | None = None

    sign_sums: dict[str, torch.Tensor] | None = None
    if vote_mode == "majority":
        sign_sums = {}

    iterator = tqdm(dataloader, desc="Computing gradient signs") if tqdm is not None else dataloader
    for batch in iterator:
        loss, named_params = recipe(model, batch)

        # First-batch initialisation
        if trainable_params is None:
            trainable_params = named_params
            params_only = [p for _, p in trainable_params]
            if vote_mode == "majority":
                sign_sums = {name: torch.zeros_like(p, device=device) for name, p in trainable_params}

        if vote_mode == "mean":
            loss_for_backward = loss.mean() if loss.dim() > 0 else loss
            (loss_for_backward * scale).backward()
            # Do NOT zero_grad here — gradients accumulate across batches.

        elif vote_mode == "majority":
            assert sign_sums is not None
            # Per-sample gradient voting requires unreduced loss; fall back
            # to full-batch backward when recipe returns a scalar.
            if loss.dim() == 0:
                # Scalar loss — use as a single "vote"
                assert params_only is not None
                grads = torch.autograd.grad(
                    loss * scale,
                    params_only,
                    retain_graph=False,
                    create_graph=False,
                )
                for (name, _), g in zip(trainable_params, grads, strict=True):
                    if g is not None:
                        sign_sums[name] += torch.sign(-g.detach())
            else:
                # Unreduced loss [B] — per-sample vote
                losses = loss * scale
                assert params_only is not None
                for i in range(losses.size(0)):
                    grads = torch.autograd.grad(
                        losses[i],
                        params_only,
                        retain_graph=(i < losses.size(0) - 1),
                        create_graph=False,
                    )
                    for (name, _), g in zip(trainable_params, grads, strict=True):
                        if g is not None:
                            sign_sums[name] += torch.sign(-g.detach())
            # majority mode uses autograd.grad (not .backward), safe to zero.
            model.zero_grad(set_to_none=True)

    # Collect signs
    gradient_signs: TensorDict = {}
    if trainable_params is None:
        return gradient_signs  # empty dataloader

    if vote_mode == "mean":
        for name, param in trainable_params:
            if param.grad is not None:
                gradient_signs[name] = torch.sign(-param.grad).cpu()
    elif vote_mode == "majority":
        assert sign_sums is not None
        gradient_signs = {name: torch.sign(acc).cpu() for name, acc in sign_sums.items()}

    model.zero_grad(set_to_none=True)
    return gradient_signs


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def apply_gradfix_mask(
    gradient_signs: Mapping[str, torch.Tensor],
    delta: Mapping[str, torch.Tensor],
    *,
    mask_mode: str = "normal",
) -> TensorDict:
    """
    Apply a gradient-sign mask to a task-vector delta.

    Parameters
    ----------
    gradient_signs : ``{param_name: sign_tensor}`` from :func:`compute_gradient_signs`.
    delta : task vector delta dict ``{param_name: Tensor}``.
    mask_mode : ``"normal"`` or ``"force"``.

    Returns
    -------
    Masked delta dict.
    """
    if mask_mode not in VALID_MASK_MODES:
        raise ValueError(f"mask_mode must be one of {VALID_MASK_MODES}, got '{mask_mode}'")

    out: TensorDict = {}
    for key in delta:
        t = delta[key]
        if key not in gradient_signs:
            # Keys not covered by gradient computation are passed through unchanged.
            out[key] = t
            continue
        g = gradient_signs[key].to(device=t.device, dtype=t.dtype)

        if mask_mode == "normal":
            out[key] = torch.where(torch.sign(g) == torch.sign(t), t, torch.zeros_like(t))
        elif mask_mode == "force":
            out[key] = torch.abs(t) * torch.sign(g)

    return out


# ---------------------------------------------------------------------------
# GradFix as a RebaseMethod
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GradFixRebase:
    """
    GradFix rebase method.

    Implements ``PreparedRebaseMethod``:
      - ``prepare``: compute gradient signs from target model + dataloader
      - ``apply``: mask a delta using precomputed gradient signs
      - ``transport``: prepare + apply in one call

    All model-specific logic is delegated to a **gradient recipe**
    (``models.grad_recipes``).  ``GradFixRebase`` itself is model-agnostic.

    Config knobs (passed via ``method_params`` or ``**kwargs``):
      - mask_mode: ``"normal"`` | ``"force"`` (default ``"normal"``)
      - vote: ``"mean"`` | ``"majority"`` | ``"max"`` (default ``"mean"``)
    """

    name: str = "gradfix"

    def prepare(
        self,
        *,
        target_model: nn.Module,
        target_dataloader,
        recipe: GradRecipe,
        device: str = "cuda",
        vote: str = "mean",
        **kwargs,
    ) -> TensorDict:
        """
        Compute gradient signs from target model + data.

        Parameters
        ----------
        target_model : The target base model.
        target_dataloader : Data from the target domain.
        recipe : A gradient recipe — ``(model, batch) -> (loss, params)``.
            Use ``clip_contrastive_recipe``, ``causal_lm_recipe``, etc.
        device : Torch device.
        vote : ``"mean"``, ``"majority"``, or ``"max"``.
        """
        logger.info("GradFix prepare: computing gradient signs (vote=%s) ...", vote)
        return compute_gradient_signs(
            model=target_model,
            dataloader=target_dataloader,
            recipe=recipe,
            device=device,
            vote=vote,
        )

    def apply(
        self,
        prepared: TensorDict,
        *,
        delta: Mapping[str, torch.Tensor],
        mask_mode: str = "normal",
        **kwargs,
    ) -> TensorDict:
        """Apply gradient-sign mask to a delta."""
        return apply_gradfix_mask(prepared, delta, mask_mode=mask_mode)

    def transport(
        self,
        *,
        source_base: Mapping[str, torch.Tensor],
        target_base: Mapping[str, torch.Tensor],
        delta: Mapping[str, torch.Tensor],
        strict: bool = False,
        target_model: nn.Module | None = None,
        target_dataloader=None,
        recipe: GradRecipe | None = None,
        device: str = "cuda",
        mask_mode: str = "normal",
        vote: str = "mean",
        gradient_signs: Mapping[str, torch.Tensor] | None = None,
        prepared: Mapping[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> TensorDict:
        """
        Full pipeline: prepare (if signs not provided) then apply.

        You can either:
          - pass ``gradient_signs`` or ``prepared`` directly (if you already called ``prepare``)
          - pass ``target_model`` + ``target_dataloader`` + ``recipe``
            and signs will be computed here.
        """
        if gradient_signs is None and prepared is not None:
            gradient_signs = prepared

        if gradient_signs is None:
            if target_model is None or target_dataloader is None or recipe is None:
                raise ValueError(
                    "GradFix.transport requires either gradient_signs/prepared or (target_model + target_dataloader + recipe)."
                )
            gradient_signs = self.prepare(
                target_model=target_model,
                target_dataloader=target_dataloader,
                recipe=recipe,
                device=device,
                vote=vote,
            )

        return self.apply(prepared=gradient_signs, delta=delta, mask_mode=mask_mode)


register(GradFixRebase())

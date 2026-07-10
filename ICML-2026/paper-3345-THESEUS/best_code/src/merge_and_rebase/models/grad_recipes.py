"""
Gradient recipes for GradFix.

A *gradient recipe* is a callable that, given a model and a single batch,
returns a scalar loss and the list of (name, parameter) pairs whose
gradients should be accumulated.  The recipe encapsulates **all**
model-specific knowledge (forward pass, loss function, which sub-module
to differentiate) so that ``compute_gradient_signs`` remains generic.

Usage
-----
>>> recipe = clip_contrastive_recipe(classnames, templates)
>>> signs  = compute_gradient_signs(model, dataloader, recipe=recipe)

>>> recipe = causal_lm_recipe(tokenizer)
>>> signs  = compute_gradient_signs(model, dataloader, recipe=recipe)
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

# ── type aliases ──────────────────────────────────────────────────────────

NamedParams = list[tuple[str, nn.Parameter]]
"""List of ``(name, param)`` pairs – the parameters to accumulate signs for."""

GradRecipe = Callable[[nn.Module, Any], tuple[torch.Tensor, NamedParams]]
"""
``(model, batch) -> (scalar_loss, [(name, param), ...])``

* ``scalar_loss`` must be a differentiable scalar tensor.
* ``named_params`` identifies which parameters to track; should be the
  **same list** (same order, same objects) across all batches.
"""


# ── CLIP / OpenCLIP zero-shot contrastive recipe ─────────────────────────

def clip_contrastive_recipe(
    classifier: Any,
    classnames: list[str],
    cfg: Any,
    *,
    text_features: torch.Tensor | None = None,
    device: str = "cuda",
    loss_fn: nn.Module | None = None,
    reduction: str = "mean",
    lowercase_classnames: bool = True,
) -> GradRecipe:
    """
    Build a gradient recipe for **OpenCLIP** zero-shot classification.

    The returned callable computes a contrastive cross-entropy loss over
    ``model.visual`` parameters:

    .. code-block:: text

        logits = logit_scale * (encode_image(x) @ text_features.T)
        loss   = CE(logits, labels)

    Parameters
    ----------
    classifier : An ``OpenClipClassifier`` instance.  Its
        ``_compute_zeroshot_text_features`` method is called to build
        the prompt-ensemble text features (no reimplementation).
    classnames : Class name strings.
    cfg : ``OpenClipBuildConfig`` forwarded to
        ``classifier._compute_zeroshot_text_features``.
    device : Target device for text feature computation.
    loss_fn : Override loss (default ``CrossEntropyLoss``).
    """
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be one of {'mean', 'none'}")

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction=reduction)

    normalize = classifier.normalize

    # Text features are computed lazily on first call and then cached.
    _cache: dict[str, Any] = {}

    def _recipe(model: nn.Module, batch: Any) -> tuple[torch.Tensor, NamedParams]:
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        # Lazy text feature build (done once, then reused).
        if "text_feats" not in _cache:
            if text_features is not None:
                feats = text_features.detach().to(device)
                if normalize:
                    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
                _cache["text_feats"] = feats
            else:
                prompt_classnames = [c.lower() for c in classnames] if lowercase_classnames else classnames
                _cache["text_feats"] = classifier._compute_zeroshot_text_features(
                    prompt_classnames, cfg,
                )
        text_feats = _cache["text_feats"]

        if "trainable" not in _cache:
            _cache["trainable"] = [
                (f"visual.{name}", p)
                for name, p in model.visual.named_parameters()
                if p.requires_grad
            ]
            if getattr(model, "logit_scale", None) is not None and model.logit_scale.requires_grad:
                _cache["trainable"].append(("logit_scale", model.logit_scale))
        trainable = _cache["trainable"]

        image_features = model.encode_image(images)
        if normalize:
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-12)
        logits = model.logit_scale.exp() * (image_features @ text_feats.t())
        loss = loss_fn(logits, labels)

        return loss, trainable

    return _recipe


# ── Causal / Seq2Seq LM recipe ───────────────────────────────────────────

def causal_lm_recipe(
    tokenizer: Any = None,
    *,
    device: str = "cuda",
    loss_fn: nn.Module | None = None,
) -> GradRecipe:
    """
    Build a gradient recipe for **HuggingFace causal-LM** (or seq2seq) models.

    Expects batches that are dicts with at least ``input_ids`` and ``labels``
    keys (as produced by ``build_nli_tokenized_loader`` or a standard HF
    data collator).

    The loss is the model's own language-modelling loss:

    .. code-block:: text

        out = model(input_ids=..., attention_mask=..., labels=...)
        loss = out.loss

    Parameters
    ----------
    tokenizer : Optional; currently unused but reserved for future
        prompt-based recipes.
    device : Target device.
    loss_fn : Override loss.  When ``None`` the model's built-in loss
        (``output.loss``) is used.
    """
    def _recipe(model: nn.Module, batch: Any) -> tuple[torch.Tensor, NamedParams]:
        # HF-style batch dict
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device).long()
        else:
            raise TypeError(
                f"causal_lm_recipe expects a dict batch with 'input_ids' and "
                f"'labels' keys, got {type(batch)}"
            )

        # All model parameters that require grad
        trainable = [
            (name, p)
            for name, p in model.named_parameters()
            if p.requires_grad
        ]

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        if loss_fn is not None:
            loss = loss_fn(out.logits.view(-1, out.logits.size(-1)), labels.view(-1))
        else:
            loss = out.loss

        return loss, trainable

    return _recipe


# ── Sequence-classification head recipe ──────────────────────────────────

def seq_classification_recipe(
    *,
    device: str = "cuda",
    mask_class: list[int] | None = None,
    loss_fn: nn.Module | None = None,
    reduction: str = "mean",
) -> GradRecipe:
    """
    Build a gradient recipe for **HuggingFace sequence-classification** models.

    Expects HF-style dict batches with ``input_ids``, ``attention_mask``,
    and ``labels``.

    Parameters
    ----------
    device : Target device.
    mask_class : Optional subset of output classes to evaluate.
    loss_fn : Override loss (default ``CrossEntropyLoss``).
    """
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be one of {'mean', 'none'}")
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction=reduction)

    def _recipe(model: nn.Module, batch: Any) -> tuple[torch.Tensor, NamedParams]:
        if not isinstance(batch, dict):
            raise TypeError(f"seq_classification_recipe expects dict batch, got {type(batch)}")

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = batch["labels"].to(device).long()

        trainable = [
            (name, p)
            for name, p in model.named_parameters()
            if p.requires_grad
        ]

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        if mask_class is not None:
            idx = torch.tensor(mask_class, device=logits.device, dtype=torch.long)
            logits = logits.index_select(dim=1, index=idx)
            # Re-map labels to the masked index space
            inv = {int(c): i for i, c in enumerate(mask_class)}
            labels = torch.tensor(
                [inv.get(int(y), 0) for y in labels.tolist()],
                device=labels.device,
                dtype=torch.long,
            )

        loss = loss_fn(logits, labels)
        return loss, trainable

    return _recipe

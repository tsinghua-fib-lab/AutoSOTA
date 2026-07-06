"""One-call unlearning/editing for HuggingFace causal LMs.

``edit_llm`` packages the whole pipeline — tokenize, build loaders, collect the
answer/text covariances over the **forget** and **total** sets, compute the engram,
and apply it — into a single call, so the common case is a one-liner::

    from engram import edit_llm
    edited = edit_llm(model, tokenizer, forget=forget_texts, total=forget_texts + retain_texts)

``forget`` / ``total`` items are either plain ``str`` (covariance over every real token)
or ``(prompt, answer)`` tuples (covariance over the **answer** tokens only — the prompt
is masked, matching answer-only unlearning). ``total`` is the reference distribution,
typically ``forget + retain`` or a broad corpus. Everything past tokenization is just
:class:`EngramEditor`, so the same ``alpha`` / ``scale`` / ``target_modules`` knobs apply.

To **tune ``alpha`` without recollecting**, split the call: :func:`get_engram` does the
expensive part once (tokenize + two covariance passes + one pseudo-inverse per layer) and
returns the ``alpha``-free engram; :func:`apply_engram` then applies it cheaply (a copy +
one subtraction per layer), so you can sweep ``alpha`` / ``scale`` interactively::

    engram = get_engram(model, tokenizer, forget=forget_texts, total=forget_texts + retain_texts)
    for a in (0.2, 0.4, 0.6, 0.8, 1.0):
        edited = apply_engram(model, engram, alpha=a)   # cheap; no recollection

``edit_llm`` is exactly ``get_engram`` followed by ``apply_engram``.

The tokenizer is passed in (no chat template is applied — pre-format prompts yourself if
you need one), so this module imports nothing from ``transformers``.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .config import EditorConfig
from .editor import EngramEditor
from .scaling import EngramResult, ScaleFn

_IGNORE = -100
Item = Union[str, Tuple[str, str]]


def _encode(tokenizer: Any, item: Item, max_length: int) -> dict:
    """One item -> {input_ids, labels}. str: all tokens count; (prompt, answer): answer only."""
    if isinstance(item, (tuple, list)):
        prompt, answer = item
        p_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        ids = (p_ids + a_ids)[:max_length]
        labels = ([_IGNORE] * len(p_ids) + a_ids)[:max_length]  # mask the prompt
    else:
        ids = tokenizer(item, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
        labels = list(ids)  # every real token enters the covariance
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)}


def _loader(tokenizer: Any, items: Iterable[Item], max_length: int, batch_size: int, pad_id: int) -> DataLoader:
    data: List[dict] = [_encode(tokenizer, it, max_length) for it in items]

    def collate(batch: List[dict]) -> dict:
        ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
        labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=_IGNORE)
        return {"input_ids": ids, "attention_mask": ids.ne(pad_id).long(), "labels": labels}

    return DataLoader(data, batch_size=batch_size, collate_fn=collate)


def get_engram(
    model: nn.Module,
    tokenizer: Any,
    forget: Iterable[Item],
    total: Iterable[Item],
    *,
    target_modules: Optional[Union[str, List[str]]] = None,
    layers_to_transform: Optional[Union[int, List[int]]] = None,
    layers_pattern: Optional[Union[str, List[str]]] = None,
    max_length: int = 512,
    batch_size: int = 8,
    compute_erank: bool = False,
    config: Optional[EditorConfig] = None,
    adapters: Optional[List[Any]] = None,
) -> EngramResult:
    """Tokenize forget/total, collect covariances, and compute the engram (``alpha``-free).

    The **expensive** half of :func:`edit_llm` — two covariance passes plus one
    pseudo-inverse per layer — separated out so you can run it **once** and then call
    :func:`apply_engram` cheaply for as many ``alpha`` / ``scale`` settings as you like
    (the projection ``P = W . C_target . pinv(C_total)`` carries no ``alpha``). Item types
    and selection knobs are the same as :func:`edit_llm`.

    Args:
        model: a HuggingFace causal LM (forward takes ``input_ids`` / ``attention_mask``).
        tokenizer: a HF tokenizer (no chat template is applied).
        forget: the set to unlearn — an iterable of ``str`` or ``(prompt, answer)``.
        total: the reference set (typically ``forget + retain``), same item types.
        target_modules: restrict which layers are collected (LoRA/PEFT convention).
        layers_to_transform: restrict to these decoder-layer indices.
        layers_pattern: the layer-index container name, e.g. ``"layers"``.
        max_length: tokenizer truncation length.
        batch_size: covariance-collection batch size.
        compute_erank: also compute per-layer effective ranks (needed only by
            :func:`engram.scaling.effective_rank`).
        config: an :class:`EditorConfig` (covariance storage device, bias absorption).
        adapters: optional fused-MoE adapters (e.g. ``[engram.moe.FusedExpertAdapter()]``).

    Returns:
        an :class:`engram.scaling.EngramResult` — feed it to :func:`apply_engram`.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id; set tokenizer.pad_token.")

    editor = EngramEditor(model, config or EditorConfig(), adapters=adapters)
    dev = editor._model_device
    sel = dict(target_modules=target_modules, layers_to_transform=layers_to_transform, layers_pattern=layers_pattern)
    feats = lambda b: {"input_ids": b["input_ids"].to(dev), "attention_mask": b["attention_mask"].to(dev)}
    mask_fn = lambda b: b["labels"] != _IGNORE  # answer tokens (tuples) or all real tokens (str)

    g_forget = editor.collect_statistics(
        _loader(tokenizer, forget, max_length, batch_size, pad_id), batch_fn=feats, mask_fn=mask_fn, **sel
    )
    g_total = editor.collect_statistics(
        _loader(tokenizer, total, max_length, batch_size, pad_id), batch_fn=feats, mask_fn=mask_fn, **sel
    )
    return editor.compute_engram_weights(g_forget, g_total, compute_erank=compute_erank)


def apply_engram(
    model: nn.Module,
    engram: EngramResult,
    *,
    alpha: float = 1.0,
    scale: Optional[ScaleFn] = None,
    inplace: bool = False,
    adapters: Optional[List[Any]] = None,
) -> nn.Module:
    """Apply a precomputed engram (from :func:`get_engram`): ``W <- W - alpha * f_l * P_l``.

    Cheap — a model copy plus one subtraction per layer — so call it repeatedly with
    different ``alpha`` / ``scale`` to tune the forget/retain trade-off **without
    re-collecting** covariances. ``alpha=0`` is a no-op (returns the unedited model).

    Args:
        model: the **same** model :func:`get_engram` saw (the projection was computed
            from its weights).
        engram: the :class:`engram.scaling.EngramResult` from :func:`get_engram`.
        alpha: global edit strength.
        scale: per-layer scaling function (default :func:`engram.scaling.count_ratio`,
            the paper's ``n/N``); see :mod:`engram.scaling`.
        inplace: edit ``model`` in place; otherwise edit and return a deep copy (default).
        adapters: fused-MoE adapters, if any were passed to :func:`get_engram`.

    Returns:
        the edited model.
    """
    return EngramEditor(model, adapters=adapters).apply(engram, alpha=alpha, scale=scale, inplace=inplace)


def edit_llm(
    model: nn.Module,
    tokenizer: Any,
    forget: Iterable[Item],
    total: Iterable[Item],
    *,
    alpha: float = 1.0,
    scale: Optional[ScaleFn] = None,
    target_modules: Optional[Union[str, List[str]]] = None,
    layers_to_transform: Optional[Union[int, List[int]]] = None,
    layers_pattern: Optional[Union[str, List[str]]] = None,
    max_length: int = 512,
    batch_size: int = 8,
    inplace: bool = False,
    config: Optional[EditorConfig] = None,
    adapters: Optional[List[Any]] = None,
) -> nn.Module:
    """Collect forget/total covariances over text and apply the engram, in one call.

    Exactly :func:`get_engram` followed by :func:`apply_engram`. To sweep ``alpha`` without
    recollecting, call those two directly: ``engram = get_engram(...)`` once, then
    ``apply_engram(model, engram, alpha=...)`` per setting.

    Args:
        model: a HuggingFace causal LM (or any module whose ``forward`` takes
            ``input_ids`` / ``attention_mask``).
        tokenizer: a HF tokenizer (used as ``tokenizer(text)["input_ids"]``; no chat
            template is applied).
        forget: the set to unlearn — an iterable of ``str`` or ``(prompt, answer)``.
        total: the reference set (typically ``forget + retain`` or a broad corpus), same
            item types as ``forget``.
        alpha: global edit strength.
        scale: per-layer scaling function (default :func:`engram.scaling.count_ratio`,
            the paper's ``n/N``); see :mod:`engram.scaling`.
        target_modules: restrict which layers are edited (LoRA/PEFT convention; list =
            name suffix, str = regex), forwarded to :meth:`EngramEditor.collect_statistics`.
        layers_to_transform: restrict to these decoder-layer indices (forwarded).
        layers_pattern: the layer-index container name, e.g. ``"layers"`` (forwarded).
        max_length: tokenizer truncation length.
        batch_size: covariance-collection batch size.
        inplace: edit ``model`` in place; otherwise edit and return a deep copy.
        config: an :class:`EditorConfig` (covariance storage device, bias absorption).
        adapters: optional fused-MoE adapters (e.g. ``[engram.moe.FusedExpertAdapter()]``).

    Returns:
        the edited model.
    """
    engram = get_engram(
        model, tokenizer, forget, total,
        target_modules=target_modules, layers_to_transform=layers_to_transform,
        layers_pattern=layers_pattern, max_length=max_length, batch_size=batch_size,
        config=config, adapters=adapters,
    )
    return apply_engram(model, engram, alpha=alpha, scale=scale, inplace=inplace, adapters=adapters)

import copy
import logging
import os
import types
from typing import Optional, Tuple

import torch.nn as nn

from llava.model.builder import load_pretrained_model

from .adapters import MAAAdapter


LOGGER = logging.getLogger(__name__)


def maa_vision_tower_forward(self, pixel_values, output_hidden_states: bool = False, **kwargs):
    return self.vision_tower(pixel_values, output_hidden_states=output_hidden_states, **kwargs)


def _wrap_layer_with_maa_adapter(layer: nn.Module, dim: int, kernel_size: int) -> None:
    if not hasattr(layer, "maa_adapter"):
        layer.maa_adapter = MAAAdapter(dim, kernel_size=kernel_size)

    if hasattr(layer, "_maa_original_forward"):
        return

    layer._maa_original_forward = layer.forward

    def new_forward(self, *args, **kwargs):
        hidden_states = None
        new_args = list(args)

        if new_args:
            hidden_states = new_args.pop(0)
        elif "hidden_states" in kwargs:
            hidden_states = kwargs.pop("hidden_states")
        elif "inputs_embeds" in kwargs:
            hidden_states = kwargs.pop("inputs_embeds")

        if hidden_states is None:
            raise TypeError("MAA adapter wrapper expected hidden states but received none.")

        adapted = self.maa_adapter(hidden_states)
        try:
            return self._maa_original_forward(adapted, *new_args, **kwargs)
        except TypeError:
            return self._maa_original_forward(hidden_states=adapted, *new_args, **kwargs)

    layer.forward = types.MethodType(new_forward, layer)


def inject_maa_adapters(vision_tower, kernel_size: int = 3, num_maa_layers = None):
    if num_maa_layers is None:
        num_maa_layers = int(__import__("os").environ.get("MAA_NUM_LAYERS", "0")) or None
    try:
        vision_encoder = vision_tower.vision_tower.vision_model.encoder
    except AttributeError as exc:
        raise RuntimeError(
            "Could not locate the CLIP vision encoder at "
            "vision_tower.vision_tower.vision_model.encoder."
        ) from exc

    dim = int(vision_encoder.config.hidden_size)
    num_layers = num_maa_layers if num_maa_layers else len(vision_encoder.layers)
    for layer in list(vision_encoder.layers)[:num_layers]:
        _wrap_layer_with_maa_adapter(layer, dim=dim, kernel_size=kernel_size)

    vision_tower.maa_forward = types.MethodType(maa_vision_tower_forward, vision_tower)
    LOGGER.info("Injected MAA adapters into %d vision encoder layers.", len(vision_encoder.layers))
    return vision_tower


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return trainable, total


def prepare_maa_model(
    model_name_or_path: str,
    kernel_size: int = 3,
    trainable: bool = True,
    with_teacher: bool = True,
):
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_name_or_path,
        model_base=None,
        model_name=os.path.basename(model_name_or_path),
    )

    _set_requires_grad(model, False)

    vision_tower_pristine: Optional[nn.Module] = None
    if with_teacher:
        vision_tower_pristine = copy.deepcopy(model.get_vision_tower())
        vision_tower_pristine.maa_forward = types.MethodType(
            maa_vision_tower_forward, vision_tower_pristine
        )
        vision_tower_pristine.eval()
        _set_requires_grad(vision_tower_pristine, False)

    vision_tower = model.get_vision_tower()
    _set_requires_grad(vision_tower, False)
    inject_maa_adapters(vision_tower, kernel_size=kernel_size)

    for name, param in model.named_parameters():
        param.requires_grad = trainable and ".maa_adapter." in name

    trainable_count, total_count = _count_trainable_parameters(model)
    LOGGER.info(
        "Prepared MAA model with %d trainable parameters (%.4f%% of total).",
        trainable_count,
        trainable_count / max(1, total_count) * 100,
    )
    return model, tokenizer, image_processor, vision_tower_pristine

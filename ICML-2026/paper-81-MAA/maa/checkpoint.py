from typing import Dict, Mapping, MutableMapping, Tuple

import torch
import torch.nn as nn


def _strip_module_prefix(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def remap_legacy_adapter_keys(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped = {}
    for key, value in state_dict.items():
        key = _strip_module_prefix(key)
        key = key.replace(".content_adapter.", ".maa_adapter.")
        remapped[key] = value
    return remapped


def _select_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, MutableMapping):
        raise TypeError("Adapter checkpoint must be a state dict or a mapping containing a state dict.")

    for key in ("adapter_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, MutableMapping):
            return value
    return checkpoint


def extract_maa_adapter_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value for key, value in model.state_dict().items() if ".maa_adapter." in key}


def save_maa_adapter_state(model: nn.Module, path: str) -> None:
    torch.save(extract_maa_adapter_state(model), path)


def load_maa_adapter_state(
    model: nn.Module,
    adapter_path: str,
    map_location: str = "cpu",
    strict: bool = False,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    checkpoint = torch.load(adapter_path, map_location=map_location)
    state_dict = remap_legacy_adapter_keys(_select_state_dict(checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=strict)
    # Move MAA adapter parameters to the same device as their parent encoder
    # layers.  Needed when the model uses accelerate device_map (e.g. from
    # low_cpu_mem_usage=True) and load_state_dict keeps new parameters on CPU.
    vision_tower = model.get_vision_tower()
    encoder_layers = vision_tower.vision_tower.vision_model.encoder.layers
    for layer in encoder_layers:
        if not hasattr(layer, "maa_adapter"):
            continue
        # Find the device of any non-adapter parameter in this layer
        target_device = None
        for n, p in layer.named_parameters():
            if ".maa_adapter." not in n:
                target_device = p.device
                break
        if target_device is None or target_device.type == "cpu":
            continue
        # Move the whole maa_adapter sub-module to the correct device
        adapter = layer.maa_adapter
        if isinstance(adapter, nn.Module):
            needs_move = any(p.device.type == "cpu" for p in adapter.parameters())
            if needs_move:
                adapter.to(target_device)
                # Also cast adapter to the same dtype as the parent layer
                if target_device.type != "cpu":
                    target_dtype = None
                    for n, p in layer.named_parameters():
                        if ".maa_adapter." not in n:
                            target_dtype = p.dtype
                            break
                    if target_dtype is not None:
                        adapter.to(target_dtype)

    # Apply alpha scaling for inference-time optimization (CODE-ALPHA)
    _alpha_scale = float(__import__("os").environ.get("MAA_ALPHA_SCALE", "1.0"))
    if _alpha_scale != 1.0:
        for layer in encoder_layers:
            if hasattr(layer, "maa_adapter"):
                adp = layer.maa_adapter
                if hasattr(adp, "alpha_mlp"):
                    adp.alpha_mlp.data *= _alpha_scale
                if hasattr(adp, "alpha_pool"):
                    adp.alpha_pool.data *= _alpha_scale

    return tuple(incompatible.missing_keys), tuple(incompatible.unexpected_keys)

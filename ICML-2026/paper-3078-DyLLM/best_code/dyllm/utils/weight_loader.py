from __future__ import annotations
from typing import Dict, Generator, Iterable, Optional, Tuple, Callable, Any, Union
import os
import glob
import torch
from torch import nn
from safetensors.torch import safe_open


def _list_safetensor_files(path: str) -> list[str]:
    if os.path.isfile(path) and path.endswith(".safetensors"):
        return [path]
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "**", "*.safetensors"), recursive=True))
    raise FileNotFoundError(f"No .safetensors found at: {path}")


def safetensors_weights_iterator(
    path_or_file: str,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Yield (name, tensor) pairs across all .safetensors files in a folder/file."""
    for st_file in _list_safetensor_files(path_or_file):
        with safe_open(st_file, framework="pt") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


LoaderFunction = Callable[[torch.Tensor, torch.Tensor], None]


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default: shape must match; scalars broadcast."""
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
    else:
        if param.shape != loaded_weight.shape:
            raise RuntimeError(f"Shape mismatch for {param.shape=} vs {loaded_weight.shape=}")
        param.data.copy_(loaded_weight)


def row_parallel_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """
    Load weights sharded along dim=0 (row-parallel). Assumes the parameter
    knows its local slice size and (optionally) sets param.weight_loader itself.
    If no TP is used, this is just a copy.
    """
    # If the param supplies its own loader delegate.
    wl = getattr(param, "weight_loader", None)
    if callable(wl):
        wl(param, loaded_weight)
        return
    # Fallback: assume single-rank (no sharding).
    default_weight_loader(param, loaded_weight)


def sharded_weight_loader(shard_axis: int) -> LoaderFunction:
    """Create a loader that narrows loaded_weight along shard_axis to this rank's local slice.
    If the param provides its own 'weight_loader', we delegate to it.
    """

    def _loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        wl = getattr(param, "weight_loader", None)
        if callable(wl):
            wl(param, loaded_weight)
            return
        # Fallback: single-rank (no shard) path
        default_weight_loader(param, loaded_weight)

    return _loader


def maybe_remap_kv_scale_name(name: str, params_dict: Dict[str, torch.nn.Parameter]) -> Optional[str]:
    """
    Normalize older/alternate kv scale names to the current expected names.
    Returns a (possibly) remapped name or None if it shouldn't be loaded.
    """
    # Legacy: ".kv_scale" (single tensor) -> ".attn.k_scale" (we'll also mirror into v_scale in LLaDA loader)
    if name.endswith(".kv_scale"):
        remapped = name.replace(".kv_scale", ".attn.k_scale")
        if remapped in params_dict:
            return remapped
        return None

    # Common: ".k_scale" or ".v_scale" might lack the ".attn." hop
    # e.g. "...self_attn.k_scale" -> "...self_attn.attn.k_scale"
    if name.endswith(".k_scale") and ".attn." not in name:
        remapped = name.replace(".self_attn.", ".self_attn.attn.")
        if remapped in params_dict:
            return remapped
    if name.endswith(".v_scale") and ".attn." not in name:
        remapped = name.replace(".self_attn.", ".self_attn.attn.")
        if remapped in params_dict:
            return remapped

    return name


def load_model(
    model: torch.nn.Module,
    path_or_file: str,
    *,
    filter_fn: Optional[Callable[[str], bool]] = None,
) -> set[str]:
    """
    Convenience: iterate .safetensors and call model.load_weights(generator).
    Returns set of loaded parameter names (as reported by the model).
    """

    def _gen():
        for name, tensor in safetensors_weights_iterator(path_or_file):
            if filter_fn is not None and not filter_fn(name):
                continue
            # print(f"{name}") model.transforemer.blocks.1.ff_proj
            yield name, tensor

    if not hasattr(model, "load_weights"):
        raise AttributeError("Model must implement .load_weights(weights_iter).")
    return model.load_weights(_gen())


class AutoWeightsLoader:
    """
    Simplified loader that:
      - Skips keys by prefix (e.g., 'lm_head.' when tying embeddings)
      - Supports 'packed' params (e.g., q_proj/k_proj/v_proj -> qkv_proj)
        if the module exposes `packed_modules_mapping = {"qkv_proj":["q_proj","k_proj","v_proj"], ...}`
      - Delegates to a param's own `weight_loader(param, tensor, *maybe_shard)` if present.
    """

    def __init__(self, root_module: nn.Module, skip_prefixes: Optional[list[str]] = None):
        self.root = root_module
        self.skip_prefixes = tuple(skip_prefixes or [])
        self.params: Dict[str, torch.nn.Parameter] = dict(self.root.named_parameters())
        self.packed = getattr(self.root, "packed_modules_mapping", {})
        self.norm_name = self.root.normalize_weight_name

    def _skipped(self, name: str) -> bool:
        return any(name.startswith(p) for p in self.skip_prefixes)

    def _apply(self, pname: str, tensor: torch.Tensor, shard_id=None):
        param: nn.Parameter = dict(self.root.named_parameters())[pname]

        wl = getattr(param, "weight_loader", None)
        if callable(wl):
            if shard_id is None:
                wl(param, tensor)
            else:
                wl(param, tensor, shard_id)
        else:
            default_weight_loader(param, tensor)

    def _try_packed_routes(self, name: str, tensor: torch.Tensor) -> bool:
        shard_map = {
            "qk_proj": {"q_proj": "q", "k_proj": "k"},
            "kv_proj": {"k_proj": "k", "v_proj": "v"},
            "qkv_proj": {"q_proj": "q", "k_proj": "k", "v_proj": "v"},
            "gate_up_proj": {"gate_proj": 0, "up_proj": 1},
        }
        for packed_key, parts in self.packed.items():
            for idx, part in enumerate(parts):
                if name.endswith(f".{part}.weight") or name.endswith(f".{part}.bias") or f".{part}." in name:
                    cand = name.replace(f".{part}", f".{packed_key}")
                    if cand in self.params:
                        self._apply(cand, tensor, shard_id=shard_map[packed_key][part])
                        return True
        return False

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:  # weights safetensor name
        loaded: set[str] = set()
        for name, tensor in weights:
            param_name = self.norm_name(name)  # model.layers.#.self.
            if self._skipped(param_name):
                continue

            if param_name in self.params:
                self._apply(param_name, tensor)
                loaded.add(param_name)
                continue

            # try packed remap (q/k/v -> qkv; gate/up -> gate_up)
            if self._try_packed_routes(param_name, tensor):
                for packed_key, parts in self.packed.items():
                    for part in parts:
                        if f".{part}" in param_name:
                            param_name = param_name.replace(part, packed_key)
                            loaded.add(param_name)
                            break

        return loaded

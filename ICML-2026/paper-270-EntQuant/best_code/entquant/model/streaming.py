"""Safetensors I/O and block-streaming utilities.

Lazy safetensors reading, key grouping by block pattern, device
resolution, and per-block shard saving.
"""

import json
import logging
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..utils import DeviceMap
from .utils import non_persistent_buffer_names

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


class ShardReader:
    """Lazy safetensors reader that handles multi-shard models.

    Memory-maps shard files and reads tensors on demand.
    """

    def __init__(self, model_id: str | Path):
        """Initialize reader from a model directory.

        Args:
            model_id: Path to the directory containing safetensors
                shard files (and optionally an index JSON).
        """
        self.model_dir = Path(model_id)
        self._handles: dict[str, Any] = {}
        self._key_to_shard: dict[str, str] = {}
        self._init_shards()

    def _init_shards(self) -> None:
        """Discover shard files and build key-to-shard mapping."""
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            self._key_to_shard = index["weight_map"]
        else:
            # Single-shard model
            shard_path = self.model_dir / "model.safetensors"
            if not shard_path.exists():
                raise FileNotFoundError(f"No safetensors found in {self.model_dir}")
            handle = safe_open(str(shard_path), framework="pt")
            for key in handle.keys():
                self._key_to_shard[key] = "model.safetensors"
            self._handles["model.safetensors"] = handle

    def _get_handle(self, shard_file: str) -> Any:
        """Get or open a safetensors file handle."""
        if shard_file not in self._handles:
            path = self.model_dir / shard_file
            self._handles[shard_file] = safe_open(str(path), framework="pt")
        return self._handles[shard_file]

    def get_tensor(self, key: str) -> torch.Tensor:
        """Read a single tensor from the appropriate shard."""
        shard_file = self._key_to_shard[key]
        handle = self._get_handle(shard_file)
        return handle.get_tensor(key)

    def keys(self) -> list[str]:
        """All available tensor keys across all shards."""
        return list(self._key_to_shard.keys())

    def close(self) -> None:
        """Close all open file handles."""
        self._handles.clear()


def open_safetensors_shards(model_id: str | Path) -> ShardReader:
    """Open safetensors shards for lazy reading.

    Args:
        model_id: Path to the model directory.

    Returns:
        ShardReader instance for tensor access.
    """
    return ShardReader(model_id)


# ---------------------------------------------------------------------------
# Block discovery
# ---------------------------------------------------------------------------


def _discover_blocks(
    keys: list[str],
    block_pattern: str,
) -> tuple[dict[str, list[str]], list[str]]:
    """Group keys by block pattern using longest prefix match.

    Discovers block prefixes matching the pattern, then assigns each
    key to its longest matching prefix. Keys not matching any block
    are returned separately.

    Args:
        keys: Flat list of dotted key names.
        block_pattern: fnmatch pattern for block prefixes.

    Returns:
        (block_groups, non_block_keys) where block_groups maps
        block_name -> list of keys.
    """
    # Discover unique block prefixes
    block_prefixes: set[str] = set()
    for key in keys:
        parts = key.split(".")
        for i in range(1, len(parts)):
            prefix = ".".join(parts[:i])
            if fnmatch(prefix, block_pattern):
                block_prefixes.add(prefix)
                break

    block_groups: dict[str, list[str]] = {p: [] for p in block_prefixes}
    non_block_keys: list[str] = []

    for key in keys:
        best_match = ""
        for prefix in block_prefixes:
            if key.startswith(prefix + ".") and len(prefix) > len(best_match):
                best_match = prefix
        if best_match:
            block_groups[best_match].append(key)
        else:
            non_block_keys.append(key)

    return block_groups, non_block_keys


def group_keys_by_block(
    model_id: str | Path,
    block_pattern: str = "model.layers.*",
) -> tuple[dict[str, list[str]], list[str]]:
    """Group safetensors keys by block pattern.

    Reads the safetensors index and groups keys by which block they
    belong to, using longest prefix match. Keys not matching any
    block are returned separately as non-block keys.

    Args:
        model_id: Path to the model directory.
        block_pattern: fnmatch pattern for block prefixes.

    Returns:
        (block_groups, non_block_keys) where block_groups maps
        block_name -> list of keys.
    """
    model_dir = Path(model_id)
    index_path = model_dir / "model.safetensors.index.json"

    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        all_keys = list(index["weight_map"].keys())
    else:
        shard_path = model_dir / "model.safetensors"
        if not shard_path.exists():
            raise FileNotFoundError(f"No safetensors found in {model_dir}")
        with safe_open(str(shard_path), framework="pt") as f:
            all_keys = list(f.keys())

    block_groups, non_block_keys = _discover_blocks(all_keys, block_pattern)
    logger.info(f"Grouped {len(all_keys)} keys into {len(block_groups)} blocks + {len(non_block_keys)} non-block keys")
    return block_groups, non_block_keys


def group_parameters_by_block(
    model: torch.nn.Module,
    block_pattern: str = "model.layers.*",
) -> tuple[dict[str, list[str]], list[str]]:
    """Group model parameter keys by block pattern.

    Same logic as group_keys_by_block but operates on in-memory
    named_parameters instead of safetensors index keys.

    Args:
        model: The model with materialized parameters.
        block_pattern: fnmatch pattern for block prefixes.

    Returns:
        (block_groups, non_block_keys) where block_groups maps
        block_name -> list of parameter keys.
    """
    all_keys = [name for name, _ in model.named_parameters()]
    # Include only persistent buffers; non-persistent ones (e.g., rotary
    # inv_freq) are recomputed from config and must not be serialized.
    non_persistent = non_persistent_buffer_names(model)
    all_keys += [
        name
        for name, _ in model.named_buffers()
        if not name.endswith("._compressed_weights") and name not in non_persistent
    ]

    return _discover_blocks(all_keys, block_pattern)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def longest_prefix_match(
    name: str,
    mapping: dict[str, Any],
    default: Any = "cuda:0",
) -> Any:
    """Resolve value for a dotted name by longest matching prefix.

    Args:
        name: Dotted name to resolve (e.g. "model.layers.0").
        mapping: Prefix -> value dict.
        default: Fallback if no prefix matches. Also checks
            mapping[""] as a default.

    Returns:
        Value from the longest matching prefix.
    """
    best, result = "", mapping.get("", default)
    for prefix, value in mapping.items():
        if (name == prefix or name.startswith(prefix + ".")) and len(prefix) > len(best):
            best, result = prefix, value
    return result


def resolve_block_devices(
    model: torch.nn.Module,
    block_groups: dict[str, list[str]],
    device_map: DeviceMap,
    weight_dtype: torch.dtype,
) -> dict[str, torch.device]:
    """Resolve target device for each block.

    Args:
        model: Meta-device model skeleton.
        block_groups: Block name -> key list mapping.
        device_map: A single device string (e.g. ``"cuda"``,
            ``"cpu"``, ``"cuda:1"``), ``"auto"``, an explicit dict,
            or None (defaults to ``"cuda"``).
        weight_dtype: Quantized weight dtype for memory estimation
            (only used with device_map="auto").

    Returns:
        Dict mapping block names to target devices.
    """
    if device_map is None:
        target = torch.device("cuda")
        return {name: target for name in block_groups}

    if isinstance(device_map, dict):
        return {name: torch.device(longest_prefix_match(name, device_map)) for name in block_groups}

    if device_map == "auto":
        from accelerate import infer_auto_device_map

        # Gather no-split module classes from blocks
        no_split = set()
        for block_name in block_groups:
            try:
                block = model.get_submodule(block_name)
                no_split.add(type(block).__name__)
            except AttributeError:
                pass
        if hasattr(model, "_no_split_modules"):
            no_split.update(model._no_split_modules)

        auto_map = infer_auto_device_map(
            model,
            dtype=weight_dtype,
            no_split_module_classes=list(no_split),
        )
        logger.info(f"Auto device map: {auto_map}")

        return {name: torch.device(longest_prefix_match(name, auto_map)) for name in block_groups}

    # Single device string (e.g. "cuda", "cpu", "cuda:1")
    target = torch.device(device_map)
    return {name: target for name in block_groups}


def resolve_non_block_device(
    key: str,
    block_device_map: dict[str, torch.device],
    device_map: DeviceMap,
) -> torch.device:
    """Resolve device for a non-block parameter key.

    Uses longest prefix match for explicit device maps, falls back
    to the first GPU in block_device_map for "auto".

    Args:
        key: Parameter key to resolve.
        block_device_map: Block name -> device mapping.
        device_map: A single device string (e.g. ``"cuda"``,
            ``"cpu"``, ``"cuda:1"``), ``"auto"``, an explicit dict,
            or None (defaults to ``"cuda"``).

    Returns:
        Target device for the parameter.
    """
    if device_map is None:
        return torch.device("cuda")

    if isinstance(device_map, dict):
        return torch.device(longest_prefix_match(key, device_map))

    if device_map == "auto":
        # Use first GPU from block device map
        if block_device_map:
            gpu_devices = [d for d in block_device_map.values() if d.type == "cuda"]
            if gpu_devices:
                return gpu_devices[0]
        return torch.device("cuda")

    # Single device string
    return torch.device(device_map)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_block_shard(
    block_module: torch.nn.Module,
    block_name: str,
    save_dir: str | Path,
    shard_idx: int,
    total_shards: int,
) -> dict[str, str]:
    """Save a block's weights as a safetensors shard.

    Uses block_module.state_dict() which triggers quanto's
    _save_to_state_dict (Marlin unpack -> float8 + scale).

    Args:
        block_module: The block module to save.
        block_name: Full dotted name for key prefixing.
        save_dir: Output directory.
        shard_idx: 1-based shard index.
        total_shards: Total number of shards.

    Returns:
        Dict mapping key -> shard filename.
    """
    save_dir = Path(save_dir)
    shard_name = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"

    # Get state dict and prefix keys with block name.
    # Skip _compressed_weights buffers (runtime ANS compression artifacts).
    state_dict = block_module.state_dict()
    prefixed: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key == "_compressed_weights":
            continue
        full_key = f"{block_name}.{key}"
        prefixed[full_key] = tensor.cpu().contiguous()

    save_file(prefixed, str(save_dir / shard_name))
    logger.debug(f"Saved block {block_name} to {shard_name} ({len(prefixed)} tensors)")

    return {key: shard_name for key in prefixed}


def save_non_block_shard(
    model: torch.nn.Module,
    non_block_keys: list[str],
    save_dir: str | Path,
    total_shards: int,
) -> dict[str, str]:
    """Save non-block weights (embeddings, norms, head) as shard 1.

    Args:
        model: The model with non-block weights materialized.
        non_block_keys: Parameter keys to save.
        save_dir: Output directory.
        total_shards: Total number of shards.

    Returns:
        Dict mapping key -> shard filename.
    """
    save_dir = Path(save_dir)
    shard_name = f"model-00001-of-{total_shards:05d}.safetensors"

    tensors: dict[str, torch.Tensor] = {}
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    for key in non_block_keys:
        tensor = params.get(key)
        if tensor is None:
            tensor = buffers.get(key)
        if tensor is not None:
            tensors[key] = tensor.cpu().contiguous()
        else:
            logger.warning(f"Non-block key {key} not found in model")

    if tensors:
        save_file(tensors, str(save_dir / shard_name))
        logger.debug(f"Saved non-block weights to {shard_name} ({len(tensors)} tensors)")

    return {key: shard_name for key in tensors}


def write_index_and_config(
    save_dir: str | Path,
    config: Any,
    weight_map: dict[str, str],
    quant_config: dict[str, Any],
) -> None:
    """Write model.safetensors.index.json and update config.json.

    Args:
        save_dir: Output directory.
        config: Model config object (has to_dict() and to_json_file()).
        weight_map: Complete key -> shard filename mapping.
        quant_config: Quantization config dict for config.json.
    """
    save_dir = Path(save_dir)

    # Write safetensors index
    index = {
        "metadata": {"total_size": 0},  # placeholder
        "weight_map": weight_map,
    }
    with open(save_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Write config.json with quantization_config
    config_dict = config.to_dict()
    config_dict["quantization_config"] = quant_config
    with open(save_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Wrote index and config to {save_dir}")


def write_quanto_qmap(
    save_dir: str | Path,
    qmap: dict[str, dict[str, str]],
) -> None:
    """Write quanto_qmap.json for quanto checkpoint compatibility.

    Args:
        save_dir: Output directory.
        qmap: Module name -> {"weights": qtype_name, "activations": "none"}.
    """
    save_dir = Path(save_dir)
    with open(save_dir / "quanto_qmap.json", "w") as f:
        json.dump(qmap, f, indent=2)
    logger.info(f"Wrote quanto_qmap.json ({len(qmap)} modules) to {save_dir}")

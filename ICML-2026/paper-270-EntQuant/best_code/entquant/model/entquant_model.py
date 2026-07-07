"""EntQuantModel: main user-facing API for EntQuant.

Wraps a PreTrainedModel with block-streaming quantization, ANS compression,
and standard checkpoint saving. Inherits from nn.Module + PushToHubMixin
(PeftModel pattern) so save_pretrained / push_to_hub work correctly -
including for compressed models where weights live in ANS buffers.

All three scenarios (quantize+save, load+compress, quantize+compress)
flow through _stream_build().
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch
from accelerate import dispatch_model, init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from optimum.quanto import Optimizer, QLinear, qtype
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils import PushToHubMixin

from ..compression.backends import CompressionBackend
from ..compression.compressor import BlockCompressor
from ..quantization.quantizer import load_quantized_block, quantize_block
from ..utils import clear_cache, DeviceMap, get_memory_stats, str_to_dtype, str_to_qtype
from .streaming import (
    group_keys_by_block,
    group_parameters_by_block,
    open_safetensors_shards,
    resolve_block_devices,
    resolve_non_block_device,
    save_block_shard,
    save_non_block_shard,
    write_index_and_config,
    write_quanto_qmap,
)
from .utils import non_persistent_buffer_names, resolve_model_path

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = "0.1"


class EntQuantModel(PushToHubMixin, nn.Module):
    """User-facing wrapper for EntQuant models.

    Inherits from nn.Module + PushToHubMixin (PeftModel pattern).
    Stores the inner PreTrainedModel as ``self.model`` (registered
    submodule) and delegates attribute access for backward compatibility.

    Args:
        model: The wrapped PreTrainedModel.
        compressor: Optional BlockCompressor managing ANS compression.
        weight_qtype: Quantization type (qtype or string name).
        activation_qtype: Activation quantization type for W8A8
            (e.g., "qfloat8"). None disables activation quantization.
        block_pattern: fnmatch pattern for block modules (metadata for save).
        dtype: Dtype used for non-quantized weights (stored in checkpoint).
    """

    def __init__(
        self,
        model: nn.Module,
        compressor: BlockCompressor | None = None,
        weight_qtype: qtype | str = "qfloat8",
        activation_qtype: qtype | str | None = None,
        block_pattern: str = "model.layers.*",
        dtype: torch.dtype = torch.bfloat16,
    ):
        nn.Module.__init__(self)
        self.model = model  # registered in _modules
        self._compressor = compressor
        self._weight_qtype = str_to_qtype(weight_qtype)
        self._activation_qtype = str_to_qtype(activation_qtype) if activation_qtype else None
        self._block_pattern = block_pattern
        self._dtype = dtype

    def __getattr__(self, name: str):
        """Delegate attribute access to the inner model for backward compat.

        Accesses __dict__["_modules"] directly to avoid recursion.
        Enables transparent access to model.config, model.device,
        model.generation_config, etc.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        model = self.__dict__.get("_modules", {}).get("model")
        if model is not None:
            return getattr(model, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def forward(self, *args, **kwargs):
        """Delegate forward pass to the inner model."""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Delegate generation to the inner model."""
        return self.model.generate(*args, **kwargs)

    @property
    def compressor(self) -> BlockCompressor | None:
        """Access the BlockCompressor (if compression is active)."""
        return self._compressor

    def compression_stats(self) -> dict[str, Any]:
        """Compute compression statistics for the model.

        Returns a dict with per-block and aggregate stats. Three size
        categories: "original" (base dtype, e.g. bf16), "quantized"
        (qtype, e.g. fp8, before compression), "compressed" (after ANS).
        """
        if self._compressor is None:
            raise RuntimeError("No compressor attached - model was loaded without compression.")

        compressor = self._compressor

        # Per-block stats (quantized = before ANS, compressed = after)
        per_block: dict[str, dict[str, float]] = {}
        injected_quantized = 0
        injected_compressed = 0

        for block_name, block in compressor.blocks.items():
            quant = block.original_size
            comp = block.compressed_size
            ratio = quant / comp if comp > 0 else 0.0
            per_block[block_name] = {
                "quantized_bytes": quant,
                "compressed_bytes": comp,
                "ratio": ratio,
                "entropy": block.entropy,
            }
            injected_quantized += quant
            injected_compressed += comp

        injected_ratio = injected_quantized / injected_compressed if injected_compressed > 0 else 0.0

        # Weighted average entropy (weighted by quantized block size)
        avg_entropy = (
            sum(s["entropy"] * s["quantized_bytes"] for s in per_block.values()) / injected_quantized
            if injected_quantized > 0
            else 0.0
        )

        # Non-compressed params + buffers (reconstruct full names
        # from block name + relative weight name)
        injected_names = {
            f"{block_name}.{rel_name}" for block_name, block in compressor.blocks.items() for rel_name in block.weights
        }
        other_bytes = 0
        for name, p in self.model.named_parameters():
            if name not in injected_names:
                other_bytes += p.numel() * p.element_size()
        for name, b in self.model.named_buffers():
            if name not in injected_names and not name.endswith("._compressed_weights"):
                other_bytes += b.numel() * b.element_size()

        full_quantized = injected_quantized + other_bytes
        full_compressed = injected_compressed + other_bytes
        full_ratio = full_quantized / full_compressed if full_compressed > 0 else 0.0

        # Original (base-dtype) sizes: use logical qtype element size
        # (not storage dtype, which may be packed, e.g. Marlin int32).
        qtype_elem_size = torch.tensor([], dtype=self._weight_qtype.dtype).element_size()
        num_logical_params = injected_quantized // qtype_elem_size

        original_dtype_raw = getattr(self.model.config, "dtype", None) or "bfloat16"
        original_dt = str_to_dtype(original_dtype_raw)
        original_dtype_name = str(original_dt).replace("torch.", "")
        original_elem_size = torch.tensor([], dtype=original_dt).element_size()
        injected_original = num_logical_params * original_elem_size

        injected_vs_original_ratio = injected_original / injected_compressed if injected_compressed > 0 else 0.0

        # Full model compared to original dtype
        full_original = injected_original + other_bytes
        full_vs_original_ratio = full_original / full_compressed if full_compressed > 0 else 0.0

        return {
            "per_block": per_block,
            "injected_quantized_bytes": injected_quantized,
            "injected_compressed_bytes": injected_compressed,
            "injected_ratio": injected_ratio,
            "avg_entropy": avg_entropy,
            "other_bytes": other_bytes,
            "full_quantized_bytes": full_quantized,
            "full_compressed_bytes": full_compressed,
            "full_ratio": full_ratio,
            "original_dtype": original_dtype_name,
            "injected_original_bytes": injected_original,
            "injected_vs_original_ratio": injected_vs_original_ratio,
            "full_original_bytes": full_original,
            "full_vs_original_ratio": full_vs_original_ratio,
        }

    def save_pretrained(
        self,
        save_directory: str | Path,
        **kwargs,
    ) -> None:
        """Save model as a standard safetensors checkpoint.

        For compressed models, decompresses each block into the shared
        buffer before saving its shard. Writes config.json,
        model.safetensors.index.json, and quanto_qmap.json.

        Args:
            save_directory: Output directory.
            **kwargs: Accepted for PushToHubMixin compatibility
                (max_shard_size, safe_serialization, etc.).
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        model = self.model
        block_groups, non_block_keys = group_parameters_by_block(model, self._block_pattern)

        # Verify full coverage: block_groups + non_block_keys must span
        # all saveable keys (params + persistent buffers, excluding
        # _compressed_weights and non-persistent buffers).
        grouped_keys = set(non_block_keys)
        for keys in block_groups.values():
            grouped_keys.update(keys)
        non_persistent = non_persistent_buffer_names(model)
        saveable_keys = {n for n, _ in model.named_parameters()} | {
            n for n, _ in model.named_buffers() if not n.endswith("._compressed_weights") and n not in non_persistent
        }
        missing = saveable_keys - grouped_keys
        if missing:
            raise ValueError(
                f"Key coverage check failed - {len(missing)} keys not "
                f"grouped by block_pattern '{self._block_pattern}': "
                f"{sorted(missing)[:10]}"
            )

        total_shards = 1 + len(block_groups)
        weight_map: dict[str, str] = {}

        # Save non-block weights (embeddings, norms, lm_head, etc.)
        shard_map = save_non_block_shard(model, non_block_keys, save_dir, total_shards)
        weight_map.update(shard_map)

        # Save each block
        for shard_idx, block_name in enumerate(sorted(block_groups), start=2):
            block_module = model.get_submodule(block_name)

            # Decompress into buffer so state_dict reads actual weights
            if self._compressor is not None and block_name in self._compressor.blocks:
                self._compressor.decompress(block_name)

            shard_map = save_block_shard(
                block_module,
                block_name,
                save_dir,
                shard_idx,
                total_shards,
            )
            weight_map.update(shard_map)

        # Build qmap from QLinear modules
        qmap: dict[str, dict[str, str]] = {}
        for name, m in model.named_modules():
            if isinstance(m, QLinear):
                act_name = m.activation_qtype.name if m.activation_qtype is not None else "none"
                qmap[name] = {
                    "weights": m.weight_qtype.name,
                    "activations": act_name,
                }

        # Write metadata
        config = model.config
        quant_config: dict[str, Any] = {
            "quant_method": "entquant",
            "weight_qtype": self._weight_qtype.name,
            "dtype": str(self._dtype).replace("torch.", ""),
            "block_pattern": self._block_pattern,
            "entquant_version": CHECKPOINT_VERSION,
        }
        if self._activation_qtype is not None:
            quant_config["activation_qtype"] = self._activation_qtype.name
        write_index_and_config(save_dir, config, weight_map, quant_config)
        write_quanto_qmap(save_dir, qmap)

        logger.info(f"Saved checkpoint to {save_dir}")

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | None = None,
        *,
        model: nn.Module | None = None,
        quantize: bool = False,
        compress: bool = True,
        save_dir: str | None = None,
        block_pattern: str = "model.layers.*",
        device_map: DeviceMap = "cuda",
        weight_qtype: str = "qfloat8",
        activation_qtype: str | None = None,
        fallback_layers: set[str] | None = None,
        include: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
        optimizer: Optimizer | None = None,
        optimizer_fallback: Optimizer | None = None,
        backend: CompressionBackend | None = None,
        dtype: torch.dtype | None = None,
        model_cls: type | None = None,
        model_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> "EntQuantModel":
        """Load or build an EntQuant model with block-streaming.

        Auto-detects whether model_id is a pre-quantized checkpoint
        (has quantization_config in config.json) or a base model
        (requires quantize=True).

        Args:
            model_id: HuggingFace model ID or local path. Optional
                when ``model`` is provided.
            model: Pre-initialized model with weights on device.
                When provided, ``quantize`` must be True.
            quantize: Whether to quantize from scratch.
            compress: Whether to ANS-compress weights.
            save_dir: Optional directory to save checkpoint during build.
            block_pattern: fnmatch pattern for block modules.
            device_map: A single device string (e.g. ``"cuda"``,
                ``"cpu"``, ``"cuda:1"``), ``"auto"``, an explicit
                dict, or None. Defaults to ``"cuda"``.
            weight_qtype: Quantization type name (e.g., "qfloat8").
            activation_qtype: Activation quantization type for W8A8
                (e.g., "qfloat8"). None disables activation quantization.
            fallback_layers: Layer names that use the fallback optimizer
                (e.g. layers containing super weights).
            include: fnmatch include patterns for quantization.
            exclude: fnmatch exclude patterns for quantization.
            optimizer: Optimizer for normal layers.
            optimizer_fallback: Optimizer for fallback layers.
            backend: Compression backend (defaults to nvCOMPBackend).
            dtype: Dtype for non-quantized weights. Inferred from
                checkpoint for pre-quantized models; defaults to
                bfloat16 when quantizing from scratch.
            model_cls: Model class override (default: AutoModelForCausalLM).
            model_kwargs: Extra kwargs for model instantiation.
            config_kwargs: Extra kwargs for AutoConfig.from_pretrained
                (e.g. ``trust_remote_code``).
            **kwargs: Additional kwargs.

        Returns:
            EntQuantModel wrapping the built model.
        """
        # BYOM path: pre-initialized model provided directly
        if model is not None:
            if not quantize:
                raise ValueError("quantize=True is required when passing a pre-initialized model.")
            if dtype is None:
                dtype = torch.bfloat16
            return cls._build_from_model(
                model,
                compress=compress,
                save_dir=save_dir,
                block_pattern=block_pattern,
                weight_qtype=weight_qtype,
                activation_qtype=activation_qtype,
                fallback_layers=fallback_layers,
                include=include,
                exclude=exclude,
                optimizer=optimizer,
                optimizer_fallback=optimizer_fallback,
                backend=backend,
                dtype=dtype,
                **kwargs,
            )

        if model_id is None:
            raise ValueError("Either model_id or model must be provided.")

        config = AutoConfig.from_pretrained(model_id, **(config_kwargs or {}))
        is_prequantized = hasattr(config, "quantization_config")
        if is_prequantized and not quantize:
            qconfig = config.quantization_config
            if isinstance(qconfig, dict):
                weight_qtype = qconfig.get("weight_qtype", weight_qtype)
                activation_qtype = qconfig.get("activation_qtype", activation_qtype)
                if dtype is None:
                    saved_dtype = qconfig.get("dtype")
                    dtype = str_to_dtype(saved_dtype) if saved_dtype else torch.bfloat16
            else:
                weight_qtype = getattr(qconfig, "weight_qtype", weight_qtype)
                activation_qtype = getattr(qconfig, "activation_qtype", activation_qtype)
                if dtype is None:
                    saved_dtype = getattr(qconfig, "dtype", None)
                    dtype = str_to_dtype(saved_dtype) if saved_dtype else torch.bfloat16
            logger.info(f"Detected pre-quantized checkpoint: {weight_qtype}, dtype={dtype}")
        elif not quantize:
            raise ValueError(f"Model {model_id} is not pre-quantized. Pass quantize=True to quantize from scratch.")

        if dtype is None:
            dtype = torch.bfloat16

        if compress and device_map == "auto":
            raise NotImplementedError(
                "device_map='auto' is not supported with compression. "
                "infer_auto_device_map is not aware of ANS compression "
                "memory savings. Use an explicit device map dict or "
                "device_map=None for single GPU."
            )

        return cls._stream_build(
            model_id,
            quantize=quantize,
            compress=compress,
            save_dir=save_dir,
            block_pattern=block_pattern,
            device_map=device_map,
            weight_qtype=weight_qtype,
            activation_qtype=activation_qtype,
            fallback_layers=fallback_layers,
            include=include,
            exclude=exclude,
            optimizer=optimizer,
            optimizer_fallback=optimizer_fallback,
            backend=backend,
            dtype=dtype,
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            config_kwargs=config_kwargs,
            **kwargs,
        )

    @classmethod
    def convert(
        cls,
        model_id: str,
        output_dir: str,
        *,
        block_pattern: str = "model.layers.*",
        weight_qtype: str = "qfloat8",
        activation_qtype: str | None = None,
        fallback_layers: set[str] | None = None,
        include: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
        optimizer: Optimizer | None = None,
        optimizer_fallback: Optimizer | None = None,
        dtype: torch.dtype = torch.bfloat16,
        model_cls: type | None = None,
        model_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Quantize a model and save as a standard safetensors checkpoint.

        Streaming disk-to-disk quantization. No ANS compression is
        applied; the output is a standard quanto checkpoint loadable by
        QuantizedModelForCausalLM.from_pretrained().

        Args:
            model_id: HuggingFace model ID or local path.
            output_dir: Directory to write the checkpoint.
            block_pattern: fnmatch pattern for block modules.
            weight_qtype: Quantization type name (e.g., "qfloat8").
            activation_qtype: Activation quantization type for W8A8
                (e.g., "qfloat8"). None disables activation quantization.
            fallback_layers: Layer names that use the fallback optimizer.
            include: fnmatch include patterns for quantization.
            exclude: fnmatch exclude patterns for quantization.
            optimizer: Optimizer for normal layers.
            optimizer_fallback: Optimizer for fallback layers.
            dtype: Model dtype for loading bf16 weights.
            model_cls: Model class override (default: AutoModelForCausalLM).
            model_kwargs: Extra kwargs for model instantiation.
            config_kwargs: Extra kwargs for AutoConfig.from_pretrained.
            **kwargs: Additional kwargs.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cls._stream_build(
            model_id,
            quantize=True,
            compress=False,
            save_dir=output_dir,
            block_pattern=block_pattern,
            device_map=None,
            weight_qtype=weight_qtype,
            activation_qtype=activation_qtype,
            fallback_layers=fallback_layers,
            include=include,
            exclude=exclude,
            optimizer=optimizer,
            optimizer_fallback=optimizer_fallback,
            backend=None,
            dtype=dtype,
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            config_kwargs=config_kwargs,
            **kwargs,
        )
        logger.info(f"Conversion complete: {output_dir}")

    @classmethod
    def _build_from_model(
        cls,
        model: nn.Module,
        *,
        compress: bool,
        save_dir: str | None,
        block_pattern: str,
        weight_qtype: qtype | str = "qfloat8",
        activation_qtype: qtype | str | None = None,
        fallback_layers: set[str] | None = None,
        include: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
        optimizer: Optimizer | None = None,
        optimizer_fallback: Optimizer | None = None,
        backend: CompressionBackend | None = None,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "EntQuantModel":
        """Build from a pre-initialized model (BYOM path).

        Quantizes an already-loaded model in-place, block by block.
        Optionally saves checkpoint and/or applies ANS compression.

        Args:
            model: Pre-initialized model with weights on device.
            compress: Whether to ANS-compress weights.
            save_dir: Optional directory to save checkpoint.
            block_pattern: fnmatch pattern for block modules.
            weight_qtype: Quantization type.
            activation_qtype: Activation quantization type for W8A8.
            fallback_layers: Layers using the fallback optimizer.
            include: fnmatch include patterns for quantization.
            exclude: fnmatch exclude patterns for quantization.
            optimizer: Optimizer for normal layers.
            optimizer_fallback: Optimizer for fallback layers.
            backend: Compression backend instance.
            dtype: Dtype for non-quantized weights.
            **kwargs: Additional kwargs.

        Returns:
            EntQuantModel wrapping the quantized model.
        """
        weight_qtype = str_to_qtype(weight_qtype)
        activation_qtype = str_to_qtype(activation_qtype) if activation_qtype else None
        model.eval()

        # 1. Group parameters by block pattern
        block_groups, non_block_keys = group_parameters_by_block(model, block_pattern)

        # 2. Infer block device map from actual parameter placement
        block_device_map: dict[str, torch.device] = {}
        for block_name in block_groups:
            block_module = model.get_submodule(block_name)
            param = next(block_module.parameters(), None)
            if param is None:
                raise ValueError(f"Block {block_name} has no parameters")
            block_device_map[block_name] = param.device

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        total_shards = 1 + len(block_groups)
        weight_map: dict[str, str] = {}

        # 3. Init compressor before quantization (needs nn.Linear)
        compressor = None
        if compress:
            compressor = BlockCompressor(backend=backend)
            compressor.update_buffer_sizes(model, block_device_map, weight_qtype.dtype)

        qmap_out: dict[str, dict[str, str]] = {}

        # 4. Process blocks one at a time
        for shard_idx, block_name in enumerate(sorted(block_groups), start=2):
            block_module = model.get_submodule(block_name)

            quantize_block(
                model,
                block_module,
                block_name,
                weight_qtype,
                activation_qtype=activation_qtype,
                include=include,
                exclude=exclude,
                fallback_layers=fallback_layers,
                optimizer=optimizer,
                optimizer_fallback=optimizer_fallback,
            )

            for name, m in block_module.named_modules():
                if isinstance(m, QLinear):
                    full_name = f"{block_name}.{name}" if name else block_name
                    act_name = m.activation_qtype.name if m.activation_qtype is not None else "none"
                    qmap_out[full_name] = {
                        "weights": m.weight_qtype.name,
                        "activations": act_name,
                    }

            if save_dir:
                shard_map = save_block_shard(
                    block_module,
                    block_name,
                    save_dir,
                    shard_idx,
                    total_shards,
                )
                weight_map.update(shard_map)

            if compress and compressor is not None:
                compressor.compress_block(block_module, block_name)
                compressor.register_block(block_name)

            logger.info(f"Processed block {block_name}")

        # 5. Save non-block shard and metadata
        if save_dir:
            shard_map = save_non_block_shard(model, non_block_keys, save_dir, total_shards)
            weight_map.update(shard_map)

            config = model.config
            quant_config: dict[str, Any] = {
                "quant_method": "entquant",
                "weight_qtype": weight_qtype.name,
                "dtype": str(dtype).replace("torch.", ""),
                "block_pattern": block_pattern,
                "entquant_version": CHECKPOINT_VERSION,
            }
            if activation_qtype is not None:
                quant_config["activation_qtype"] = activation_qtype.name
            write_index_and_config(save_dir, config, weight_map, quant_config)
            write_quanto_qmap(save_dir, qmap_out)

        # 6. Multi-GPU dispatch if needed
        unique_devices = {d for d in block_device_map.values() if d.type != "cpu"}
        if len(unique_devices) > 1:
            full_device_map = {name: str(dev) for name, dev in block_device_map.items()}
            params = dict(model.named_parameters())
            buffers = dict(model.named_buffers())
            for key in non_block_keys:
                module_name = key.rsplit(".", 1)[0] if "." in key else key
                if module_name not in full_device_map:
                    t = params.get(key) or buffers.get(key)
                    if t is not None:
                        full_device_map[module_name] = str(t.device)
            dispatch_model(model, full_device_map)

        clear_cache()
        get_memory_stats("Memory after BYOM build")

        return cls(
            model,
            compressor=compressor,
            weight_qtype=weight_qtype,
            activation_qtype=activation_qtype,
            block_pattern=block_pattern,
            dtype=dtype,
        )

    @classmethod
    def _stream_build(
        cls,
        model_id: str,
        *,
        quantize: bool,
        compress: bool,
        save_dir: str | None,
        block_pattern: str,
        device_map: DeviceMap,
        weight_qtype: qtype | str = "qfloat8",
        activation_qtype: qtype | str | None = None,
        fallback_layers: set[str] | None = None,
        include: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
        optimizer: Optimizer | None = None,
        optimizer_fallback: Optimizer | None = None,
        backend: CompressionBackend | None = None,
        dtype: torch.dtype = torch.bfloat16,
        model_cls: type | None = None,
        model_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> "EntQuantModel":
        """Core block-streaming pipeline.

        All three scenarios flow through here. Processes one block at
        a time: load -> [quantize] -> [save] -> [compress + register hook].

        Args:
            model_id: HuggingFace model ID or local path.
            quantize: Whether to quantize from scratch.
            compress: Whether to ANS-compress weights.
            save_dir: Optional directory to save checkpoint.
            block_pattern: fnmatch pattern for block modules.
            device_map: Device placement strategy.
            weight_qtype: Quantization type.
            activation_qtype: Activation quantization type for W8A8.
            fallback_layers: Layers using the fallback optimizer.
            include: fnmatch include patterns for quantization.
            exclude: fnmatch exclude patterns for quantization.
            optimizer: Optimizer for normal layers.
            optimizer_fallback: Optimizer for fallback layers.
            backend: Compression backend instance.
            dtype: Dtype for non-quantized weights.
            model_cls: Model class override.
            model_kwargs: Extra kwargs for model instantiation.
            config_kwargs: Extra kwargs for AutoConfig.from_pretrained.
            **kwargs: Additional kwargs.

        Returns:
            EntQuantModel wrapping the built model.
        """
        weight_qtype = str_to_qtype(weight_qtype)
        activation_qtype = str_to_qtype(activation_qtype) if activation_qtype else None

        _ckw = config_kwargs or {}
        model_path = resolve_model_path(model_id, **_ckw)
        config = AutoConfig.from_pretrained(model_id, **_ckw)

        model_cls = model_cls or AutoModelForCausalLM
        model_kwargs = model_kwargs or {}

        # 1. Create empty model skeleton on meta device
        with init_empty_weights():
            model = model_cls.from_config(config, **model_kwargs)
        model.eval()

        # 2. Group safetensors keys by block pattern
        block_groups, non_block_keys = group_keys_by_block(model_path, block_pattern)

        # Filter out checkpoint keys absent from the model skeleton.
        # Handles architecture changes (e.g. rotary_emb removed in newer
        # transformers). Also excludes non-persistent buffers (e.g., rotary
        # inv_freq) which are recomputed from config and must keep their
        # original precision.
        non_persistent = non_persistent_buffer_names(model)
        model_keys = {n for n, _ in model.named_parameters()} | {
            n for n, _ in model.named_buffers() if n not in non_persistent
        }
        for block_name in block_groups:
            block_groups[block_name] = [k for k in block_groups[block_name] if k in model_keys]
        non_block_keys = [k for k in non_block_keys if k in model_keys]

        # 3. Resolve device placement
        if compress and device_map == "auto":
            raise NotImplementedError(
                "device_map='auto' is not supported with compression. "
                "infer_auto_device_map is not aware of ANS compression "
                "memory savings. Use an explicit device map dict or "
                "device_map=None for single GPU."
            )
        block_device_map = resolve_block_devices(model, block_groups, device_map, weight_qtype.dtype)

        # 4. Load checkpoint metadata
        qmap = None
        if not quantize:
            qmap_path = model_path / "quanto_qmap.json"
            if qmap_path.exists():
                with open(qmap_path) as f:
                    qmap = json.load(f)
            else:
                raise FileNotFoundError(
                    f"quanto_qmap.json not found in {model_id}. Required for pre-quantized checkpoints."
                )

            # Validate block_pattern against checkpoint metadata
            qconfig = getattr(config, "quantization_config", None)
            if qconfig is not None:
                saved_bp = (
                    qconfig.get("block_pattern")
                    if isinstance(qconfig, dict)
                    else getattr(qconfig, "block_pattern", None)
                )
                if saved_bp and saved_bp != block_pattern:
                    logger.warning(
                        f"block_pattern mismatch: checkpoint has "
                        f"'{saved_bp}', but '{block_pattern}' was "
                        f"requested. Using '{block_pattern}'."
                    )

        shard_reader = open_safetensors_shards(model_path)

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        total_shards = 1 + len(block_groups)
        weight_map: dict[str, str] = {}

        # 5. Load non-block weights (embed_tokens, lm_head, norm)
        for key in non_block_keys:
            tensor = shard_reader.get_tensor(key)
            target = resolve_non_block_device(key, block_device_map, device_map)
            set_module_tensor_to_device(model, key, target, value=tensor, dtype=dtype)

        if save_dir:
            shard_map = save_non_block_shard(model, non_block_keys, save_dir, total_shards)
            weight_map.update(shard_map)

        # Initialize compressor with pre-computed buffer sizes
        compressor = None
        if compress:
            compressor = BlockCompressor(backend=backend)
            compressor.update_buffer_sizes(model, block_device_map, weight_qtype.dtype)

        qmap_out: dict[str, dict[str, str]] = {}

        # 6. Process blocks one at a time
        for shard_idx, (block_name, keys) in enumerate(sorted(block_groups.items()), start=2):
            block_module = model.get_submodule(block_name)
            target_device = block_device_map[block_name]

            if quantize:
                for key in keys:
                    tensor = shard_reader.get_tensor(key)
                    set_module_tensor_to_device(model, key, target_device, value=tensor, dtype=dtype)
                quantize_block(
                    model,
                    block_module,
                    block_name,
                    weight_qtype,
                    activation_qtype=activation_qtype,
                    include=include,
                    exclude=exclude,
                    fallback_layers=fallback_layers,
                    optimizer=optimizer,
                    optimizer_fallback=optimizer_fallback,
                )
            else:
                if qmap is None:
                    raise RuntimeError("qmap is unexpectedly None for a pre-quantized checkpoint. This is a bug.")
                load_quantized_block(
                    model,
                    block_name,
                    keys,
                    shard_reader,
                    qmap,
                    device=target_device,
                    dtype=dtype,
                )

            # Track which modules were quantized
            for name, m in block_module.named_modules():
                if isinstance(m, QLinear):
                    full_name = f"{block_name}.{name}" if name else block_name
                    act_name = m.activation_qtype.name if m.activation_qtype is not None else "none"
                    qmap_out[full_name] = {
                        "weights": m.weight_qtype.name,
                        "activations": act_name,
                    }

            if save_dir:
                shard_map = save_block_shard(
                    block_module,
                    block_name,
                    save_dir,
                    shard_idx,
                    total_shards,
                )
                weight_map.update(shard_map)

            # ANS compress and register hook + buffer pointers
            if compress and compressor is not None:
                compressor.compress_block(block_module, block_name)
                compressor.register_block(block_name)

            logger.info(f"Processed block {block_name}")

        shard_reader.close()

        # 7. Multi-GPU: dispatch_model for activation transfer hooks.
        unique_devices = {d for d in block_device_map.values() if d.type != "cpu"}
        if len(unique_devices) > 1:
            full_device_map = {name: str(dev) for name, dev in block_device_map.items()}
            # Include non-block modules so dispatch_model covers all parameters
            for key in non_block_keys:
                module_name = key.rsplit(".", 1)[0] if "." in key else key
                if module_name not in full_device_map:
                    dev = resolve_non_block_device(key, block_device_map, device_map)
                    full_device_map[module_name] = str(dev)
            dispatch_model(model, full_device_map)

        clear_cache()

        # 8. Write checkpoint metadata
        if save_dir:
            quant_config: dict[str, Any] = {
                "quant_method": "entquant",
                "weight_qtype": weight_qtype.name,
                "dtype": str(dtype).replace("torch.", ""),
                "block_pattern": block_pattern,
                "entquant_version": CHECKPOINT_VERSION,
            }
            if activation_qtype is not None:
                quant_config["activation_qtype"] = activation_qtype.name
            write_index_and_config(save_dir, config, weight_map, quant_config)
            write_quanto_qmap(save_dir, qmap_out)

        get_memory_stats("Memory after build")

        return cls(
            model,
            compressor=compressor,
            weight_qtype=weight_qtype,
            activation_qtype=activation_qtype,
            block_pattern=block_pattern,
            dtype=dtype,
        )

"""Per-block quantization and loading using optimum-quanto.

Provides :func:`quantize_block` for streaming quantization of a single
transformer block, and :func:`load_quantized_block` for reconstructing
a pre-quantized block from safetensors shards.
"""

from __future__ import annotations

import logging
from fnmatch import fnmatch

import torch
from accelerate.utils import set_module_tensor_to_device
from optimum.quanto import freeze, Optimizer, QLinear, qtype, quantize, WeightQBytesTensor
from optimum.quanto.tensor.weights.marlin import MarlinF8QBytesTensor
from torch import nn

from ..model.streaming import ShardReader
from ..utils import set_module_by_name, str_to_qtype
from .optimizer import SymmetricEntropyOptimizer, WrappedAbsmaxOptimizer
from .tensor import rebuild_tensors, resolve_signed_zeros, to_marlin

logger = logging.getLogger(__name__)

__all__ = ["quantize_block", "load_quantized_block"]


def quantize_block(
    model: nn.Module,
    block_module: nn.Module,
    block_name: str,
    weight_qtype: qtype | str,
    activation_qtype: qtype | str | None = None,
    include: str | list[str] | None = None,
    exclude: str | list[str] | None = None,
    fallback_layers: set[str] | None = None,
    optimizer: Optimizer | None = None,
    optimizer_fallback: Optimizer | None = None,
) -> None:
    """Quantize all linear layers in a block, per-linear.

    Different linears may use different optimizers (entropy-aware LBFGS
    for normal layers, a simpler optimizer for fallback layers).

    Iteration order follows named_modules() DFS to match the original
    pipeline for bit-identical results.

    Args:
        model: The full model (needed by quanto's quantize()).
        block_module: The block module to quantize.
        block_name: Full dotted name of the block in the model.
        weight_qtype: Quantization type (e.g., "qfloat8").
        activation_qtype: Activation quantization type for W8A8
            (e.g., "qfloat8"). None disables activation quantization.
        include: fnmatch include patterns for module names.
        exclude: fnmatch exclude patterns for module names.
        fallback_layers: Module names that should use the fallback
            optimizer (e.g. layers containing super weights).
        optimizer: Optimizer for normal layers.
        optimizer_fallback: Optimizer for fallback layers.
    """
    weight_qtype = str_to_qtype(weight_qtype)
    activation_qtype = str_to_qtype(activation_qtype) if activation_qtype else None

    if optimizer is None:
        optimizer = SymmetricEntropyOptimizer()
    if optimizer_fallback is None:
        optimizer_fallback = WrappedAbsmaxOptimizer()

    fb = fallback_layers or set()

    for name, module in block_module.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        full_name = f"{block_name}.{name}" if name else block_name

        # Apply include/exclude filtering
        if include is not None:
            patterns = [include] if isinstance(include, str) else include
            if not any(fnmatch(full_name, p) for p in patterns):
                continue
        if exclude is not None:
            patterns = [exclude] if isinstance(exclude, str) else exclude
            if any(fnmatch(full_name, p) for p in patterns):
                continue

        # Select optimizer based on fallback layer membership
        optimizer = optimizer_fallback if full_name in fb else optimizer

        quantize(
            model,
            weights=weight_qtype,
            activations=activation_qtype,
            include=f"*{full_name}*",
            optimizer=optimizer,
        )
        freeze(model)
        logger.info(f"Quantized {full_name}")

    # Resolve signed zeros for floating-point qtypes
    if weight_qtype.is_floating_point:
        for _, module in block_module.named_modules():
            if isinstance(module, QLinear):
                module.weight = resolve_signed_zeros(module.weight)
        logger.debug(f"Resolved signed zeros in {block_name}")

    # Marlin conversion for CUDA float8
    rebuild_tensors(block_module)


def load_quantized_block(
    model: nn.Module,
    block_name: str,
    keys: list[str],
    shard_reader: ShardReader,
    qmap: dict[str, dict[str, str]],
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
) -> None:
    """Load a pre-quantized block from safetensors.

    For each quantized module: reads _data + _scale, constructs
    WeightQBytesTensor, replaces nn.Linear with QLinear.
    Non-quantized params (layernorm, bias) are loaded directly.

    Args:
        model: The full model (meta-device skeleton).
        block_name: Full dotted name of the block.
        keys: Safetensors keys belonging to this block.
        shard_reader: ShardReader for tensor access.
        qmap: Module name -> {"weights": qtype_name} mapping.
        device: Target device for loaded tensors.
        dtype: Target dtype for non-quantized params (layernorm, etc.).
    """
    quantized_prefixes = {name for name in qmap if name.startswith(block_name + ".")}
    consumed_keys: set[str] = set()

    for full_name in sorted(quantized_prefixes):
        weight_qtype = str_to_qtype(qmap[full_name]["weights"])
        act_str = qmap[full_name].get("activations", "none")
        activation_qtype = str_to_qtype(act_str) if act_str != "none" else None
        linear = model.get_submodule(full_name)  # nn.Linear on meta

        # Read raw quantized components
        data_key = f"{full_name}.weight._data"
        scale_key = f"{full_name}.weight._scale"
        data = shard_reader.get_tensor(data_key).to(device)
        scale = shard_reader.get_tensor(scale_key).to(device)
        consumed_keys.add(data_key)
        consumed_keys.add(scale_key)

        # Construct quanto tensor and optimize for Marlin
        wqt = WeightQBytesTensor(
            qtype=weight_qtype,
            axis=0,
            size=linear.weight.shape,
            stride=linear.weight.stride(),
            data=data,
            scale=scale,
            activation_qtype=activation_qtype,
        )
        optimized = to_marlin(wqt)
        if optimized is wqt:
            optimized = wqt.optimize()
        if isinstance(optimized, MarlinF8QBytesTensor):
            logger.debug(f"Loaded {full_name} as MarlinF8QBytesTensor")
        wqt = optimized

        # Create QLinear and replace nn.Linear in the model tree.
        # Pass dtype so activation scale buffers match the model's
        # compute dtype (avoids float32 vs bf16 mismatch at lm_head).
        # quantize_input=True mirrors quanto's qcreate() - registers
        # both input and output activation hooks for W8A8.
        qlinear = QLinear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            dtype=dtype,
            weights=weight_qtype,
            activations=activation_qtype,
            quantize_input=activation_qtype is not None,
            device=device,
        )
        qlinear.weight = nn.Parameter(wqt, requires_grad=False)

        # Copy bias if present
        bias_key = f"{full_name}.bias"
        if linear.bias is not None and bias_key in keys:
            bias = shard_reader.get_tensor(bias_key)
            if dtype is not None:
                bias = bias.to(dtype)
            qlinear.bias = nn.Parameter(bias.to(device), requires_grad=False)
            consumed_keys.add(bias_key)

        set_module_by_name(model, full_name, qlinear)
        logger.debug(f"Loaded quantized module {full_name}")

    # Load non-quantized params (layernorm weights, biases, etc.)
    for key in keys:
        if key not in consumed_keys:
            tensor = shard_reader.get_tensor(key)
            set_module_tensor_to_device(model, key, device, value=tensor, dtype=dtype)

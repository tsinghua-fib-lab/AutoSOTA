"""Quanto tensor helpers for data extraction and signed-zero handling."""

import logging

import torch
from optimum.quanto import QBytesTensor, WeightQBytesTensor
from optimum.quanto.tensor.packed import PackedTensor
from optimum.quanto.tensor.weights.marlin import MarlinF8PackedTensor, MarlinF8QBytesTensor
from torch import nn, Tensor

logger = logging.getLogger(__name__)


def get_tensor_data(tensor: Tensor) -> Tensor:
    """Return the underlying data of a (quantized) tensor.

    Returns a raw pointer to the storage, NOT a copy. This is required
    for compression and decompression buffer management.
    """
    if not isinstance(tensor, QBytesTensor):
        return tensor

    data = tensor._data

    # Unpack packed tensors without copying - we may need to modify in place
    if isinstance(data, (MarlinF8PackedTensor, PackedTensor)):
        return data._data

    # float8_e4m3fn data from a CPU-moved Marlin tensor: pack on CUDA first
    if data.dtype == torch.float8_e4m3fn:
        return MarlinF8PackedTensor.pack(data.cuda())._data.to(tensor.device)

    return data


def to_marlin(tensor: WeightQBytesTensor) -> WeightQBytesTensor:
    """Convert a WeightQBytesTensor to MarlinF8QBytesTensor if possible.

    Directly constructs MarlinF8QBytesTensor for fp8 weights on CUDA,
    bypassing quanto's WeightQBytesTensor.create() which blocks Marlin
    when activation_qtype is set. This is safe because Marlin internally
    always sets activation_qtype=qfloat8_e4m3fn.
    """
    if isinstance(tensor, MarlinF8QBytesTensor):
        return tensor
    if tensor.device.type != "cuda":
        return tensor
    if isinstance(tensor._data, MarlinF8PackedTensor) or (tensor._data.dtype == torch.float8_e4m3fn):
        return MarlinF8QBytesTensor(
            tensor._qtype,
            tensor._axis,
            tensor.size(),
            tensor.stride(),
            tensor._data,
            tensor._scale,
        )
    return tensor


def rebuild_tensors(module: nn.Module) -> None:
    """Optimize all WeightQBytesTensor params/buffers in a model.

    Converts fp8 weights to MarlinF8QBytesTensor on CUDA for efficient
    inference via the fused gemm_f16f8_marlin kernel. Falls back to
    quanto's optimize() for non-fp8 qtypes.
    """
    for name, param in module.named_parameters():
        if isinstance(param, WeightQBytesTensor):
            optimized = to_marlin(param)
            if optimized is param:
                optimized = param.optimize()
            if optimized is not param:
                *path, attr = name.split(".")
                parent = module.get_submodule(".".join(path)) if path else module
                parent._parameters[attr] = optimized
                logger.debug(f"Optimized parameter {name}")

    for name, buf in module.named_buffers():
        if isinstance(buf, WeightQBytesTensor):
            optimized = to_marlin(buf)
            if optimized is buf:
                optimized = buf.optimize()
            if optimized is not buf:
                *path, attr = name.split(".")
                parent = module.get_submodule(".".join(path)) if path else module
                parent._buffers[attr] = optimized
                logger.debug(f"Optimized buffer {name}")


def resolve_signed_zeros(tensor: Tensor) -> Tensor:
    """Replace ``-0.0`` with ``+0.0`` in floating-point quantized tensors.

    Signed zeros cause non-deterministic entropy coding. Only applies
    to QBytesTensor with a floating-point qtype; other tensors are
    returned unchanged.

    Args:
        tensor: Possibly quantized tensor to clean up.

    Returns:
        Tensor with signed zeros resolved (in-place for the data).
    """
    if not isinstance(tensor, QBytesTensor) or not tensor.qtype.is_floating_point:
        return tensor

    if isinstance(tensor._data, MarlinF8PackedTensor):
        data = tensor._data.unpack()
        packed = True
    else:
        data = tensor._data
        packed = False

    # float casting because not all dtypes support these operations
    data_float = data.float()
    data_float[data_float == -0.0] = 0.0
    data = data_float.to(data.dtype)

    if packed:
        tensor._data = MarlinF8PackedTensor.pack(data.cuda()).to(tensor.device)
    else:
        tensor._data = data

    return tensor

from typing import Dict, Optional, Any, List, Tuple
import datetime
import os
import io
from numbers import Number
import torch
import torch.distributed as dist

__all__ = [
    "set_dist",
    "is_distributed_set",
    "get_rank",
    "get_world_size",
    "is_master",
    "barrier",
    "all_reduce_scalar",
    "all_reduce_tensor",
    "all_reduce_dict",
    "all_gather_tensor",
    "all_gather_dict",
    "broadcast_tensor",
    "broadcast_tensors",
    "broadcast_objects",
    "broadcast_any_tensor",
]


def set_dist(device_type: str = "cuda") -> Tuple[torch.device, int]:
    if device_type == "cpu":
        return torch.device("cpu"), 0
    if device_type != "cuda":
        raise ValueError("Distributed setting either support CPU or CUDA.")

    if os.environ.get("LOCAL_RANK", -1) == -1:  # not called by torchrun, do not initialize dist.
        return torch.device("cuda"), 0  # single GPU

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    local_rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)  # needed!
    print(f"Local Rank ({local_rank}) Initialized for DDP")
    return device, local_rank


def is_distributed_set() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed_set() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed_set() else 1


def is_master() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed_set():
        dist.barrier()


def all_reduce_scalar(value: Number, op: str = "sum") -> Number:
    """All-reduce single scalar value. NOT torch tensor."""
    if not is_distributed_set():
        return value

    op = op.lower()
    if (op == "sum") or (op == "mean"):
        dist_op = dist.ReduceOp.SUM
    elif op == "min":
        dist_op = dist.ReduceOp.MIN
    elif op == "max":
        dist_op = dist.ReduceOp.MAX
    elif op == "product":
        dist_op = dist.ReduceOp.PRODUCT
    else:
        raise RuntimeError(f"Invalid all_reduce_scalar op: {op}")

    backend = dist.get_backend()
    if backend == torch.distributed.Backend.NCCL:
        device = torch.device("cuda")
    elif backend == torch.distributed.Backend.GLOO:
        device = torch.device("cpu")
    else:
        raise RuntimeError(f"Unsupported distributed backend: {backend}")

    tensor = torch.tensor(value, device=device, requires_grad=False)
    dist.all_reduce(tensor, op=dist_op)
    if op == "mean":
        tensor /= get_world_size()
    ret = tensor.item()
    return ret


def all_reduce_tensor(tensor: torch.Tensor, op="sum", detach: bool = True) -> torch.Tensor:
    if not is_distributed_set():
        return tensor

    ret = tensor.clone()
    if detach:
        ret = ret.detach()
    if (op == "sum") or (op == "mean"):
        dist_op = dist.ReduceOp.SUM
    else:  # intentionally only support sum or mean
        raise RuntimeError(f"Invalid all_reduce_tensor op: {op}")

    dist.all_reduce(ret, op=dist_op)
    if op == "mean":
        ret /= get_world_size()
    return ret


def all_reduce_dict(result: Dict[str, Any], op="sum") -> Dict[str, Any]:
    # only accepts dictionary that key is string and value is either number or Tensor.
    new_result = {}
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            new_result[k] = all_reduce_tensor(v, op)
        elif isinstance(v, Number):
            new_result[k] = all_reduce_scalar(v, op)
        else:
            raise RuntimeError(f"Input dictionary for all_reduce_dict should only have "
                               f"either tensor or scalar as their values, got ({k}: {type(v)})")
    return new_result


def all_gather_tensor(tensor: torch.Tensor) -> List[torch.Tensor]:
    if not is_distributed_set():
        return [tensor]
    world_size = get_world_size()
    local_rank = get_rank()
    output = [
        tensor if (i == local_rank) else torch.empty_like(tensor) for i in range(world_size)
    ]
    dist.all_gather(output, tensor, async_op=False)
    return output


def all_gather_dict(result: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
    # only accepts dictionary that key is string and value is Tensor.
    new_result = {}
    for k, v in result.items():
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"Input dictionary of all_gather_dict needs "
                             f"every item to be torch.Tensor, got ({k}: {type(v)})")
        new_result[k] = all_gather_tensor(v)
    return new_result


def _broadcast_object(obj: Any, src_rank, device) -> Any:
    # see FairSeq/distributed/utils
    # this function is intended to use with non-tensor objects.
    if src_rank == get_rank():
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        buffer = torch.ByteTensor(buffer.getbuffer()).to(device)
        length = torch.LongTensor([len(buffer)]).to(device)
        dist.broadcast(length, src=src_rank)
        dist.broadcast(buffer, src=src_rank)
    else:
        length = torch.LongTensor([0]).to(device)
        dist.broadcast(length, src=src_rank)
        buffer = torch.ByteTensor(int(length.item())).to(device)
        dist.broadcast(buffer, src=src_rank)
        buffer = io.BytesIO(buffer.cpu().numpy())
        obj = torch.load(buffer, map_location="cpu")
    return obj


def broadcast_objects(obj_list: List[Any], src_rank: int = 0) -> List[Any]:
    # list should have same length
    # dist.broadcast_object_list(obj_list, src=src_rank)  # somehow not working
    backend = torch.distributed.get_backend()
    if backend == torch.distributed.Backend.NCCL:
        device = torch.device("cuda")
    elif backend == torch.distributed.Backend.GLOO:
        device = torch.device("cpu")
    else:
        raise RuntimeError(f"Unsupported distributed backend: {backend}")

    out = []
    for obj in obj_list:
        out.append(_broadcast_object(obj, src_rank, device=device))
    return out


def broadcast_tensor(tensor: torch.Tensor, src_rank: int = 0) -> torch.Tensor:
    # tensor should have the same number of elements and dtype through GPUs
    dist.broadcast(tensor, src=src_rank)
    return tensor


def broadcast_tensors(tensors: List[torch.Tensor], src_rank: int = 0) -> List[torch.Tensor]:
    for tensor in tensors:
        dist.broadcast(tensor, src=src_rank)
    return tensors


def broadcast_any_tensor(tensor: Optional[torch.Tensor], src_rank: int = 0) -> torch.Tensor:
    # broadcast, not restricted to tensor shape and dtype match.
    device = torch.device("cuda")

    if src_rank == get_rank():
        if tensor is None:
            raise RuntimeError(f"Broadcast tensor in src_rank, but got None as input.")
        metadata = {"shape": tensor.shape, "dtype": tensor.dtype}
        metadata = _broadcast_object(metadata, src_rank, device)
    else:
        metadata = _broadcast_object(None, src_rank, device)

    if src_rank == get_rank():
        dist.broadcast(tensor, src=src_rank)
    else:
        tensor = torch.zeros(*metadata["shape"], dtype=metadata["dtype"], device=device)
        dist.broadcast(tensor, src=src_rank)
    return tensor

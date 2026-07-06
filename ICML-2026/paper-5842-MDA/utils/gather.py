# utils/gather.py

import torch
import torch.distributed as dist
import numpy as np


def gather_heap_as_tensors(heap_data, world_size, device, rank):
    """
    Gather heap data across ranks and return full array on rank 0.
    Each row: [score, sample_index, loss]
    """
    data_list = []
    for score, _, (idx, loss) in heap_data:
        data_list.append([float(score), float(int(idx)), float(loss)])

    if len(data_list) == 0:
        tensor_local = torch.zeros(0, 3, device=device, dtype=torch.float64)
    else:
        tensor_local = torch.tensor(data_list, device=device, dtype=torch.float64)

    if not dist.is_initialized():
        return tensor_local.cpu().numpy()

    size_local = torch.tensor([tensor_local.shape[0]], device=device, dtype=torch.long)
    sizes_gather = [torch.zeros_like(size_local) for _ in range(world_size)]
    dist.all_gather(sizes_gather, size_local)

    max_size = max([s.item() for s in sizes_gather])
    pad_size = max_size - tensor_local.shape[0]
    if pad_size > 0:
        padding = torch.full((pad_size, 3), -1e18, device=device, dtype=torch.float64)
        tensor_padded = torch.cat([tensor_local, padding], dim=0)
    else:
        tensor_padded = tensor_local

    gathered_tensors = [torch.zeros_like(tensor_padded) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor_padded)

    if rank == 0:
        all_rows = []
        for i, t in enumerate(gathered_tensors):
            valid_len = sizes_gather[i].item()
            if valid_len > 0:
                all_rows.append(t[:valid_len])
        if all_rows:
            return torch.cat(all_rows, dim=0).cpu().numpy()
        else:
            return np.zeros((0, 3))

    return None

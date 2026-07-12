from __future__ import annotations

import torch
from datasets import load_from_disk


def load_compression_embeddings(path: str, device: str | torch.device = "cpu") -> torch.Tensor:
    result = load_from_disk(path)
    compression_embeddings = torch.FloatTensor(result[0]["embedding"]).unsqueeze(dim=0).to(device)  # [batch, mem, hidden]
    return compression_embeddings

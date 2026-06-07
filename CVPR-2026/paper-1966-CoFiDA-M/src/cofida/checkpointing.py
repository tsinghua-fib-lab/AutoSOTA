import numpy as np
import torch


def load_checkpoint_safe(path: str):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        from torch.serialization import add_safe_globals

        add_safe_globals(
            [np.core.multiarray.scalar, np.dtype, type(np.float64(0)), type(np.int64(0))]
        )
        return torch.load(path, map_location="cpu", weights_only=False)

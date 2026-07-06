from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import torch

import globals as g


def resolve_device():
    """Return the default torch.device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(dtype_name):
    """Return a torch dtype from a globals dtype string."""
    if dtype_name == g.DTYPE_FP32:
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_name}")


def manual_seed(seed):
    """Seed torch RNGs for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

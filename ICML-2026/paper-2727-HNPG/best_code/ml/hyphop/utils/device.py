import torch


def select_device(device: str = "auto", no_accel: bool = False) -> torch.device:
    """
    Standard device selection across scripts.

    - device="auto": cuda -> mps -> cpu (unless no_accel=True)
    - device="cuda"/"mps"/"cpu": force that backend (if available)
    - no_accel=True: always returns cpu
    """
    if no_accel:
        return torch.device("cpu")

    device = (device or "auto").lower()

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")

    if device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")

    if device == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unknown device: {device}. Use auto|cpu|cuda|mps.")


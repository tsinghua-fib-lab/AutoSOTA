import torch


class ClassFreeMasker:
    def __init__(self, enabled: bool = True, p: float = 0.1):
        self.enabled = enabled
        self.p = p

    def maybe_mask(self, y: torch.Tensor, device: torch.device) -> torch.Tensor:
        if not self.enabled:
            return y
        mask = (torch.rand(1, device=device) < self.p)
        if mask:
            return torch.ones_like(y, device=device) * -1
        return y



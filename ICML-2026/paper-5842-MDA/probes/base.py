#probes/base.py
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class ProbeBase(ABC):
    """
    Abstract base class for all probe implementations.
    Subclasses must implement build_dataloader and compute_grad.
    """

    @abstractmethod
    def build_dataloader(self, cfg: dict) -> DataLoader | None:
        """
        Return a DataLoader for probe gradient computation,
        or None if the probe uses synthetic data.
        """
        ...

    @abstractmethod
    def compute_grad(self, model, cfg: dict, mode: str) -> tuple:
        """
        Compute probe gradients with respect to the target subspace.

        Args:
            model: the transformer model instance (possibly DDP-wrapped)
            cfg:   full experiment config dict
            mode:  "qk" or "qkvo"

        Returns:
            mode == "qk"   -> (v_qk,)
            mode == "qkvo" -> (v_qk, v_v, v_o)
        """
        ...


from abc import ABC, abstractmethod
from pathlib import Path
import torch
from torch import nn

from typing import Optional, Union

from utils.utils import get_device


class BaseModel(nn.Module, ABC):
    """BaseModel class
    Takes care of easy saving and loading.
    """
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def _get_constructor_parameters(self) -> dict:
        pass

    @abstractmethod
    def get_train_fingerprint(self) -> dict:
        pass

    def reset_parameters(self):
        pass

    @staticmethod
    def model_save_name(epoch: int) -> str:
        return f"model_epoch_{epoch:03d}"

    @property
    def num_parameters(self) -> int:
        return torch.tensor([p.numel() for p in self.parameters()]).sum().item()

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    def save(self, path: Path, model_name: str) -> None:
        save_path = path / model_name
        torch.save(
            {
                "state_dict": self.state_dict(),
                "data": self._get_constructor_parameters()
            }, save_path)

    def save_state(self, path: Path, model_name: str):
        save_path = path / model_name
        torch.save(
            {
                "state_dict": self.state_dict(),
            }, save_path)

    def load_state(self, save_path, device="cpu"):
        saved_variables = torch.load(str(save_path), map_location=device)
        self.load_state_dict(saved_variables["state_dict"])
        self.to(device=device)

    @classmethod
    def load(cls,
             path: Union[str, Path],
             model_name: str = None, 
             file_extension: Optional[str] = ".p",
             device: Union[torch.device, str, int] = "auto") -> "BaseModel":
        device = get_device(device)
        if isinstance(path, str):
            path = Path(path)
        if model_name is None:
            save_path = path
        else:
            save_path = path / (model_name+file_extension)
        saved_variables = torch.load(str(save_path), map_location=device)
        # Create model object
        model = cls(**saved_variables["data"])
        model.load_state_dict(saved_variables["state_dict"])
        # Load weights
        model.to(device)
        return model
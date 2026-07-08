import torch
from abc import ABC, abstractmethod

from substantive.faircp.structs.enums import CalibrationType


class Calib(ABC):
    @abstractmethod
    def calibrate(self, logits: torch.Tensor, h_params: dict) -> torch.Tensor: ...


class TemperatureScaling(Calib):
    def calibrate(self, logits: torch.Tensor, h_params: dict) -> torch.Tensor:
        return torch.softmax(logits / h_params["T"], dim=1)


def get_calib(type: CalibrationType) -> Calib:
    match type:
        case CalibrationType.TEMPERATURE:
            return TemperatureScaling()

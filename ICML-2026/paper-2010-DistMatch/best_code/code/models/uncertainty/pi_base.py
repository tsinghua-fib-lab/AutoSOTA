import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Optional, Union, List, Dict

import numpy as np
import torch

from models.forcast.forcast_base import PredictionOutputType
from models.forcast.forcast_service import ForcastService
from models.uncertainty.components.data_mixing import MixTsData, MixDataService


@dataclass
class PIModelPrediction:
    """Result of a prediction step"""
    pred_interval: Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]  # Prediction by UC Model
    fc_Y_hat: Optional[Union[torch.Tensor, np.ndarray]] = None  # Prediction by FC Model
    fc_interval: Optional[Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]] = None # Prediction by FC Model
    uc_attention: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None


@dataclass
class PIPredictionStepData:
    """Data available for a predictions tep"""
    ts_id: str
    step_offset_prediction: int          # Offset to prediction range
    step_offset_overall: int             # Offset to overall TS of first predictions step
    X_past: torch.Tensor                 # [past_len, #features]
    Y_past: torch.Tensor                 # [past_len, 1*]
    X_step: Optional[torch.Tensor]       # [#steps, #features] (steps=1 if single step prediction)
    eps_past: Optional[Union[torch.Tensor, np.ndarray]]     # [past_len, 1*]
    alpha: Optional[float] = None
    mix_ts: Optional[List[MixTsData]] = None
    score_param: Dict = field(default_factory=dict)


@dataclass
class PICalibData:
    """Data available for calibration"""
    ts_id: str
    step_offset: int   # Offset to overall TS of first calibration step
    X_calib: torch.Tensor
    Y_calib: torch.Tensor
    X_pre_calib: Optional[torch.Tensor] = None  # X Before the calib steps (typically X_train)
    Y_pre_calib: Optional[torch.Tensor] = None  # Y Before the calib steps (typically X_train)
    score_param: Dict = field(default_factory=dict)

@dataclass
class PICalibArtifacts:
    """Additional Output of calibration used e.g. for logging"""
    fc_Y_hat: Optional[Union[torch.Tensor, np.ndarray]] = None
    fc_interval: Optional[Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]] = None
    eps: Optional[Union[torch.Tensor, np.ndarray]] = None
    fc_state_step: Optional[Union[torch.Tensor, np.ndarray]] = None
    add_info = dict()


class PIModel(ABC):
    def __init__(self, use_dedicated_calibration: bool, fc_prediction_out_modes: Tuple[PredictionOutputType]):
        self._use_dedicated_calibration = use_dedicated_calibration
        self._fc_prediction_out_modes = fc_prediction_out_modes
        self._forcast_service: Optional[ForcastService] = None

    def set_forcast_model(self, forcast_model: ForcastService):
        self._forcast_service = forcast_model
        self._forcast_service.set_active_output_mode(list(self._fc_prediction_out_modes))

    def calibrate(self, calib_data: [PICalibData], alphas: List[float], **kwargs) -> [PICalibArtifacts]:
        self._check_calib_data(calib_data, alphas)
        if not self.can_handle_different_alpha and len(alphas) > 1:
            raise ValueError("Calibration for multiple alphas not possible!")
        return self._calibrate(calib_data=calib_data, alphas=alphas, **kwargs)

    def calibrate_individual(self, calib_data: PICalibData, alpha, calib_artifact: Optional[PICalibArtifacts],
                             mix_calib_data: Optional[List[PICalibData]],
                             mix_calib_artifact: Optional[List[PICalibArtifacts]]) -> PICalibArtifacts:
        pass

    def pre_predict(self, **kwargs):
        pass  # Implement in model in case any actions has to be done before the prediction

    def predict_step(self, Y_step, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        if not self.model_ready():
            raise ValueError("Model not ready for predict!")
        self._check_pred_data(pred_data)
        prediction = self._predict_step(pred_data, **kwargs)
        self._post_predict_step(Y_step=Y_step, pred_result=prediction, pred_data=pred_data, **kwargs)
        return prediction

    @property
    def use_dedicated_calibration(self):
        """Dedicated calibration step is used"""
        return self._use_dedicated_calibration

    @property
    def has_mix_service(self) -> bool:
        return False

    def required_past_len(self) -> Tuple[int, int]:
        """Tuple: min_required_length, max_used_length"""
        return 0, sys.maxsize

    @property
    def fc_prediction_out_modes(self):
        return self._fc_prediction_out_modes

    @abstractmethod
    def model_ready(self):
        """True if model is ready prediction"""
        pass

    def _calibrate(self, calib_data: [PICalibData], alphas, **kwargs) -> [PICalibArtifacts]:
        """
        :return: dict with optional values regarding logging (e.g. y hat)
        """
        raise NotImplemented("No dedicated Calibration implemented!")

    @abstractmethod
    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        pass

    def _post_predict_step(self, Y_step, pred_result: PIModelPrediction, pred_data: PIPredictionStepData, **kwargs):
        pass    # Implement in model in case any actions has to be done after a prediction step

    def _check_calib_data(self, calib_data: PICalibData, alphas):
        pass     # Implement in model to make sanity checks

    def _check_pred_data(self, pred_data: PIPredictionStepData):
        pass    # Implement in model to make sanity checks

    @property
    @abstractmethod
    def can_handle_different_alpha(self):
        pass

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union, Optional, Iterable

import numpy as np
import torch

from models.PersistenceService import PersistenceModel


class PredictionOutputType(Enum):
    POINT = 'point',
    QUANTILE = 'quantile'


class ForcastMode(Enum):
    ALL_ON_TRAIN = "all_on_train"               # Prepare complete forcast in prediction
    TRAIN_PER_PREDICT = 'train_per_predict'     # Call Train routine for each prediction
    PREDICT_INDEPENDENT ='predict_indpendent'   # Train once and predict with routine


class FCModelPrediction(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.offset_start: int = 0  # Set by Basemodel so no need to set in concrete FC Implementation!

    @property
    @abstractmethod
    def point(self) -> Optional[Union[np.ndarray, torch.tensor]]:
        pass

    @property
    @abstractmethod
    def quantile(self) -> Optional[Union[np.ndarray, torch.tensor]]:
        pass

    @property
    @abstractmethod
    def state(self) -> Optional[Union[np.ndarray, torch.tensor]]:
        pass

    @property
    @abstractmethod
    def no_of_steps(self):
        pass

    @abstractmethod
    def _slice_internal(self, start, end):
        pass

    def get_slice(self, start_abs, end_abs):
        """
        :param start_abs: start step (absolute to complete TS)
        :param end_abs:  end end (absolute to complete TS)
        :return:
        """
        if start_abs < self.offset_start or end_abs > self.offset_start + self.no_of_steps:
            raise ValueError("Slice out of bounds!")
        prediction = self._slice_internal(start_abs - self.offset_start, end_abs - self.offset_start)
        prediction.offset_start = start_abs
        return prediction

    @abstractmethod
    def convert_datatype(self, to_tensor):
        """Convert datatype to tensor or numpy if necessary"""
        pass


@dataclass
class FcSingleModelPrediction(FCModelPrediction):
    point: Optional[Union[np.ndarray, torch.tensor]] = None
    quantile: Optional[Tuple[Union[np.ndarray, torch.tensor], Union[np.ndarray, torch.tensor]]] = None
    state: Optional[Union[np.ndarray, torch.tensor]] = None

    def __post_init__(self):
        super().__init__()

    @property
    def no_of_steps(self):
        return self.point.shape[0] if self.point is not None else self.quantile[0].shape[0]

    def convert_datatype(self, to_tensor):
        if to_tensor:
            if isinstance(self.point, np.ndarray):
                self.point = torch.Tensor(self.point)
            if isinstance(self.state, np.ndarray):
                self.state = torch.Tensor(self.state)
            if self.quantile is not None and isinstance(self.quantile[0], np.ndarray):
                self.quantile = (torch.Tensor(self.quantile[0]), torch.Tensor(self.quantile[1]))
        else:
            if isinstance(self.point, torch.Tensor):
                self.point = np.array(self.point)
            if isinstance(self.state, torch.Tensor):
                self.state = np.array(self.state)
            if self.quantile is not None and isinstance(self.quantile[0], torch.Tensor):
                self.quantile = (np.array(self.quantile[0]), np.array(self.quantile[1]))

    def _slice_internal(self, start, end):
        return FcSingleModelPrediction(
            point=self.point[start:end] if self.point is not None else None,
            quantile=(self.quantile[0][start:end], self.quantile[1][start:end]) if self.quantile is not None else None,
            state=self.state[start:end] if self.state is not None else None
        )


class FcEnsembleModelPrediction(FCModelPrediction):

    def __init__(self, point, quantile, ensemble_dim) -> None:
        """
        Some values might be nan (if the respective model should not participate in the "final" result).
        They are ignored by the nanmean methods BUT there must be one non nan value in each time step for each feature
        """
        super().__init__()
        self._point: Optional[Union[np.ndarray, torch.tensor]] = point
        self._quantile: Optional[Union[np.ndarray, torch.tensor]] = quantile
        self._e_dim = ensemble_dim

    @property
    def point(self):
        if self._point is None:
            return None
        elif isinstance(self._point, torch.Tensor):
            return torch.nanmean(self._point, dim=self._e_dim)
        else:
            return np.nanmean(self._point, axis=self._e_dim)

    @property
    def point_individual(self):
        return self._point, self._e_dim

    @property
    def quantile(self):
        if self._quantile is None:
            return None
        elif isinstance(self._quantile[0], torch.Tensor):
            return torch.nanmean(self._quantile[0], dim=-1), torch.mean(self._quantile[1], dim=self._e_dim)
        else:
            return np.nanmean(self._quantile[0], axis=-1), torch.mean(self._quantile[1], dim=self._e_dim)

    @property
    def quantile_individual(self):
        return self._quantile, self._e_dim

    @property
    def states(self) -> Optional[Union[np.ndarray, torch.tensor]]:
        return None

    @property
    def no_of_steps(self):
        return self._point.shape[0] if self._point is not None else self._quantile[0].shape[0]

    def convert_datatype(self, to_tensor):
        if to_tensor:
            if isinstance(self._point, np.ndarray):
                self._point = torch.Tensor(self._point)
            if self._quantile is not None and isinstance(self._quantile[0], np.ndarray):
                self._quantile = (torch.Tensor(self._quantile[0]), torch.Tensor(self._quantile[1]))
        else:
            if isinstance(self._point, torch.Tensor):
                self._point = np.array(self._point)
            if self._quantile is not None and isinstance(self._quantile[0], torch.Tensor):
                self._quantile = (np.array(self._quantile[0]), np.array(self._quantile[1]))

    def _slice_internal(self, start, end):
        return FcSingleModelPrediction(
            point=self._point[start:end] if self._point is not None else None,
            quantile=(self._quantile[0][start:end], self._quantile[1][start:end]) if self._quantile is not None else None
        )


@dataclass
class FCPredictionData:
    ts_id: str
    step_offset: int   # Offset to overall TS of first predictions step
    Y_past: torch.Tensor
    X_past: torch.Tensor = None
    Y_step: Optional[torch.Tensor] = None
    X_step: Optional[torch.Tensor] = None
    alpha: Optional[float] = None
    _no_fc_steps: Optional[int] = None
    step_rel: int = None

    @property
    def no_fc_steps(self):
        if self._no_fc_steps:
            return self._no_fc_steps
        elif self.X_step is not None:
            return self.X_step.shape[0]
        elif self.Y_step is not None:
            return self.Y_step.shape[0]
        else:
            return 1


class ForcastModel(ABC, PersistenceModel):

    def __init__(self, forcast_mode: ForcastMode, supported_outputs: Iterable[PredictionOutputType]) -> None:
        super().__init__()
        self._forcast_mode = forcast_mode
        self._supported_output_modes = tuple(supported_outputs)
        self._active_output_mode = []
        self.cached_fc = None

    def train(self, Y, X=None, planned_fc_steps=None, step_offset=0, *args, **kwargs):
        """
        :param planned_fc_steps: precalculate fc steps if precalc_fc_steps is not None
        :param step_offset: offset of the first TRAINING step relative to complete TS
        """
        if self._forcast_mode == ForcastMode.TRAIN_PER_PREDICT:
            raise ValueError("Model needs no explicit pre-training!")
        elif self._forcast_mode == ForcastMode.ALL_ON_TRAIN:
            self.cached_fc, specified_offset = self._train(X=X, Y=Y, precalc_fc_steps=planned_fc_steps,
                                                           train_offset=step_offset + Y.shape[0], *args, **kwargs)
            offset = step_offset + Y.shape[0] if specified_offset is None else specified_offset
            if isinstance(self.cached_fc, FcSingleModelPrediction):
                self.cached_fc.offset_start = offset
            else:
                assert callable(self.cached_fc)
                self._cached_offset = offset
        else:
            self._train(X=X, Y=Y, *args, **kwargs, train_offset=step_offset + Y.shape[0])

    def train_global(self, datsets, alphas, trainer_config, experiment_config):
        raise NotImplemented("Not Implemented!")

    def predict(self, pred_data: FCPredictionData, retrieve_tensor=True, **kwargs) -> FCModelPrediction:
        self._check_pred_data(pred_data)
        if self._forcast_mode == ForcastMode.ALL_ON_TRAIN:
            if PredictionOutputType.QUANTILE in self.active_output_mode:
                pred_result = self.cached_fc(pred_data.alpha)
                pred_result.offset_start = self._cached_offset
                pred_result = pred_result.get_slice(pred_data.step_offset, pred_data.step_offset + pred_data.no_fc_steps)
            else:
                pred_result = self.cached_fc.get_slice(pred_data.step_offset, pred_data.step_offset + pred_data.no_fc_steps)
        elif self.forcast_mode == ForcastMode.TRAIN_PER_PREDICT:
            pred_result = self._train(X=pred_data.X_past, Y=pred_data.Y_past, alpha=pred_data.alpha,
                                      precalc_fc_steps=pred_data.no_fc_steps, **kwargs)
        else:
            pred_result = self._predict(pred_data, **kwargs)
        # Convert if necessary
        pred_result.offset_start = pred_data.step_offset
        pred_result.convert_datatype(retrieve_tensor)
        return pred_result

    @property
    def active_output_mode(self) -> List[PredictionOutputType]:
        return self._active_output_mode

    @active_output_mode.setter
    def active_output_mode(self, modes: List[PredictionOutputType]):
        if any(mode not in self._supported_output_modes for mode in modes):
            raise ValueError(f"One of the {modes} is not supported by forcast model!")
        self._active_output_mode = modes

    @property
    def forcast_mode(self) -> ForcastMode:
        return self._forcast_mode

    @property
    def fc_state_dim(self):
        return None

    @property
    def uses_past_for_prediction(self):
        """
        if false only Y_t = f_t(x_t) is allowed!
        """
        return True

    def _check_pred_data(self, pred_data: FCPredictionData):
        pass    # Implement in model to make sanity checks

    @abstractmethod
    def _train(self, X, Y, precalc_fc_steps=None, *args, **kwargs) -> Optional[Tuple[FCModelPrediction, Optional[int]]]:
        """Train model and precalculate fc steps if precalc_fc_steps is not None """
        pass

    @abstractmethod
    def _predict(self, pred_data: FCPredictionData, *args, **kwargs) -> FCModelPrediction:
        pass

    @property
    @abstractmethod
    def can_handle_different_alpha(self):
        pass

    @property
    @abstractmethod
    def train_per_time_series(self):
        pass

    def save(self, path: Path, model_name: str):
        if self._forcast_mode != ForcastMode.ALL_ON_TRAIN:
            raise NotImplemented("Own Class implementation needed for save!")
        save_path = path / model_name
        torch.save(
            {
                "constructor_param": self._get_constructor_parameters(),
                "cached_fc": self.cached_fc
            }, save_path)

    @classmethod
    def load(cls, path):
        saved_variables = torch.load(str(path))
        model = cls(**saved_variables["constructor_param"])
        if model._forcast_mode != ForcastMode.ALL_ON_TRAIN:
            raise NotImplemented("Own Class implementation needed for load!")
        model.cached_fc = saved_variables['cached_fc']
        return model

    def _get_constructor_parameters(self):
        return {}

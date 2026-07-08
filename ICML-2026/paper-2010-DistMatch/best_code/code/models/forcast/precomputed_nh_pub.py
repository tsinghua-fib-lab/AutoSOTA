import logging
import sys
from pathlib import Path
import pickle
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch

from models.forcast.forcast_base import ForcastMode, ForcastModel, PredictionOutputType, FCModelPrediction, \
    FCPredictionData, FcSingleModelPrediction

LOGGER = logging.getLogger(__name__)


class PrecomputedNeuralHydrologyForcastPUB(ForcastModel):
    """Model which returns precomputed predictions from a NeuralHydrology model.
    
    To train the neuralhydrology model, run: `nh-run train configuration/hydrology/nh-config.yaml`.
    To pre-calculate the predictions, run: `nh-run evaluate --run-dir outputs/neuralhydrology/regression_camels_XYZ`.
    """
    def __init__(self, **kwargs):
        super().__init__(forcast_mode=ForcastMode.PREDICT_INDEPENDENT, supported_outputs=(PredictionOutputType.POINT,))
        self._target_variable = kwargs['model_params']['target_variable']
        # Precomputed Predictions
        self._neuralhydrology_results_calib_path = Path(kwargs['model_params']['nh_results_calib_path'])
        self._neuralhydrology_results_pub_path = Path(kwargs['model_params']['nh_results_pub_path'])
        self._precomputed_calib_predictions = {}
        self._precomputed_pub_predictions = {}
        self._prediction_offset_calib = {}
        assert self._neuralhydrology_results_calib_path.is_file()
        assert self._neuralhydrology_results_pub_path.is_file()
        # Precomputed States
        if kwargs['model_params']['nh_states_calib_path'] is not None:
            self._neuralhydrology_states_calib_path = Path(kwargs['model_params']['nh_states_calib_path'])
            assert self._neuralhydrology_states_calib_path.is_dir()
        else:
            self._neuralhydrology_states_calib_path = None
        if kwargs['model_params']['nh_states_pub_path'] is not None:
            self._neuralhydrology_states_pub_path = Path(kwargs['model_params']['nh_states_pub_path'])
            assert self._neuralhydrology_states_pub_path.is_dir()
        else:
            self._neuralhydrology_states_pub_path = None

        self._nh_states_used = kwargs['model_params']['nh_states_used']
        self._state_dim = None
        self._loaded_basin_state = None
        self._current_loaded_basin_state_id = None
        self._allow_shorter_prediction_period = kwargs['model_params'].get('allow_shorter_prediction_period', False)

    def _train(self, X, Y, precalc_fc_steps=None, *args, **kwargs) -> Optional[Tuple[FCModelPrediction, Optional[int]]]:
        raise NotImplemented("Not used")

    def train_global(self, datsets, alphas, trainer_config, experiment_config):
        if len(self._precomputed_calib_predictions) > 0:
            LOGGER.info("Already loaded - skip")

        # Load Predictions
        y_mean, y_std = datsets[0].Y_normalize_props
        normalize = lambda pred: (torch.from_numpy(pred) - y_mean) / y_std
        for d in datsets:
            basin = d.ts_id.split('_')[0]
            self._prediction_offset_calib[basin] = d.calib_step

        #slice_end = kwargs['X_full'].shape[0] - Y.shape[0]
        with self._neuralhydrology_results_calib_path.open('rb') as f:
            results = pd.read_pickle(f)
            for basin, values in results.items():
                self._precomputed_calib_predictions[basin] = self._slice_predictions(values, 0, sys.maxsize)
                self._precomputed_calib_predictions[basin] = normalize(self._precomputed_calib_predictions[basin])

        LOGGER.info(f'Loaded Calib NeuralHydrology predictions from {self._neuralhydrology_results_calib_path}.')
        with self._neuralhydrology_results_pub_path.open('rb') as f:
            results = pd.read_pickle(f)
            for basin, values in results.items():
                self._precomputed_pub_predictions[basin] = self._slice_predictions(values, 0, sys.maxsize)
                self._precomputed_pub_predictions[basin] = normalize(self._precomputed_pub_predictions[basin])
        LOGGER.info(f'Loaded Calib NeuralHydrology predictions from {self._neuralhydrology_results_pub_path}.')

        # Check State Files and load one
        if self._neuralhydrology_states_calib_path is not None:
            assert self._neuralhydrology_states_pub_path is not None
            # Load one and to get state dim
            basin = next(iter(self._precomputed_calib_predictions.keys()))
            assert self._prediction_offset_calib[basin] != 0  # TODO Why?
            self._load_basin_state(basin)
            self._state_dim = self._loaded_basin_state.shape[1]
            # Load one pub to check if state dim matches
            basin = next(iter(self._precomputed_pub_predictions.keys()))
            self._load_basin_state(basin)
            assert self._state_dim == self._loaded_basin_state.shape[1]
        return None

    def _predict(self, pred_data: FCPredictionData, *args, **kwargs) -> FCModelPrediction:
        basin = pred_data.ts_id.split('_')[0]
        if basin in self._precomputed_calib_predictions:
            start = pred_data.step_offset - self._prediction_offset_calib[basin]
            end = start + pred_data.no_fc_steps
            prediction = self._precomputed_calib_predictions[basin][start:end]
        elif basin in self._precomputed_pub_predictions:
            start = pred_data.step_rel
            end = start + pred_data.no_fc_steps
            prediction = self._precomputed_pub_predictions[basin][start:end]
        else:
            raise ValueError("Basin not loaded!")

        if self._neuralhydrology_states_calib_path is not None:
            self._load_basin_state(basin)
            states = self._loaded_basin_state[start:end]
        else:
            states = None
        return FcSingleModelPrediction(point=prediction[..., None], state=states)

    def _check_pred_data(self, pred_data: FCPredictionData):
        # We don't actually use these since everything is precomputed, but in theory (if we'd re-run the model), we
        # would need this information.
        # assert pred_data.X_step is not None
        # assert pred_data.X_past is not None
        pass

    def can_handle_different_alpha(self):
        return True

    @property
    def train_per_time_series(self):
        return False

    @property
    def uses_past_for_prediction(self):
        return True

    @property
    def fc_state_dim(self):
        return self._state_dim

    def _load_basin_state(self, basin, force_reload=False):
        # Cache single basin
        if self._current_loaded_basin_state_id != basin or force_reload:
            is_calib = basin in self._precomputed_calib_predictions
            file = self._get_basin_state_file(basin, is_calib=is_calib)
            with file.open('rb') as f:
                self._loaded_basin_state = self._slice_states(pickle.load(f), 0, sys.maxsize)
            if self._allow_shorter_prediction_period:
                if is_calib:
                    assert self._precomputed_calib_predictions[basin].shape[0] <= self._loaded_basin_state.shape[0]
                else:
                    assert self._precomputed_pub_predictions[basin].shape[0] <= self._loaded_basin_state.shape[0]
            else:
                if is_calib:
                    assert self._precomputed_calib_predictions[basin].shape[0] == self._loaded_basin_state.shape[0]
                else:
                    assert self._precomputed_pub_predictions[basin].shape[0] == self._loaded_basin_state.shape[0]
            self._current_loaded_basin_state_id = basin
            LOGGER.info(f'Loaded NeuralHydrology Model states from {file}.')
        else:
            pass

    def _get_basin_state_file(self, basin, is_calib):
        if is_calib:
            assert self._neuralhydrology_states_calib_path is not None
            return self._neuralhydrology_states_calib_path / f"validation_basin_{basin}_states.p"
        else:
            assert self._neuralhydrology_states_pub_path is not None
            return self._neuralhydrology_states_pub_path / f"test_basin_{basin}_states.p"

    def _slice_predictions(self, basin_predictions, slice_start: int, slice_end: int) -> np.ndarray:
        precomputed_basin = basin_predictions['1D']['xr'][self._target_variable].sel(time_step=0)
        if slice_end is not None:
            Y_hat = precomputed_basin.isel(date=slice(slice_start, slice_end)).to_numpy()
        else:
            Y_hat = precomputed_basin.to_numpy()
        assert np.isnan(Y_hat).sum() == 0
        if self._allow_shorter_prediction_period:
            assert len(Y_hat) <= len(precomputed_basin)  # Just for debuging in case only a subset of test is evaluated
        else:
            assert len(Y_hat) == len(precomputed_basin)  # Check to make sure we didn't mess up train/val splits.
        return Y_hat

    def _slice_states(self, basin_states, slice_start: int, slice_end: int):
        precomputed_state = np.concatenate([basin_states[s] for s in self._nh_states_used], axis=2)
        precomputed_state = precomputed_state[slice_start:slice_end, :, :]
        precomputed_state = np.swapaxes(precomputed_state, 1, 2)
        assert np.isnan(precomputed_state).sum() == 0
        return precomputed_state

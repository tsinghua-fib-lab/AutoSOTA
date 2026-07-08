import logging
import torch
from collections import defaultdict
from typing import Callable, List, Dict

from config import TaskConfig, TSDataConfig
from loader.dataset import TsDataset, ChronoSplittedTsDataset
from models.PersistenceService import ModelPersistenceService
from models.base_model import BaseModel
from models.forcast.forcast_base import PredictionOutputType, FCPredictionData
from models.forcast.forcast_service import ForcastService
from models.uncertainty.pi_base import PIModel, PIPredictionStepData, PIModelPrediction, PICalibData, PICalibArtifacts
from models.uncertainty.score_service import get_score_param
from utils.calc_torch import calc_residuals

LOGGER = logging.getLogger(__name__)


class UncertaintyService:

    def __init__(self, int_uc_model: Callable[[], PIModel], fc_service: ForcastService, save_uc_models,
                 task_config: TaskConfig, data_config: TSDataConfig, persist_dir: str):
        super().__init__()
        self._persist_service = ModelPersistenceService(model_dict=None, base_directory=persist_dir)
        self._save_uc_models = save_uc_models
        self._int_uc_model = int_uc_model
        self._fc_service = fc_service
        self._uc_models: Dict[str, Dict[float, PIModel]] = defaultdict(dict)
        self._calib_artefacts: Dict[str, Dict[float, PICalibArtifacts]] = defaultdict(dict)
        self._own_per_alpha = None
        self._own_per_data = False
        self._force_eval_start_at_test = True  # Make sure all models use the same calibration start (also if no calibration is needed)
        self._task_config = task_config
        self._data_config = data_config

    def prepare(self, datasets: List[TsDataset], alphas, experiment_config, calib_trainer_config):
        uc_prototype: PIModel = self._int_uc_model()
        fingerprint = None
        if isinstance(uc_prototype, BaseModel):
            fingerprint = self._persistence_fingerprint(
                model_fp=uc_prototype.get_train_fingerprint(),
                experiment_config=experiment_config, trainer_config=calib_trainer_config)
            if self._persist_service.model_existing(fingerprint) and self._save_uc_models:
                LOGGER.info("Load calibrated from persistence")
                self._uc_models = self._persist_service.load_models(fingerprint=fingerprint,
                                                                    init_model_func=self._int_uc_model)[0]
                return

        if not uc_prototype.use_dedicated_calibration:
            LOGGER.info("No calibration of uncertainty model needed.")
            self._own_per_alpha = False
            self._own_per_data = False
            uc_prototype.set_forcast_model(self._fc_service)
            self._set_model(None, None, uc_prototype)
        elif uc_prototype.can_handle_different_alpha:
            self._own_per_alpha = False
            uc_prototype.set_forcast_model(self._fc_service)
            calib_data = [self._map_to_calib_data(dataset) for dataset in datasets]
            artifacts = uc_prototype.calibrate(calib_data=calib_data, alphas=alphas,
                                               experiment_config=experiment_config,
                                               trainer_config=calib_trainer_config)
            self._set_model(None, None, uc_prototype)
            if artifacts is not None:
                for idx, dataset in enumerate(datasets):
                    self._set_calib_artifact(dataset.ts_id, None, artifacts[idx])
        else:
            self._own_per_alpha = True
            for alpha in alphas:
                model = self._int_uc_model()
                model.set_forcast_model(self._fc_service)
                calib_data = [self._map_to_calib_data(dataset) for dataset in datasets]
                artifacts = uc_prototype.calibrate(calib_data=calib_data, alphas=[alpha],
                                                   experiment_config=experiment_config,
                                                   calib_trainer_config=calib_trainer_config)
                self._set_model(None, alpha, model)
                if artifacts is not None:
                    for idx, dataset in enumerate(datasets):
                        self._set_calib_artifact(dataset.ts_id, alpha, artifacts[idx])

        if isinstance(uc_prototype, BaseModel) and self._save_uc_models:
            LOGGER.info("Persist trained models!")
            class_name = uc_prototype.__class__.__name__
            self._persist_service.save_models(self._uc_models, class_name, fingerprint,
                                              own_per_alpha=self._own_per_alpha, own_per_data=self._own_per_data)

    def pre_predict(self, dataset: TsDataset, alpha, other_datasets: List[TsDataset]):
        # 1) Execute Dataset individual calibration on uc model in case its used
        model = self._get_model(dataset.ts_id, alpha)
        try:
            calib_artifact = self.get_calib_artifact(dataset, alpha)
        except KeyError:
            calib_artifact = None

        if model.use_dedicated_calibration:
            if model.has_mix_service:
                selected_mix_data = model._mix_data_service.select_mix_inference_data(other_datasets)
                if len(selected_mix_data) > 0:
                    mix_calib_data = [self._map_to_calib_data(d) for d in selected_mix_data]
                    mix_calib_artifact = [self.get_calib_artifact(d, alpha) for d in selected_mix_data]
                else:
                    mix_calib_data = None
                    mix_calib_artifact = None
            else:
                mix_calib_data = None
                mix_calib_artifact = None
            calib_artifact = model.calibrate_individual(calib_data=self._map_to_calib_data(dataset), alpha=alpha,
                                                        calib_artifact=calib_artifact, mix_calib_data=mix_calib_data,
                                                        mix_calib_artifact=mix_calib_artifact)
            self._set_calib_artifact(dataset.ts_id, alpha, calib_artifact)
        elif dataset.has_calib_set:
            # Calculate the Calibration prediction nevertheless to be able to log/compare and use as "first window"
            calib_data = self._map_to_calib_data(dataset)
            fc_result = self._fc_service.predict(
                FCPredictionData(ts_id=calib_data.ts_id, X_past=calib_data.X_pre_calib, Y_past=calib_data.Y_pre_calib,
                                 X_step=calib_data.X_calib, step_offset=calib_data.step_offset))
            eps = None
            if PredictionOutputType.POINT in self._fc_service.active_output_mode(dataset.ts_id, alpha):
                eps = calc_residuals(Y_hat=fc_result.point, Y=calib_data.Y_calib)
            calib_artifact = PICalibArtifacts(fc_Y_hat=fc_result.point, fc_interval=fc_result.quantile, eps=eps)
            self._set_calib_artifact(dataset.ts_id, alpha, calib_artifact)

        # Set first prediction step
        if model.use_dedicated_calibration or self._force_eval_start_at_test:
            start_step = dataset.test_step
        else:
            start_step = dataset.first_prediction_step

        # 2) Prepare First Epsilon window if needed
        min_window_len, max_window_len = model.required_past_len()
        pre_predict_len = 0
        if min_window_len > 0 and start_step == dataset.test_step and dataset.has_calib_set and dataset.no_calib_steps > min_window_len:
            # Use Simply the calibration last calib epsilons
            eps = calib_artifact.eps[-min_window_len:].tolist()
        elif min_window_len > 0 and PredictionOutputType.POINT in self._fc_service.active_output_mode(dataset.ts_id, alpha):
            # No Calibration Data existing -> Cut first test data for eps window
            pre_predict_len = min_window_len
            Y = dataset.Y_full
            X = dataset.X_full
            Y_hat = self._fc_service.predict(
                FCPredictionData(ts_id=dataset.ts_id, step_offset=start_step, X_past=X[:start_step],
                                 Y_past=Y[:start_step], X_step=X[start_step:start_step+min_window_len])).point
            eps = calc_residuals(Y=Y[start_step:start_step+min_window_len], Y_hat=Y_hat).tolist()
            calib_artifact = PICalibArtifacts()
            calib_artifact.add_info['Y_hat_first_window'] = Y_hat, start_step, start_step + min_window_len
            calib_artifact.add_info['eps_first_window'] = torch.tensor(eps)
            self._set_calib_artifact(dataset.ts_id, alpha, calib_artifact)
        elif PredictionOutputType.POINT in self._fc_service.active_output_mode(dataset.ts_id, alpha):
            # Init eps for later usage
            eps = []
        else:
            eps = None

        # 3) Execute model pre_predict
        model.pre_predict(alpha=alpha)

        return start_step, pre_predict_len, max_window_len, eps

    def predict_step(self, Y_step, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        ts_id = pred_data.ts_id
        alpha = pred_data.alpha
        return self._get_model(ts_id, alpha).predict_step(Y_step=Y_step, pred_data=pred_data, **kwargs)

    def pack_mix_data(self, ts_id, alpha, **mix_kwargs):
        if self._get_model(ts_id, alpha).has_mix_service:
            return self._get_model(ts_id, alpha)._mix_data_service.pack_mix_inference_data(**mix_kwargs)
        else:
            return None

    def has_calib_artifact(self, dataset, alpha):
        return self._has_access(self._calib_artefacts, dataset.ts_id, alpha, force_per_data=True)

    def get_calib_artifact(self, dataset, alpha):
        return self._access(self._calib_artefacts, dataset.ts_id, alpha, force_per_data=True)

    def _set_calib_artifact(self, ts_id, alpha, artefact):
        self._set_access(self._calib_artefacts, ts_id, alpha, artefact, force_per_data=True)

    def has_model(self, ts_id, alpha):
        return self._has_access(self._uc_models, ts_id, alpha)

    def _get_model(self, ts_id, alpha):
        return self._access(self._uc_models, ts_id, alpha)

    def _set_model(self, ts_id, alpha, model):
        self._set_access(self._uc_models, ts_id, alpha, model)

    def _access(self, access_dict, ts_id, alpha, force_per_data=False):
        return access_dict[ts_id if self._own_per_data or force_per_data else "0"][str(alpha) if self._own_per_alpha else "0"]

    def _set_access(self, access_dict, ts_id, alpha, element, force_per_data=False):
        access_dict[ts_id if self._own_per_data or force_per_data else "0"][str(alpha) if self._own_per_alpha else "0"] = element

    def _has_access(self, access_dict, ts_id, alpha, force_per_data=False):
        return (ts_id if self._own_per_data or force_per_data else "0") in access_dict and\
               (str(alpha) if self._own_per_alpha else "0") in access_dict[ts_id if self._own_per_data or force_per_data else "0"]

    @staticmethod
    def _map_to_calib_data(dataset):
        return PICalibData(ts_id=dataset.ts_id, X_calib=dataset.X_calib, Y_calib=dataset.Y_calib,
                           X_pre_calib=dataset.X_train if isinstance(dataset, ChronoSplittedTsDataset) else None,
                           Y_pre_calib=dataset.Y_train if isinstance(dataset, ChronoSplittedTsDataset) else None,
                           step_offset=dataset.calib_step, score_param=get_score_param(dataset))

    def _persistence_fingerprint(self, model_fp, experiment_config, trainer_config):
        model_print = [f'{key}-{item}' for key, item in model_fp.items()]
        data_print = [f'{key}-{item}' for key, item in self._data_config.items()]
        task_print = [f'{key}-{item}' for key, item in self._task_config.items()]
        trainer_print = [f'{key}-{item}' for key, item in trainer_config.items()]
        return f"M#{'_'.join(model_print)}#D#{'_'.join(data_print)}#TASK#{'_'.join(task_print)}#TRAIN#{'_'.join(trainer_print)}"

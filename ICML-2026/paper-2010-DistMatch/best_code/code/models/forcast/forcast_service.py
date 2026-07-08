import itertools
import logging
from collections import defaultdict
from typing import Dict, Callable, List, Union

import numpy as np
import torch

from config import TaskConfig, TSDataConfig
from loader.dataset import BoostrapEnsembleTsDataset, TsDataset, ChronoSplittedTsDataset
from models.PersistenceService import ModelPersistenceService
from models.forcast.forcast_base import ForcastModel, FCPredictionData, FCModelPrediction, ForcastMode, \
    PredictionOutputType, FcEnsembleModelPrediction

from models.forcast.darts import SimpleDartsModel
from models.forcast.simple_point import ForestRegForcast, RidgeRegForcast

LOGGER = logging.getLogger(__name__)

PERSISTENCE_MODEL_DICT = {
    'SimpleDartsModel': SimpleDartsModel,
    'ForestRegForcast': ForestRegForcast,
    'RidgeRegForcast': RidgeRegForcast,
}


class ForcastService:

    def __init__(self, int_fc_model: Callable[[], ForcastModel], task_config: TaskConfig, model_config,
                 data_config: TSDataConfig, persist_dir: str, save_new_reg_bak=False, **kwargs) -> None:
        super().__init__()
        self._persist_service = ModelPersistenceService(model_dict=PERSISTENCE_MODEL_DICT, base_directory=persist_dir,
                                                        save_new_reg_bak=save_new_reg_bak)
        self._init_fc_model = int_fc_model
        self._own_per_alpha = None
        self._own_per_data = None
        self._task_config = task_config
        self._model_config = model_config
        self._data_config = data_config
        mode = task_config.fc_estimator_mode
        if mode == 'single':
            self._n_estimator = 1
            self._is_enbPiBoostrap = False
            self._enbpibootstrap_access = None
        elif mode.startswith("ensemble_"):
            self._n_estimator = int(mode.split("_")[-1])
            self._is_enbPiBoostrap = False
            self._enbpibootstrap_access = None
        elif mode.startswith("enbpi_boostrap_"):
            self._n_estimator = int(mode.split("_")[-1])
            self._is_enbPiBoostrap = True
            self._enbpibootstrap_access: Dict[str, Dict[str, List[ForcastModel]]] = defaultdict(lambda: defaultdict(list))
        else:
            raise ValueError("Invalid Estimator Mode!")
        self._fc_models_access: Dict[str, Dict[str, Union[ForcastModel, List[ForcastModel]]]] =\
            defaultdict(lambda: defaultdict(list)) if self.has_ensemble else defaultdict(dict)
        self._fc_models: List[ForcastModel] = []
        self._trainer_config = kwargs.get("trainer_config", None)
        self._experiment_config = kwargs.get("experiment_config", {})

    @property
    def has_ensemble(self):
        return self._n_estimator > 1

    @property
    def fc_state_dim(self):
        return self._fc_models[0].fc_state_dim if self._fc_models is not None else None

    def prepare(self, datasets: List[ChronoSplittedTsDataset], alphas) -> List[TsDataset]:
        fc_prototype: ForcastModel = self._init_fc_model()
        # Check if model is cached
        if fc_prototype.forcast_mode == ForcastMode.TRAIN_PER_PREDICT:
            self._own_per_data = False
            self._own_per_alpha = False
        else:
            self._own_per_data = fc_prototype.train_per_time_series
            self._own_per_alpha = not fc_prototype.can_handle_different_alpha

        # Prepare Models
        if self._is_enbPiBoostrap:
            # Persistence not working for enbPi Boostrap!
            if fc_prototype.uses_past_for_prediction:
                raise ValueError("FC Predictor which uses the past is not suitable for enbpi bootstrap")
            return self._prepare_enbpibootstrap_models(fc_prototype, datasets, alphas)
        else:
            # Check if persisted
            fingerprint = self._persistence_fingerprint()
            if self._persist_service.model_existing(fingerprint):
                LOGGER.info("Load models from persistence")
                self._fc_models_access, self._fc_models = self._persist_service.load_models(fingerprint)
            else:
                for i in range(self._n_estimator):
                    self._prepare_single(fc_prototype, datasets=datasets, alphas=alphas, idx=i, boostrap_mask=dict())
                class_name = self._fc_models[0].__class__.__name__
                if self._persist_service.model_supported(class_name):
                    LOGGER.info("Persist trained models!")
                    self._persist_service.save_models(self._fc_models_access, class_name, fingerprint,
                                                      own_per_alpha=self._own_per_alpha, own_per_data=self._own_per_data)
                else:
                    LOGGER.info("Model persistence not supported!")
            return datasets  # No Dataset Change

    def predict(self, pred_data: FCPredictionData, retrieve_tensor=True) -> FCModelPrediction:
        ts_id = pred_data.ts_id
        alpha = pred_data.alpha
        if self.has_ensemble:
            predictions = [self._get_model(ts_id, alpha)[i].predict(pred_data, retrieve_tensor=retrieve_tensor)
                           for i in range(self._n_estimator)]
            if self._is_enbPiBoostrap:
                point, quantile, e_dim = self._merge_enbpibootstrap_predictions(predictions, ts_id=ts_id, alpha=alpha,
                                                                                is_tensor=retrieve_tensor)
            else:
                point, quantile, e_dim = self._merge_ensemble_prediction(predictions, is_tensor=retrieve_tensor)
            return FcEnsembleModelPrediction(point=point, quantile=quantile, ensemble_dim=e_dim)
        else:
            return self._get_model(ts_id, alpha).predict(pred_data, retrieve_tensor=retrieve_tensor)

    def active_output_mode(self, ts_id, alpha) -> List[PredictionOutputType]:
        if self.has_ensemble:
            return self._get_model(ts_id, alpha)[0].active_output_mode
        else:
            return self._get_model(ts_id, alpha).active_output_mode

    def set_active_output_mode(self, modes: List[PredictionOutputType]):
        for model in self._fc_models:
            model.active_output_mode = modes

    def _prepare_single(self, fc_prototype: ForcastModel, datasets: List[TsDataset],
                        alphas: List[float], idx: int, boostrap_mask: Dict[str, torch.Tensor]):
        if fc_prototype.forcast_mode == ForcastMode.TRAIN_PER_PREDICT:
            LOGGER.info("Forcast model does not need extra training.")
            self._set_model(None, None, fc_prototype)
        elif self._own_per_data:
            for dataset in datasets:
                mask = boostrap_mask.get(dataset.ts_id, None)
                if mask is None:
                    Y_train = dataset.Y_train
                    X_train = dataset.X_train
                else:
                    Y_train, X_train = self._mask_train_data(dataset=dataset, mask=mask[idx])
                if not self._own_per_alpha:
                    model = self._init_fc_model()
                    LOGGER.info(f"Train Forcast model ({idx+1}/{self._n_estimator}) for ts: {dataset.ts_id} and alphas: {alphas}.")
                    model.train(Y=Y_train, X=X_train, alpha=alphas,
                                planned_fc_steps=dataset.no_calib_steps + dataset.no_test_steps if not self._is_enbPiBoostrap else None,
                                X_full=dataset.X_full,
                                ts_id=dataset.ts_id,
                                y_normalize_props=dataset.Y_normalize_props)
                    self._set_model(dataset.ts_id, None, model)
                    self._set_enbpibootstrap_mask(dataset.ts_id, None, mask)
                else:
                    for alpha in alphas:
                        model = self._init_fc_model()
                        LOGGER.info(f"Train Forcast model ({idx+1}/{self._n_estimator}) for ts: {dataset.ts_id} and alpha: {alpha}.")
                        model.train(Y=Y_train, X=X_train, alpha=alpha,
                                    planned_fc_steps=dataset.no_calib_steps + dataset.no_test_steps if not self._is_enbPiBoostrap else None,
                                    X_full=dataset.X_full,
                                    ts_id=dataset.ts_id,
                                    y_normalize_props=dataset.Y_normalize_props)
                        self._set_model(dataset.ts_id, alpha, model)
                        self._set_enbpibootstrap_mask(dataset.ts_id, alpha, mask)
        else:
            if not self._own_per_alpha:
                model = self._init_fc_model()
                if type(model).__name__ == "PrecomputedNeuralHydrologyForcast":
                    assert type(model).__name__ == "PrecomputedNeuralHydrologyForcast"
                    # TODO HACK - STILL CALL TRAINING METHOD MULTIPLE TIMES FOR FC THAT HAVE PRELOAODED DATASET
                    LOGGER.info(f"Train Forcast model ({idx+1}/{self._n_estimator}) for all TS and alphas: {alphas}.")
                    for dataset in datasets:
                        assert boostrap_mask.get(dataset.ts_id, None) is None
                        model.train(Y=dataset.Y_train, X=dataset.X_train, alpha=alphas,
                                    planned_fc_steps=dataset.no_calib_steps + dataset.no_test_steps if not self._is_enbPiBoostrap else None,
                                    X_full=dataset.X_full,
                                    ts_id=dataset.ts_id,
                                    y_normalize_props=dataset.Y_normalize_props)
                    self._set_model(None, None, model)
                elif type(model).__name__ == "PrecomputedNeuralHydrologyForcastPUB":
                    LOGGER.info(f"Train Forcast model ({idx+1}/{self._n_estimator}) for all TS and alphas: {alphas}.")
                    model.train_global(datasets, alphas, self._trainer_config, self._experiment_config)
                    self._set_model(None, None, model)
                elif type(model).__name__ == "GlobalLSTM":
                    LOGGER.info(f"Train Forcast model ({idx+1}/{self._n_estimator}) for all TS and alphas: {alphas}.")
                    model.train_global(datasets, alphas, self._trainer_config, self._experiment_config)
                    self._set_model(None, None, model)
            else:
                for alpha in alphas:
                    model = self._init_fc_model()
                    LOGGER.info(f"Train Forcast model ({idx+1}/{self._n_estimator}) for all TS and alpha: {alpha}.")
                    raise ValueError("Not implemented yet!")
                    #self._set_model(None, alpha, model)

    def _get_model(self, ts_id, alpha):
        return self._access(self._fc_models_access, ts_id, alpha)

    def _access(self, access_dict, ts_id, alpha):
        return access_dict[ts_id if self._own_per_data else "0"][str(alpha) if self._own_per_alpha else "0"]

    def _set_model(self, ts_id, alpha, model):
        self._fc_models.append(model)
        self._set_access(self._fc_models_access, ts_id, alpha, model)

    def _set_access(self, access_dict, ts_id, alpha, element, one_per_ensemble_model=True):
        if self.has_ensemble and one_per_ensemble_model:
            access_dict[ts_id if self._own_per_data else "0"][str(alpha) if self._own_per_alpha else "0"].append(element)
        else:
            access_dict[ts_id if self._own_per_data else "0"][str(alpha) if self._own_per_alpha else "0"] = element

    @staticmethod
    def _merge_ensemble_prediction(predictions: List[FCModelPrediction], is_tensor):
        # If there is a certain prediction in one model it should be in ALL models
        dim = -1
        point = None
        if predictions[0].point is not None:
            if is_tensor:
                point = torch.stack([prediction.point for prediction in predictions], dim=dim)
            else:
                point = np.stack([prediction.point for prediction in predictions], axis=dim)
        quantile = None
        if predictions[0].quantile is not None:
            if is_tensor:
                quantile = torch.stack([prediction.quantile[0] for prediction in predictions], dim=dim),\
                           torch.stack([prediction.quantile[1] for prediction in predictions], dim=dim)
            else:
                quantile = np.stack([prediction.quantile[0] for prediction in predictions], axis=dim), \
                           np.stack([prediction.quantile[1] for prediction in predictions], axis=dim)
        return point, quantile, dim

    #
    # EnbPi Boostrap related methods
    #

    @staticmethod
    def _mask_train_data(dataset, mask):
        """
        Use only the training data which is assigned by the boostrap mask
        :param mask: [steps]
        """
        mask_indices = torch.nonzero(mask, as_tuple=False).squeeze()
        Y = torch.index_select(dataset.Y_train, dim=0, index=mask_indices)
        X = torch.index_select(dataset.X_train, dim=0, index=mask_indices)
        return Y, X

    def _prepare_enbpibootstrap_models(self, fc_prototype: ForcastModel, datasets: List[ChronoSplittedTsDataset],
                                       alphas) -> List[BoostrapEnsembleTsDataset]:
        transformed_datasets = []
        boostrap_masks = dict()
        for dataset in datasets:
            steps = dataset.no_train_steps + dataset.no_calib_steps
            n_e = self._n_estimator
            # 2) Create boostrap mask for each dataset
            while True:
                # Boostrap samples for estimators
                bootstrap_selected = torch.multinomial(torch.ones(n_e, steps), int(steps * 0.66), replacement=True)
                # Check if all points are selected once AND also once not selected!
                # (first unique per estimator and unique + count over estimator)
                unique_per_estimator = []
                for select in bootstrap_selected:
                    unique_per_estimator.append(torch.unique(select))
                unique, counts = torch.unique(torch.tensor(list(itertools.chain.from_iterable(unique_per_estimator))), return_counts=True)
                if counts.shape[0] == steps and torch.all(counts < n_e):
                    break
                else: #TODO
                    LOGGER.warning(f"Bootstraping for ts {dataset.ts_id} lead to undesired behaviour - Try again")
            # Create a mask which indicates which timesteps are used by which estimator
            boostrap_mask = torch.zeros((n_e, steps), dtype=torch.bool)
            for est_idx, selected_idx in enumerate(unique_per_estimator):
                boostrap_mask[est_idx].index_fill_(0, selected_idx, 1)
            boostrap_masks[dataset.ts_id] = boostrap_mask   # [n_e, steps]
            # 3) Transform ChronoSplittedTsDataset in BoostrapEnsembleTsDataset
            transformed_dataset = BoostrapEnsembleTsDataset(ts_id=dataset.ts_id, X=dataset.X_full, Y=dataset.Y_full,
                                                            test_step=dataset.test_step)
            transformed_datasets.append(transformed_dataset)

        for i in range(self._n_estimator):
            self._prepare_single(fc_prototype, transformed_datasets, alphas, i, boostrap_masks)
        return transformed_datasets

    def _set_enbpibootstrap_mask(self, ts_id, alpha, mask):
        if self._is_enbPiBoostrap:
            self._set_access(self._enbpibootstrap_access, ts_id, alpha, mask, one_per_ensemble_model=False)

    def _get_enbpibootstrap_mask(self, ts_id, alpha):
        return self._access(self._enbpibootstrap_access, ts_id, alpha)

    def _merge_enbpibootstrap_predictions(self, predictions: List[FCModelPrediction], ts_id, alpha, is_tensor):
        start_step = predictions[0].offset_start
        mask = self._get_enbpibootstrap_mask(ts_id, alpha)
        point, quantile, e_dim = self._merge_ensemble_prediction(predictions, is_tensor)
        if start_step >= mask.shape[1]:
            # Already out of train/calib -> use all ensemble models
            return point, quantile, e_dim
        elif start_step + predictions[0].no_of_steps <= mask.shape[1]:
            # All within train/calib -> mask out values to nan for values that are used in training
            mask = mask[:, start_step: start_step + predictions[0].no_of_steps]    #Relevant steps
            if point is not None:
                point = self._mask_prediction(point, mask, e_dim)
            if quantile is not None:
                quantile = self._mask_prediction(quantile[0], mask, e_dim),\
                           self._mask_prediction(quantile[1], mask, e_dim)
            return point, quantile, e_dim
        else:
            # Some within - Some out train/calib - mask the respective values
            # First Slice -> within train/calib
            if point is not None:
                split_point = mask.shape[1] - start_step
                point_c = point[:split_point]
                point_d = point[split_point:]
                mask = mask[:, start_step: split_point]
                if is_tensor:
                    point = torch.cat((self._mask_prediction(point_c, mask, e_dim), point_d), dim=0)
                else:
                    point = np.concatenate((self._mask_prediction(point_c, mask, e_dim), point_d), axis=0)
            if quantile is not None:
                raise NotImplemented("Not IMplemented yet")
            return point, quantile, e_dim

    @staticmethod
    def _mask_prediction(prediction: torch.Tensor, mask: torch.Tensor, estimator_dim: int):
        """
        :param prediction: [steps,..,est_idx]
        :param mask:       [est_idx, steps]
        :return:
        """
        if estimator_dim != -1:
            raise NotImplemented("Can not handle ensemble dimension not on last position!")
        # TODO Maybe vectorize
        for est_idx, est_mask in enumerate(mask):
            prediction[est_mask, ..., est_idx] = float('nan')
        return prediction

    def _persistence_fingerprint(self):
        model_print = [f'{key}-{item}' for key, item in self._model_config.items()]
        data_print = [f'{key}-{item}' for key, item in self._data_config.items()]
        task_print = [f'{key}-{item}' for key, item in self._task_config.items() if key != 'global_norm' or item is True]
        return f"M#{'_'.join(model_print)}#D#{'_'.join(data_print)}#T#{'_'.join(task_print)}"

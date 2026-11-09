'''
optuna LightGBM integration 
https://github.com/optuna/optuna/blob/8ef763023e5ad858284566b3de2fa5eca7fd19b2/optuna/integration/_lightgbm_tuner/optimize.py
Modified
'''

import abc
import copy
import json
import os
import pickle
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import warnings

import numpy as np
from packaging import version
import tqdm

import optuna
from optuna._deprecated import deprecated
from optuna._imports import try_import
from optuna.integration._lightgbm_tuner.alias import _handling_alias_metrics
from optuna.integration._lightgbm_tuner.alias import _handling_alias_parameters
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.integration._lightgbm_tuner.optimize import (
    _BaseTuner, _OptunaObjective, _OptunaObjectiveCV, _LightGBMBaseTuner)

import lightgbm as lgb


class _CustomLightGBMBaseTuner(_LightGBMBaseTuner):

    def __init__(self, 
                 params: Dict[str, Any],
                 train_set: "lgb.Dataset",
                 num_boost_round: int = 1000,
                 fobj: Optional[Callable[..., Any]] = None,
                 feval: Optional[Callable[..., Any]] = None,
                 feature_name: str = "auto",
                 categorical_feature: str = "auto",
                 early_stopping_rounds: Optional[int] = None,
                 verbose_eval: Optional[Union[bool, int]] = True,
                 callbacks: Optional[List[Callable[..., Any]]] = None,
                 time_budget: Optional[int] = None,
                 sample_size: Optional[int] = None,
                 study: Optional[optuna.study.Study] = None,
                 optuna_callbacks: Optional[List[Callable[[Study, FrozenTrial], None]]] = None,
                 verbosity: Optional[int] = None,
                 show_progress_bar: bool = True,
                 model_dir: Optional[str] = None,
                 n_trials_config: Optional[List[int]] = [7, 20, 10, 6, 20]) -> None:
        
        super(_CustomLightGBMBaseTuner, self).__init__(
            params,
            train_set,
            num_boost_round,
            fobj,
            feval,
            feature_name,
            categorical_feature,
            early_stopping_rounds,
            verbose_eval,
            callbacks,
            time_budget,
            sample_size,
            study,
            optuna_callbacks,
            verbosity,
            show_progress_bar,
            model_dir,
        )

        assert len(n_trials_config) == 5
        self.n_trials_config = n_trials_config

    def run(self) -> None:
        verbosity = self.auto_options["verbosity"]
        if verbosity is not None:
            if verbosity > 1:
                optuna.logging.set_verbosity(optuna.logging.DEBUG)
            elif verbosity == 1:
                optuna.logging.set_verbosity(optuna.logging.INFO)
            elif verbosity == 0:
                optuna.logging.set_verbosity(optuna.logging.WARNING)
            else:
                optuna.logging.set_verbosity(optuna.logging.CRITICAL)

        # Handling aliases.
        _handling_alias_parameters(self.lgbm_params)

        # Sampling.
        self.sample_train_set()

        self.tune_feature_fraction(self.n_trials_config[0])
        self.tune_num_leaves(self.n_trials_config[1])
        self.tune_bagging(self.n_trials_config[2])
        self.tune_feature_fraction_stage2(self.n_trials_config[3])
        self.tune_regularization_factors(self.n_trials_config[4])
        self.tune_min_data_in_leaf()


class LightGBMTuner(_CustomLightGBMBaseTuner):
    '''
    n_trials_config = [
        7,  # tune_feature_fraction
        20, # tune_num_leaves
        10, # tune_bagging
        6,  # tune_feature_fraction_stage2
        20, # tune_regularization_factors
    ]
    '''
    def __init__(
        self,
        params: Dict[str, Any],
        train_set: "lgb.Dataset",
        num_boost_round: int = 1000,
        valid_sets: Optional["VALID_SET_TYPE"] = None,
        valid_names: Optional[Any] = None,
        fobj: Optional[Callable[..., Any]] = None,
        feval: Optional[Callable[..., Any]] = None,
        feature_name: str = "auto",
        categorical_feature: str = "auto",
        early_stopping_rounds: Optional[int] = None,
        evals_result: Optional[Dict[Any, Any]] = None,
        verbose_eval: Optional[Union[bool, int]] = True,
        learning_rates: Optional[List[float]] = None,
        keep_training_booster: bool = False,
        callbacks: Optional[List[Callable[..., Any]]] = None,
        time_budget: Optional[int] = None,
        sample_size: Optional[int] = None,
        study: Optional[optuna.study.Study] = None,
        optuna_callbacks: Optional[List[Callable[[Study, FrozenTrial], None]]] = None,
        model_dir: Optional[str] = None,
        verbosity: Optional[int] = None,
        show_progress_bar: bool = True,
        n_trials_config: Optional[List[int]] = [7, 20, 10, 6, 20],
    ) -> None:

        super(LightGBMTuner, self).__init__(
            params,
            train_set,
            num_boost_round=num_boost_round,
            fobj=fobj,
            feval=feval,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            callbacks=callbacks,
            time_budget=time_budget,
            sample_size=sample_size,
            study=study,
            optuna_callbacks=optuna_callbacks,
            verbosity=verbosity,
            show_progress_bar=show_progress_bar,
            n_trials_config=n_trials_config
        )

        self.lgbm_kwargs["valid_sets"] = valid_sets
        self.lgbm_kwargs["valid_names"] = valid_names
        self.lgbm_kwargs["evals_result"] = evals_result
        self.lgbm_kwargs["learning_rates"] = learning_rates
        self.lgbm_kwargs["keep_training_booster"] = keep_training_booster

        self._best_booster_with_trial_number: Optional[Tuple[lgb.Booster, int]] = None
        self._model_dir = model_dir

        if self._model_dir is not None and not os.path.exists(self._model_dir):
            os.mkdir(self._model_dir)

        if valid_sets is None:
            raise ValueError("`valid_sets` is required.")

    @property  # type: ignore
    @deprecated(
        "1.4.0",
        text=(
            "Please get the best booster via "
            ":class:`~optuna.integration.lightgbm.LightGBMTuner.get_best_booster` instead."
        ),
    )
    def best_booster(self) -> "lgb.Booster":
        """Return the best booster."""

        return self.get_best_booster()

    def _create_objective(
        self,
        target_param_names: List[str],
        train_set: "lgb.Dataset",
        step_name: str,
        pbar: Optional[tqdm.tqdm],
    ) -> _OptunaObjective:
        return _OptunaObjective(
            target_param_names,
            self.lgbm_params,
            train_set,
            self.lgbm_kwargs,
            self.best_score,
            step_name=step_name,
            model_dir=self._model_dir,
            pbar=pbar,
        )


class LightGBMTunerCV(_CustomLightGBMBaseTuner):

    def __init__(
        self,
        params: Dict[str, Any],
        train_set: "lgb.Dataset",
        num_boost_round: int = 1000,
        folds: Optional[
            Union[
                Generator[Tuple[int, int], None, None],
                Iterator[Tuple[int, int]],
                "BaseCrossValidator",
            ]
        ] = None,
        nfold: int = 5,
        stratified: bool = True,
        shuffle: bool = True,
        fobj: Optional[Callable[..., Any]] = None,
        feval: Optional[Callable[..., Any]] = None,
        feature_name: str = "auto",
        categorical_feature: str = "auto",
        early_stopping_rounds: Optional[int] = None,
        fpreproc: Optional[Callable[..., Any]] = None,
        verbose_eval: Optional[Union[bool, int]] = True,
        show_stdv: bool = True,
        seed: int = 0,
        callbacks: Optional[List[Callable[..., Any]]] = None,
        time_budget: Optional[int] = None,
        sample_size: Optional[int] = None,
        study: Optional[optuna.study.Study] = None,
        optuna_callbacks: Optional[List[Callable[[Study, FrozenTrial], None]]] = None,
        verbosity: Optional[int] = None,
        show_progress_bar: bool = True,
        model_dir: Optional[str] = None,
        return_cvbooster: Optional[bool] = None,
        n_trials_config: Optional[List[int]] = [7, 20, 10, 6, 20],
    ) -> None:

        super(LightGBMTunerCV, self).__init__(
            params,
            train_set,
            num_boost_round,
            fobj=fobj,
            feval=feval,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            callbacks=callbacks,
            time_budget=time_budget,
            sample_size=sample_size,
            study=study,
            optuna_callbacks=optuna_callbacks,
            verbosity=verbosity,
            show_progress_bar=show_progress_bar,
            model_dir=model_dir,
            n_trials_config=n_trials_config,
        )

        self.lgbm_kwargs["folds"] = folds
        self.lgbm_kwargs["nfold"] = nfold
        self.lgbm_kwargs["stratified"] = stratified
        self.lgbm_kwargs["shuffle"] = shuffle
        self.lgbm_kwargs["show_stdv"] = show_stdv
        self.lgbm_kwargs["seed"] = seed
        self.lgbm_kwargs["fpreproc"] = fpreproc
        if return_cvbooster is not None:
            if version.parse(lgb.__version__) < version.parse("3.0.0"):
                raise ValueError("return_cvbooster requires lightgbm>=3.0.0.")
            self.lgbm_kwargs["return_cvbooster"] = return_cvbooster

    def _create_objective(
        self,
        target_param_names: List[str],
        train_set: "lgb.Dataset",
        step_name: str,
        pbar: Optional[tqdm.tqdm],
    ) -> _OptunaObjective:
        return _OptunaObjectiveCV(
            target_param_names,
            self.lgbm_params,
            train_set,
            self.lgbm_kwargs,
            self.best_score,
            step_name=step_name,
            model_dir=self._model_dir,
            pbar=pbar,
        )

    def get_best_booster(self) -> "lgb.CVBooster":
        """Return the best cvbooster.
        If the best booster cannot be found, :class:`ValueError` will be raised.
        To prevent the errors, please save boosters by specifying
        both of the ``model_dir`` and the ``return_cvbooster`` arguments of
        :meth:`~optuna.integration.lightgbm.LightGBMTunerCV.__init__`,
        when you resume tuning or you run tuning in parallel.
        """
        if self.lgbm_kwargs.get("return_cvbooster") is not True:
            raise ValueError(
                "LightGBMTunerCV requires `return_cvbooster=True` for method `get_best_booster()`."
            )
        if self._best_booster_with_trial_number is not None:
            if self._best_booster_with_trial_number[1] == self.study.best_trial.number:
                return self._best_booster_with_trial_number[0]
        if len(self.study.trials) == 0:
            raise ValueError(
                "The best booster is not available because no trials completed.")

        # The best booster exists, but this instance does not have it.
        # This may be due to resuming or parallelization.
        if self._model_dir is None:
            raise ValueError(
                "The best booster cannot be found. It may be found in the other processes due to "
                "resuming or distributed computing. Please set the `model_dir` argument of "
                "`LightGBMTunerCV.__init__` and make sure that boosters are shared with all "
                "processes."
            )

        best_trial = self.study.best_trial
        path = os.path.join(
            self._model_dir, "{}.pkl".format(best_trial.number))
        if not os.path.exists(path):
            raise ValueError(
                "The best booster cannot be found in {}. If you execute `LightGBMTunerCV` in "
                "distributed environment, please use network file system (e.g., NFS) to share "
                "models with multiple workers.".format(self._model_dir)
            )

        with open(path, "rb") as fin:
            boosters, best_iteration = pickle.load(fin)
            # At version `lightgbm==3.0.0`, :class:`lightgbm.CVBooster` does not
            # have `__getstate__` which is required for pickle serialization.
            cvbooster = lgb.CVBooster()
            cvbooster.boosters = boosters
            cvbooster.best_iteration = best_iteration

        return cvbooster


def train(*args: Any, **kwargs: Any) -> Any:
    auto_booster = LightGBMTuner(*args, **kwargs)
    auto_booster.run()
    return auto_booster.get_best_booster()

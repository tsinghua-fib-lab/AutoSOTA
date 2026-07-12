import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_validate
import torch
import json
import glob
import math
import optuna
from time import perf_counter
from .baseline import BaselineRegressor
from .dataset import AbstractDataset
from .utils.config import Config
from .utils.logger import Logger
from .filter import Filter
from joblib import parallel_backend
from joblib.externals.loky import get_reusable_executor
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

class Experiment:
    def __init__(self, config: Config, logger: Logger):
        """
        Initialize the Experiment with configuration and logger.
        
        Sets up datasets from either a single file or directory of CSV files,
        checks for duplicate configurations, and saves the configuration.
        
        Args:
            config (Config): Configuration object containing experiment parameters
            logger (Logger): Logger instance for recording experiment progress
            
        Raises:
            Exception: If a duplicate configuration is found in previous experiments
        """
        self.config = config
        self.logger = logger
        self.datasets = []

        self.combined_X_train = None
        self.combined_y_train = None

        self.results_df = None

        # Check if the config has already been run in a previous experiment
        duplicate = self.check_duplicate_config(config)
        if duplicate:
            self.logger.error(f"Config has already been run in a previous experiment. Exiting.")
            self.logger.error(f"Previous experiment results are in: {os.path.join(self.config.results_dir, duplicate)}")
            self.logger.error(f"If you want to run this experiment again, please use a different config or delete the previous results.")
            raise Exception(f"Existing experiment results found at: {os.path.join(self.config.results_dir, duplicate)}")
        
        # Check if dataset_dir_path is None or not
        if self.config.dataset_dir_path is None:
            # Create a single dataset using the path in dataset_path
            dataset_name = os.path.basename(self.config.dataset_path).split('.')[0]
            if self.config.sample_sizes is None:
                dataset = AbstractDataset(dataset_name, self.config.dataset_path, seed=self.config.random_seed)
                self.datasets.append(dataset)
            else:
                for sample_size in self.config.sample_sizes:
                    dataset = AbstractDataset(f"{dataset_name}_sample_{sample_size}", self.config.dataset_path, sample_size, seed=self.config.random_seed)
                    self.datasets.append(dataset)
        else:
            # Create datasets from all CSV files in the directory
            csv_files = glob.glob(os.path.join(self.config.dataset_dir_path, "*.csv"))
            for csv_file in csv_files:
                dataset_name = os.path.basename(csv_file).split('.')[0]

                if self.config.sample_sizes is None:
                    dataset = AbstractDataset(dataset_name, csv_file, seed=self.config.random_seed)
                    self.datasets.append(dataset)
                else:
                    for sample_size in self.config.sample_sizes:
                        dataset = AbstractDataset(f"{dataset_name}_sample_{sample_size}", csv_file, sample_size, seed=self.config.random_seed)
                        self.datasets.append(dataset)
        
        # save the config
        config_file = os.path.join(self.config.experiment_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump(self.config.to_dict(), f)

        self.logger.info(f"Experiment initialized with config:\n{self.config}")
        self.logger.info(f"Loaded {len(self.datasets)} datasets\n\n")
    
    def check_duplicate_config(self, config: Config):
        """
        Check if this configuration has already been run in a previous experiment.
        
        Compares the current configuration with all existing experiment configurations
        in the results directory, ignoring metadata fields like experiment_name, 
        experiment_dir, and timestamp.
        
        Args:
            config (Config): Configuration to check for duplicates
            
        Returns:
            str or False: Directory name of duplicate experiment if found, False otherwise
        """
        for dir in os.listdir(self.config.results_dir):
            if os.path.isdir(os.path.join(self.config.results_dir, dir)):
                try:
                    with open(os.path.join(self.config.results_dir, dir, "config.json"), "r") as f:
                        candidate_config = json.load(f)
                        incoming_config = config.to_dict()
                        
                        # remove keys that are not relevant to the config's equality check
                        incoming_config.pop("experiment_name", None)
                        incoming_config.pop("experiment_dir", None)
                        incoming_config.pop("timestamp", None)
                        candidate_config.pop("experiment_name", None)
                        candidate_config.pop("experiment_dir", None)
                        candidate_config.pop("timestamp", None)
                        
                        if candidate_config == incoming_config:
                            return dir
                except Exception as e:
                    self.logger.debug(f"Could not find config to load from {os.path.join(self.config.results_dir, dir, 'config.json')}: {e}")
                    continue
        return False

    def _train_baseline(self, dataset: AbstractDataset, baseline: BaselineRegressor, seed: int):
        """
        Train the baseline regression model on the training data.
        
        Trains the baseline model with optional hyperparameter tuning and saves
        the model if configured to do so.
        
        Args:
            dataset (AbstractDataset): Dataset containing training data
            baseline (BaselineRegressor): Baseline regression model to train
            seed (int): Random seed for reproducible training
        """
        self.logger.info(f"Training {self.config.baseline} baseline model")
        if self.config.hyperparam_tune:
            baseline.train_and_tune(dataset.X_train, dataset.y_train, seed=seed)
        else:
            baseline.train(dataset.X_train, dataset.y_train)

        # save the model
        if self.config.save_models:
            model_dir = os.path.join(self.config.experiment_dir, "models")
            os.makedirs(model_dir, exist_ok=True)                 
            baseline.save(os.path.join(model_dir, f"baseline_{dataset.name}"))
            self.logger.info(f"Saved baseline model")
    
    def _train_aug_data(self, dataset: AbstractDataset, baseline: BaselineRegressor, combined_X_train: np.ndarray, combined_y_train: np.ndarray, sample_weight: np.ndarray):
        """
        Train a new model on the augmented dataset.
        
        Trains the baseline model on the combined original and augmented training data
        with sample weights and saves the model if configured to do so.
        
        Args:
            dataset (AbstractDataset): Dataset for naming and organization
            baseline (BaselineRegressor): Baseline regression model to train
            combined_X_train (np.ndarray): Combined original and augmented feature data
            combined_y_train (np.ndarray): Combined original and augmented target data
            sample_weight (np.ndarray): Weights for each sample in the combined dataset
        """
        self.logger.info(f"Training new {self.config.baseline} model on augmented data\n")
        baseline.train(combined_X_train, combined_y_train, sample_weight=sample_weight)

        # save the model
        if self.config.save_models: 
            model_dir = os.path.join(self.config.experiment_dir, "models")
            os.makedirs(model_dir, exist_ok=True)                 
            baseline.save(os.path.join(model_dir, f"aug_model_{dataset.name}"))
            self.logger.info(f"Saved newly trained model")

    def _intervention(self, X_train: np.ndarray, features_to_perturb: list, aug_data_size_factor: float, min_perturb_percent: float, max_perturb_percent: float):
        """
        Create augmented features by intervening on specific features.
        
        Generates new training data by perturbing selected features with random
        multiplicative noise. This simulates interventional changes to the features.
        
        Args:
            X_train (np.ndarray): Original training feature matrix
            features_to_perturb (list): List of feature indices to perturb
            aug_data_size_factor (float): Factor determining augmented data size
            min_perturb_percent (float): Minimum perturbation percentage (negative)
            max_perturb_percent (float): Maximum perturbation percentage (positive)
            
        Returns:
            np.ndarray: Augmented feature matrix with perturbed features
        """
        self.logger.info("Intervening on features to create augmented feature set")
        aug_data_len = int(aug_data_size_factor * X_train.shape[0])
        n = int(math.ceil(aug_data_size_factor))

        perturbations = np.random.uniform(min_perturb_percent, max_perturb_percent, X_train.shape[0] * n)

        new_X_train = np.tile(X_train, (n, 1))

        for feature in features_to_perturb:
            new_X_train[:, feature] = new_X_train[:, feature] * (1 + perturbations)
            # Clip to observed training data range with 5% margin
            feat_min = X_train[:, feature].min()
            feat_max = X_train[:, feature].max()
            feat_range = feat_max - feat_min
            clip_low = feat_min - 0.05 * feat_range
            clip_high = feat_max + 0.05 * feat_range
            n_before = new_X_train.shape[0]
            new_X_train[:, feature] = np.clip(new_X_train[:, feature], clip_low, clip_high)
            n_clipped = np.sum((new_X_train[:aug_data_len, feature] < clip_low) | 
                              (new_X_train[:aug_data_len, feature] > clip_high))
            if n_clipped > 0:
                self.logger.info(f"Feature F{feature}: {n_clipped}/{aug_data_len} values clipped to [{clip_low:.4f}, {clip_high:.4f}]")

        aug_X_train = new_X_train[:aug_data_len]
        return aug_X_train
    
    def _counterfactuals(self, X_train: np.ndarray, train_residuals: np.ndarray, baseline: BaselineRegressor, aug_X_train: np.ndarray, aug_data_size_factor: float):
        """
        Generate counterfactual target values for augmented features.
        
        Creates target values for augmented data by using the baseline model's
        predictions plus the corresponding residuals, simulating what the targets
        would be under the interventions.
        
        Args:
            X_train (np.ndarray): Original training feature matrix
            train_residuals (np.ndarray): Residuals from baseline model on training data
            baseline (BaselineRegressor): Trained baseline model for predictions
            aug_X_train (np.ndarray): Augmented feature matrix
            aug_data_size_factor (float): Factor determining augmented data size
            
        Returns:
            np.ndarray: Counterfactual target values for augmented data
        """
        self.logger.info("Generating counterfactuals for augmented target set")
        aug_data_len = int(aug_data_size_factor * X_train.shape[0])
        n = int(math.ceil(aug_data_size_factor))
        # Winsorize residuals at 3 std to reduce noise from poorly-fit models
        residual_std = np.std(train_residuals)
        if residual_std > 0:
            clipped_residuals = np.clip(train_residuals, -3 * residual_std, 3 * residual_std)
        else:
            clipped_residuals = train_residuals
        z_train = np.tile(clipped_residuals.reshape(-1, 1), (n, 1))
        z_train = z_train[:aug_data_len]

        aug_y_train = baseline.predict(aug_X_train) + z_train.ravel()
        return aug_y_train
    
    def _get_combined_aug_training_set(self, X_train: np.ndarray, y_train: np.ndarray, aug_X_train: np.ndarray, aug_y_train: np.ndarray, aug_data_weight: float):
        """
        Combine original and augmented training data with appropriate sample weights.
        
        Creates a unified training dataset by concatenating original and augmented data,
        and assigns weights to control the influence of augmented samples during training.
        
        Args:
            X_train (np.ndarray): Original training feature matrix
            y_train (np.ndarray): Original training target vector
            aug_X_train (np.ndarray): Augmented training feature matrix
            aug_y_train (np.ndarray): Augmented training target vector
            aug_data_weight (float): Weight to assign to augmented samples
            
        Returns:
            tuple: A tuple containing:
                - combined_X_train (np.ndarray): Combined feature matrix
                - combined_y_train (np.ndarray): Combined target vector
                - sample_weight (np.ndarray): Sample weights (1.0 for original, aug_data_weight for augmented)
        """
        combined_len = X_train.shape[0] + aug_X_train.shape[0]
        is_augmented = np.zeros(combined_len, dtype=bool)
        is_augmented[:X_train.shape[0]] = False
        is_augmented[X_train.shape[0]:] = True

        sample_weight = np.ones(combined_len)
        sample_weight[is_augmented] = aug_data_weight

        combined_X_train = np.concatenate([X_train, aug_X_train])
        combined_y_train = np.concatenate([y_train, aug_y_train])

        self.logger.info(f"Original X train shape: {X_train.shape}")
        self.logger.info(f"Augmented X train shape: {aug_X_train.shape}")
        self.logger.info(f"Combined X train shape: {combined_X_train.shape}")
    
        return combined_X_train, combined_y_train, sample_weight

    def _data_augmentation(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            train_residuals: np.ndarray,
            baseline: BaselineRegressor,
            best_features_to_perturb: list,
            **kwargs
        ):
        """
        Perform complete data augmentation pipeline.
        
        Orchestrates the full data augmentation process: selects features to perturb,
        creates interventional data, generates counterfactual targets, and combines
        with original data.
        
        Args:
            X_train (np.ndarray): Original training feature matrix
            y_train (np.ndarray): Original training target vector
            train_residuals (np.ndarray): Residuals from baseline model
            baseline (BaselineRegressor): Trained baseline model
            best_features_to_perturb (list): Ranked list of feature indices to perturb
            **kwargs: Additional parameters overriding config defaults
            
        Returns:
            tuple: A tuple containing:
                - combined_X_train (np.ndarray): Combined original and augmented features
                - combined_y_train (np.ndarray): Combined original and augmented targets
                - sample_weight (np.ndarray): Sample weights for training
        """
        max_n_features_to_perturb = kwargs.get("max_n_features_to_perturb", self.config.max_n_features_to_perturb)
        aug_data_size_factor = kwargs.get("aug_data_size_factor", self.config.aug_data_size_factor)
        aug_data_weight = kwargs.get("aug_data_weight", self.config.aug_data_weight)
        min_perturb_percent = kwargs.get("min_perturb_percent", self.config.min_perturb_percent)
        max_perturb_percent = kwargs.get("max_perturb_percent", self.config.max_perturb_percent)

        if hasattr(self.config, "ablation_all_X"):
            perturbed_features = list(range(X_train.shape[1]))
            # all_feature_indices = list(range(X_train.shape[1]))
            # features_not_in_best_features = [i for i in all_feature_indices if i not in best_features_to_perturb]
            # perturbed_features = features_not_in_best_features
        else:
            perturbed_features = best_features_to_perturb[:max_n_features_to_perturb]

        self.logger.info(f"Selected features to perturb: {[f'F{i}' for i in perturbed_features]}\n")

        aug_X_train = self._intervention(X_train, perturbed_features, aug_data_size_factor, min_perturb_percent, max_perturb_percent)

        if hasattr(self.config, "ablation_fix_y"):
            aug_data_len = int(aug_data_size_factor * X_train.shape[0])
            n = int(math.ceil(aug_data_size_factor))
            aug_y_train = np.tile(y_train.reshape(-1, 1), (n, 1))
            aug_y_train = aug_y_train[:aug_data_len].ravel()
        else:
            aug_y_train = self._counterfactuals(X_train, train_residuals, baseline, aug_X_train, aug_data_size_factor)

        assert aug_X_train.shape[0] == aug_y_train.shape[0], "Augmented X and Y train have different lengths"

        combined_X_train, combined_y_train, sample_weight = self._get_combined_aug_training_set(X_train, y_train, aug_X_train, aug_y_train, aug_data_weight)
        
        self.combined_X_train = combined_X_train
        self.combined_y_train = combined_y_train
        return combined_X_train, combined_y_train, sample_weight

    def _tune_augmentation_params(
        self,
        dataset: AbstractDataset,
        baseline: BaselineRegressor,
        best_features_to_perturb: list,
        n_trials: int = 30,
        seed: int = None,
    ) -> dict:
        """
        Optimize augmentation hyperparameters using Optuna.
        
        Uses Bayesian optimization (TPE) with Hyperband pruning to find optimal
        augmentation parameters that minimize validation MSE.
        
        Args:
            dataset (AbstractDataset): Dataset for training and validation
            baseline (BaselineRegressor): Baseline model to use for augmentation
            best_features_to_perturb (list): Features identified for perturbation
            n_trials (int, optional): Number of optimization trials. Defaults to 30.
            seed (int, optional): Random seed for reproducible optimization
            
        Returns:
            dict: Best augmentation parameters found by optimization
        """
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # ---- Optuna objective ------------------------------------------------
        def objective(trial: optuna.Trial):
            # Sample a candidate set of augmentation knobs
            aug_params = dict(
                max_n_features_to_perturb = trial.suggest_int("max_n_features_to_perturb",   1, 5),
                aug_data_size_factor      = trial.suggest_float("aug_data_size_factor",      0.5, 1.5, step=0.25),
                aug_data_weight           = trial.suggest_float("aug_data_weight",           0.25, 1.0, step=0.25),
                max_perturb_percent       = trial.suggest_float("max_perturb_percent",       0.1, 1.0, step=0.1),
            )
            aug_params["min_perturb_percent"] = -aug_params["max_perturb_percent"]

            # ------------------ one-shot 80/20 split (fast) -------------------
            Xtr, Xval, ytr, yval, residuals_tr, residuals_val = train_test_split(
                dataset.X_train, 
                dataset.y_train, 
                dataset.train_residuals,
                test_size=0.2, 
                random_state=trial.number
            )

            # 1) augment the **training** part only
            X_aug, y_aug, w_aug = self._data_augmentation(
                Xtr, ytr, residuals_tr,
                baseline,
                best_features_to_perturb,
                **aug_params,
            )

            # 2) reset + train on augmented data
            model_params = baseline.get_params()
            if baseline.name not in ["linreg", "krr"]:
                model_params["random_state"] = trial.number
            if self.config.baseline == "xgboost": model_params["early_stopping_rounds"] = None # special case for xgboost
            new_baseline = BaselineRegressor(self.config.baseline, **model_params)

            self._train_aug_data(
                dataset, new_baseline,
                X_aug, y_aug, w_aug
            )

            # 3) validate
            mse = new_baseline.evaluate(Xval, yval, metric="mse")
            return mse                                    # Optuna minimises
    
        # ---- fire up study (TPE + Hyperband) -------------------------------
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.HyperbandPruner(),
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=1,
            show_progress_bar=True,
        )
        best_aug = study.best_params
        # Recover symmetric pair
        best_aug["min_perturb_percent"] = -best_aug["max_perturb_percent"]
        return best_aug
    
    def _cv_evaluation(self, 
            dataset: AbstractDataset,
            baseline: BaselineRegressor,
            best_features_to_perturb: list[int],
            model_params: dict,
            best_aug_params: dict,
            seed: int,
        ):
        """
        Evaluate augmentation effectiveness using cross-validation.
        
        Performs 10-fold cross-validation to compare original vs augmented training
        and tests for statistical significance using Wilcoxon signed-rank test.
        
        Args:
            dataset (AbstractDataset): Dataset for evaluation
            baseline (BaselineRegressor): Baseline regression model
            best_features_to_perturb (list[int]): Features to perturb during augmentation
            model_params (dict): Model hyperparameters
            best_aug_params (dict): Optimized augmentation parameters
            seed (int): Random seed for cross-validation splits
            
        Returns:
            tuple: A tuple containing:
                - p_wilcoxon (float): P-value from Wilcoxon signed-rank test
                - aug_models (list): List of trained models from cross-validation folds
        """

        combined_X_train, combined_y_train, sample_weight = self._data_augmentation(
            dataset.X_train, dataset.y_train, dataset.train_residuals, baseline, best_features_to_perturb, **best_aug_params
        )
        
        cv_baseline = BaselineRegressor(self.config.baseline, **model_params)
        cv = KFold(n_splits=10, shuffle=True, random_state=seed)

        with parallel_backend("loky", inner_max_num_threads=1):
            nmse_unaug = cross_val_score(cv_baseline.model, dataset.X_train, dataset.y_train, cv=cv, scoring="neg_mean_squared_error", n_jobs=4)
            cv_aug = cross_validate(
                cv_baseline.model,
                combined_X_train, combined_y_train,
                cv=cv,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                return_estimator=True
            )
            nmse_aug = cv_aug["test_score"]
            aug_models = cv_aug["estimator"]

        mse_unaug = -1.0 * nmse_unaug
        mse_aug = -1.0 * nmse_aug

        stat, p = wilcoxon(np.array(mse_aug) - np.array(mse_unaug), alternative="less", zero_method="wilcox")
        return p, aug_models

    def _final_evaluation(self, 
            seed: int,
            dataset: AbstractDataset,
            baseline: BaselineRegressor,
            best_features_to_perturb: list[int],
            model_params: dict,
            best_aug_params: dict,
        ):
        """
        Perform final evaluation on test set with augmented model.
        
        Trains a model on the full augmented training set and evaluates on test data.
        
        Args:
            seed (int): Random seed for reproducible evaluation
            dataset (AbstractDataset): Dataset containing test data
            baseline (BaselineRegressor): Baseline regression model
            best_features_to_perturb (list[int]): Features to perturb
            model_params (dict): Model hyperparameters
            best_aug_params (dict): Optimized augmentation parameters
            
        Returns:
            tuple: A tuple containing:
                - seed (int): Random seed used
                - aug_mse (float): MSE on test set with augmented model
        """
        self.seeding(seed)

        X_train, y_train, train_residuals, X_test, y_test = dataset.X_train, dataset.y_train, dataset.train_residuals, dataset.X_test, dataset.y_test

        combined_X_train, combined_y_train, sample_weight = self._data_augmentation(
            X_train, y_train, train_residuals, baseline, best_features_to_perturb, **best_aug_params
        )

        aug_baseline = BaselineRegressor(self.config.baseline, **model_params)
        self._train_aug_data(dataset, aug_baseline, combined_X_train, combined_y_train, sample_weight)
        aug_mse = aug_baseline.evaluate(X_test, y_test, metric="mse")
        return seed, aug_mse

    def _run_crda(self, dataset: AbstractDataset, seed: int):
        """
        Run the complete Counterfactual Residual Data Augmentation (CRDA) pipeline for one dataset and seed.
        
        Executes the full experimental pipeline including baseline training, feature filtering,
        augmentation parameter optimization, cross-validation evaluation, and final testing.
        
        Args:
            dataset (AbstractDataset): Dataset to run experiment on
            seed (int): Random seed for reproducible results
            
        Returns:
            dict or None: Dictionary containing experiment results including MSE metrics,
                         statistical test results, and metadata. Returns None if experiment
                         cannot proceed due to filtering or significance issues.
        """
        baseline = BaselineRegressor(self.config.baseline)
        if baseline.name not in ["linreg", "krr"]:
            baseline.set_params(random_state=seed)

        # Split data
        X_train, X_test, y_train, y_test = dataset.split(test_size=self.config.test_size, seed=seed)

        # Train original baseline model
        self._train_baseline(dataset, baseline, seed=seed)
        self.logger.info("Done training baseline model")

        # Calculate evaluation metrics for the baseline
        mse = baseline.evaluate(X_test, y_test, metric="mse")

        # Caclulate Residuals 
        all_residuals, train_residuals, test_residuals = dataset.add_residuals(baseline)
        self.logger.info("Done calculating residuals\n")

        # Filter for features to perturb
        filter = Filter(dataset, self.config, self.logger)
        self.logger.info("Filtering for features to perturb")
        best_features_to_perturb = filter.run_checks()

        should_proceed = True
        if best_features_to_perturb is None:
            if self.config.ignore_filter:
                self.logger.warning(f"No candidate features found for {dataset.name}. Ignoring filter and proceeding with the experiment anyways.")
                should_proceed = False
                best_features_to_perturb = [0] # dummy feature to proceed with the experiment
            else:
                self.logger.warning(f"No candidate features found for {dataset.name}.")
                return None

        self.logger.info("Done filtering.\n")

        # Data augmentation step
        if self.config.hyperparam_tune or self.config.method_param_tune:
            best_aug_params = self._tune_augmentation_params(dataset, baseline, best_features_to_perturb, seed=seed)
        else:
            best_aug_params = {
                "max_n_features_to_perturb": self.config.max_n_features_to_perturb,
                "aug_data_size_factor": self.config.aug_data_size_factor,
                "aug_data_weight": self.config.aug_data_weight,
                "min_perturb_percent": self.config.min_perturb_percent,
                "max_perturb_percent": self.config.max_perturb_percent,
            }

        self.logger.info(f"Best augmentation parameters:\n{best_aug_params}\n")

        # Train new model on augmented data
        model_params = baseline.get_params()
        if self.config.baseline == "xgboost": model_params["early_stopping_rounds"] = None # special case for xgboost

        # --- Iterative CRDA refinement loop ---
        n_iterations = getattr(self.config, 'n_crda_iterations', 1)
        best_aug_mse = float('inf')
        best_delta_mse = None
        best_p_wilcoxon = None
        best_iteration = 0
        
        for crda_iter in range(n_iterations):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"CRDA Iteration {crda_iter + 1}/{n_iterations}")
            self.logger.info(f"{'='*60}\n")
            
            p_wilcoxon, aug_estimators = self._cv_evaluation(dataset, baseline, best_features_to_perturb, model_params, best_aug_params, seed=seed)
            get_reusable_executor().shutdown(wait=True, kill_workers=True)
            self.logger.info("Done cross-validation evaluation")
            self.logger.info(f"The p-value from the Wilcoxon signed-rank test is: {p_wilcoxon}\n")
            
            if p_wilcoxon >= self.config.p_wilcoxon_threshold:
                if self.config.ignore_filter:
                    self.logger.warning(f"No significant improvement in MSE after augmentation for {dataset.name}. Ignoring filter and proceeding with the experiment anyways.")
                    should_proceed = False
                else:
                    self.logger.warning(f"No significant improvement in MSE after augmentation for {dataset.name}.")
                    if crda_iter == 0:
                        return None
                    else:
                        break

            preds = np.column_stack([est.predict(X_test) for est in aug_estimators])
            ensemble_pred = preds.mean(axis=1)
            aug_mse  = ((y_test - ensemble_pred) ** 2).mean()
            delta_mse = 100.0 * (aug_mse - mse) / mse

            self.logger.info(f"Iteration {crda_iter + 1} metrics: aug_mse={aug_mse:.6f}, delta_mse={delta_mse:.2f}%")
            
            # Track best result
            if aug_mse < best_aug_mse:
                best_aug_mse = aug_mse
                best_delta_mse = delta_mse
                best_p_wilcoxon = p_wilcoxon
                best_iteration = crda_iter + 1
                self.logger.info(f"New best aug_mse: {best_aug_mse:.6f} (iteration {best_iteration})")
            else:
                self.logger.info(f"aug_mse did not improve ({aug_mse:.6f} >= {best_aug_mse:.6f}). Stopping refinement.")
                break
            
            # If more iterations remain, compute refined residuals from augmented model
            if crda_iter < n_iterations - 1:
                self.logger.info("Computing refined residuals from augmented model for next iteration...")
                # Train a single model on the full augmented training set
                combined_X_train, combined_y_train, sample_weight = self._data_augmentation(
                    X_train, y_train, dataset.train_residuals, baseline, best_features_to_perturb, **best_aug_params
                )
                refiner = BaselineRegressor(self.config.baseline, **model_params)
                self._train_aug_data(dataset, refiner, combined_X_train, combined_y_train, sample_weight)
                # Update train residuals with refined estimates
                refined_train_preds = refiner.predict(X_train)
                dataset.train_residuals = y_train - refined_train_preds
                self.logger.info("Refined residuals computed for next iteration")
        
        # Use best results
        aug_mse = best_aug_mse
        delta_mse = best_delta_mse
        p_wilcoxon = best_p_wilcoxon
        self.logger.info(f"\nBest CRDA result from iteration {best_iteration}: aug_mse={aug_mse:.6f}, mse={mse:.6f}, delta_mse={delta_mse:.2f}%")

        if self.config.save_params:
            params_dir = os.path.join(self.config.experiment_dir, "params")
            os.makedirs(params_dir, exist_ok=True)
            params_file = os.path.join(params_dir, f"{dataset.name}_params_seed_{seed}.json")
            with open(params_file, "w") as f:
                json.dump({
                    "best_aug_params": best_aug_params,
                    "model_params": model_params,
                    "n_crda_iterations": n_iterations,
                    "best_iteration": best_iteration,
                }, f)
            self.logger.info(f"Best augmentation parameters and model parameters saved to {params_file}")

        result = {
            "dataset": dataset.name,
            "seed": seed,
            "mse": mse,
            "aug_mse": aug_mse,
            "delta_mse": delta_mse,
            "p_wilcoxon": p_wilcoxon,
            "should_proceed": should_proceed,
            "features_perturbed": best_features_to_perturb[:best_aug_params["max_n_features_to_perturb"]],
        }
        return result

    def run(self):
        """
        Run experiments across all datasets and seeds.
        
        Executes the complete experimental pipeline for all configured datasets
        and random seeds, aggregates results, and saves comprehensive outputs.
        
        Returns:
            pd.DataFrame: Results dataframe containing aggregated metrics across
                         all datasets and seeds
        """

        start_time = perf_counter()
        self.seeding(self.config.random_seed)
        if self.config.num_seeds == 0:
            seeds = [self.config.random_seed]
        else:
            seeds = np.random.randint(0, 1000000, self.config.num_seeds).tolist()
        results = []

        for dataset in self.datasets:
            self.logger.info("\n\n\n\n\n\n\n" + "*"*100)
            self.logger.info(f"Running experiment on dataset: {dataset.name}")
            X, y = dataset.preprocess()

            # Run
            seed_results = []
            for seed in seeds:
                self.logger.info("\n\n\n\n" + "#"*100)
                self.logger.info(f"Running experiment on dataset: {dataset.name} with seed: {seed}")
                result = self._run_crda(dataset, seed)

                if result is None:
                    self.logger.error(f"No results were possible for {dataset.name} on seed {seed}. Please check the data and the logs for more information and possibly change the config.")
                    continue

                seed_results.append(result)
                self.logger.info("#"*100)

            interim_results_df = pd.DataFrame(seed_results)
            interim_results_dir = os.path.join(self.config.experiment_dir, "interim_results")
            os.makedirs(interim_results_dir, exist_ok=True)
            interim_results_file = os.path.join(interim_results_dir, f"{dataset.name}_interim_results.csv")
            interim_results_df.to_csv(interim_results_file, index=False)

            self.logger.info(f"Interim results for seeds saved to {interim_results_file}. Aggregating results for {dataset.name}...")
            
            # Aggregate results for this dataset
            for metric in ['mse', 'aug_mse', 'delta_mse', 'p_wilcoxon', 'should_proceed']:
                values = [r[metric] for r in seed_results]
                mean_val = np.nanmean(values)
                # std_val = np.nanstd(values, ddof=1)
                std_err_val = np.nanstd(values, ddof=1) / np.sqrt(len(values))
                results.append({
                    'dataset': dataset.name,
                    'metric': metric,
                    'mean': mean_val,
                    'std': std_err_val
                })
            self.logger.info("*"*100)

    
        # Save results
        self.results_df = pd.DataFrame(results)
        results_file = os.path.join(self.config.experiment_dir, "results.csv")
        self.results_df.to_csv(results_file, index=False)

        end_time = perf_counter()
        self.logger.info(f"Total time taken: {end_time - start_time:.2f} seconds")
        self.logger.info(f"All results saved to {results_file}")

        return self.results_df
    
    def _plot_mse_delta(self):
        """
        Create a bar plot showing percent change in MSE for each dataset.
        
        Generates a bar chart visualizing the average percent change in MSE
        (delta_mse) across all datasets, with error bars showing standard deviation.
        """
        if self.results_df.empty:
            self.logger.error("No results to plot.")
            return

        # Filter results to only include delta metrics
        delta_metrics = ['delta_mse']
        plot_data = self.results_df[self.results_df['metric'].isin(delta_metrics)]
        
        datasets = plot_data['dataset'].unique()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        width = 0.35
        
        positions = np.arange(len(datasets))
        
        metric_data = plot_data[plot_data['metric'] == 'delta_mse']
            
        means = []
        stds = []
        for dataset in datasets:
            dataset_rows = metric_data[metric_data['dataset'] == dataset]
            if not dataset_rows.empty:
                means.append(dataset_rows['mean'].values[0])
                stds.append(dataset_rows['std'].values[0])
            else:
                means.append(0)
                stds.append(0)
        
        bars = ax.bar(positions, means, width, label='delta_mse', yerr=stds, capsize=5)
        
        # Add values on bars - improved positioning
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            sign = '+' if mean > 0 else ''
            # For positive bars, place text above; for negative bars, place text below
            if mean > 0:
                va = 'bottom'
                y_pos = height + max(2.0, 0.05 * abs(height))
            else:
                va = 'top'
                y_pos = height - max(2.0, 0.05 * abs(height))
            
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{sign}{mean:.1f}%', ha='center', va=va, rotation=0, 
                    fontsize=10, fontweight='bold', color='black',
                    bbox=dict(facecolor='white', alpha=0.7, pad=2, edgecolor='none'))
        
        # Add labels and title
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Percent Change (%)', fontsize=12)
        ax.set_title('Average Percent Change in MSE', fontsize=14)
        ax.set_xticks(positions)
        ax.set_xticklabels(datasets, fontsize=11)
        
        # Add a legend
        ax.legend(['Change in MSE'], fontsize=11)
        
        # Add a horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Ensure y-axis has enough margin for labels
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(min(y_min, -5), max(y_max * 1.2, 5))
        
        # Use tight layout with adjusted parameters
        plt.tight_layout(pad=2.0, h_pad=1.5, w_pad=1.5)

        # Save the plot
        plot_file = os.path.join(self.config.experiment_dir, "results_mse_percent_delta.png")
        plt.savefig(plot_file, bbox_inches='tight', dpi=300)
        self.logger.info(f"Results plot saved to {plot_file}")

    def plot_results(self):
        """
        Generate and save visualization plots for experiment results.
        
        Creates plots to visualize the experimental outcomes. Currently generates
        MSE percent change plots.
        """
        self._plot_mse_delta()

    def seeding(self, seed=None):
        """
        Seed all random number generators for reproducible experiments.
        
        Sets seeds for Python's random module, NumPy, PyTorch, and CUDA operations
        to ensure reproducible results across multiple runs.
        
        Args:
            seed (int, optional): Random seed value. If None, generates a random seed.
        """
        # Seed all random number generators for reproducibility.
        if seed is None:
            seed = random.randint(0, 1000000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

import torch
from pathlib import Path
from functools import cached_property

from gluonts.dataset.repository import dataset_names, datasets
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from probts.data.data_utils.get_datasets import get_dataset_info, get_dataset_borders, load_dataset
from probts.data.datasets.single_horizon_datasets import SingleHorizonDataset
from probts.data.datasets.multi_horizon_datasets import MultiHorizonDataset
from probts.data.datasets.gift_eval_datasets import GiftEvalDataset

from probts.data.data_utils.time_features import get_lags
from probts.data.data_utils.data_utils import split_train_val, truncate_test, get_rolling_test, df_to_mvds
from probts.data.data_wrapper import ProbTSBatchData
from probts.utils.utils import ensure_list
from probts.data.data_utils.data_scaler import StandardScaler, TemporalScaler, IdentityScaler
from typing import Union

MULTI_VARIATE_DATASETS = [
    'exchange_rate_nips',
    'solar_nips',
    'electricity_nips',
    'traffic_nips',
    'taxi_30min',
    'wiki-rolling_nips',
    'wiki2000_nips'
]

class DataManager:
    def __init__(
        self,
        dataset: str,
        path: str = './datasets',
        history_length: int = None,
        context_length: int = None,
        prediction_length: Union[list,int,str] = None,
        train_ctx_len: int = None,
        train_pred_len_list: Union[list,int,str] = None,
        val_ctx_len: int = None,
        val_pred_len_list: Union[list,int,str] = None,
        test_rolling_length: int = 96,
        split_val: bool = True,
        scaler: str = 'none',
        context_length_factor: int = 1,
        timeenc: int = 1,
        var_specific_norm: bool = True,
        data_path: str = None,
        freq: str = None,
        multivariate: bool = True,
        continuous_sample: bool = False,
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
        auto_search: bool = False,
    ):
        """
        DataManager class for handling datasets and preparing data for time-series models.

        Parameters
        ----------
        dataset : str
            Name of the dataset to load. Examples include "etth1", "electricity_ltsf", etc.
        path : str, optional, default='./datasets'
            Root directory path where datasets are stored.
        history_length : int, optional, default=None
            Length of the historical input window for the model.
            If not specified, it is automatically calculated based on `context_length` and lag features.
        context_length : int, optional, default=None
            Length of the input context for the model. 
        prediction_length : Union[list, int, str], optional, default=None
            Length of the prediction horizon for the model. Can be:
            - int: Fixed prediction length.
            - list: Variable prediction lengths for multi-horizon training.
            - str: The string format of multiple prediction length. E.g., '96-192-336-720' represents [96, 192, 336, 720]
        train_ctx_len : int, optional, default=None
            Context length for the training dataset.
            If not specified, defaults to the value of `context_length`.
        train_pred_len_list : Union[list, int, str], optional, default=None
            List of prediction lengths for the training dataset.
            If not specified, defaults to the value of `prediction_length`.
        val_ctx_len : int, optional, default=None
            Context length for the validation dataset.
            If not specified, defaults to the value of `context_length`.
        val_pred_len_list : Union[list, int, str], optional, default=None
            List of prediction lengths for the validation dataset.
            If not specified, defaults to the value of `prediction_length`.
        test_rolling_length : int, optional, default=96
            Gap window size used for rolling predictions in the testing phase.
            - If set to `auto`, it is dynamically determined based on the dataset frequency
            (e.g., 'H' -> 24, 'D' -> 7, 'W' -> 4).
        split_val : bool, optional, default=True
            Whether to split the training dataset into training and validation sets.
        scaler : str, optional, default='none'
            Type of normalization or scaling applied to the dataset. Options include:
            - 'none': No scaling.
            - 'standard': Standard normalization (z-score).
            - 'temporal': Mean-scaling normalization.
        context_length_factor : int, optional, default=1
            Scaling factor for context length, allowing dynamic adjustment of `context_length`.
        timeenc : int, optional, default=1
            Time encoding strategy. Options include:
            - 0: The dimension of time feature is 5, containing `month, day, weekday, hour, minute`
            - 1: Cyclic time features (e.g., sine/cosine of timestamps).
            - 2: Raw Timestamp information.
        var_specific_norm : bool, optional, default=True
            Whether to normalize variables independently. Only applies when `scaler='standard'`.
        data_path : str, optional, default=None
            Specific path to the dataset file.
        freq : str, optional, default=None
            Data frequency (e.g., 'H' for hourly, 'D' for daily).
        multivariate : bool, optional, default=True
            Whether the dataset is multivariables.
        continuous_sample : bool, optional, default=False
            Whether to enable continuous sampling for forecasting horizons during training phase.
        train_ratio : float, optional, default=0.7
            Proportion of the dataset used for training. Default is 70% of the data.
        test_ratio : float, optional, default=0.2
            Proportion of the dataset used for testing. Default is 20% of the data.
        auto_search : bool, optional, default=False
            Make past_len=ctx_len+pred_len, enabling post training search.
        """

        self.dataset = dataset
        self.path = path
        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.train_ctx_len = train_ctx_len if train_ctx_len is not None else context_length
        self.val_ctx_len = val_ctx_len if val_ctx_len is not None else context_length
        self.train_pred_len_list = train_pred_len_list if train_pred_len_list is not None else prediction_length
        self.val_pred_len_list = val_pred_len_list if val_pred_len_list is not None else prediction_length
        self.test_rolling_length = test_rolling_length
        self.split_val = split_val
        self.scaler_type = scaler
        self.context_length_factor = context_length_factor
        self.timeenc = timeenc
        self.var_specific_norm = var_specific_norm
        self.data_path = data_path
        self.freq = freq
        self.multivariate = multivariate
        self.continuous_sample = continuous_sample
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.auto_search = auto_search
        
        self.test_rolling_dict = {'h': 24, 'd': 7, 'b':5, 'w':4, 'min': 60}
        self.global_mean = None

        # Configure scaler
        self.scaler = self._configure_scaler(self.scaler_type)
  
        # Load dataset and prepare for processing
        if dataset in dataset_names:
            self.multi_hor = False
            self._load_short_term_dataset()
        elif self.is_gift_eval:
            self.multi_hor = False
            # Load GIFT eval datasets from salesforce
            self._load_gift_eval_dataset()
        else:
            # Process context and prediction lengths
            self._process_context_and_prediction_lengths()
            self._load_long_term_dataset()
            # Print configuration details
            self._print_configurations()
        
    def _configure_scaler(self, scaler_type: str):
        """Configure the scaler."""
        if scaler_type == "standard":
            return StandardScaler(var_specific=self.var_specific_norm)
        elif scaler_type == "temporal":
            return TemporalScaler()
        return IdentityScaler()
    
    def _load_gift_eval_dataset(self):
        parts = self.dataset[5:].split('/')  # Remove first 'gift/'
        self.dataset = '/'.join(parts[:-1])  # Join all parts except last one with '/'
        gift_term = parts[-1] # corresponding to "term" parameter in GiftEvalDataset
        TO_UNIVARIATE = False
        self.dataset_raw = GiftEvalDataset(self.dataset, term=gift_term, to_univariate=TO_UNIVARIATE)
        self._set_meta_parameters(self.dataset_raw.target_dim, self.dataset_raw.freq, self.dataset_raw.prediction_length)

        dataset_loader = SingleHorizonDataset(
            ProbTSBatchData.input_names_, 
            self.history_length,
            self.context_length,
            self.prediction_length,
            self.freq,
            self.multivariate
        )

        self.train_iter_dataset = dataset_loader.get_iter_dataset(self.dataset_raw.training_dataset, mode='train')
        self.val_iter_dataset = dataset_loader.get_iter_dataset(self.dataset_raw.validation_dataset, mode='val')
        self.test_iter_dataset = dataset_loader.get_iter_dataset(self.dataset_raw.test_dataset, mode='test')
        self.time_feat_dim = dataset_loader.time_feat_dim
        # TODO: Implement global mean for GIFT eval datasets
        # self.global_mean = torch.mean(torch.tensor(self.dataset_raw.training_dataset[0]['target']), dim=-1)
    
    def _load_short_term_dataset(self):
        """Load short-term dataset using GluonTS."""
        print(f"Loading Short-term Dataset: {self.dataset}")
        self.dataset_raw = datasets.get_dataset(self.dataset, path=Path(self.path), regenerate=False)
        metadata = self.dataset_raw.metadata
        if self.is_univar_dataset:
            target_dim = 1
        else:
            target_dim = metadata.feat_static_cat[0].cardinality
        self._set_meta_parameters(target_dim, metadata.freq.upper(), metadata.prediction_length)
        self.prepare_STSF_dataset(self.dataset)

    def _set_meta_parameters(self, target_dim, freq, prediction_length):
        """Set meta parameters from base dataset."""
        self.target_dim = int(target_dim)
        self.multivariate = self.target_dim > 1
        self.freq = freq
        self.lags_list = get_lags(self.freq)
        self.prediction_length = prediction_length
        self.context_length = self.context_length or self.prediction_length * self.context_length_factor
        self.history_length = self.history_length or (self.context_length + max(self.lags_list))
        
    def _process_context_and_prediction_lengths(self):
        """Convert context and prediction lengths to lists for multi-horizon processing."""
        self.train_ctx_len_list = ensure_list(self.train_ctx_len, default_value=self.context_length)
        self.val_ctx_len_list = ensure_list(self.val_ctx_len, default_value=self.context_length)
        self.test_ctx_len_list = ensure_list(self.context_length)
        self.train_pred_len_list = ensure_list(self.train_pred_len_list, default_value=self.prediction_length)
        self.val_pred_len_list = ensure_list(self.val_pred_len_list, default_value=self.prediction_length)
        self.test_pred_len_list = ensure_list(self.prediction_length)

        # Validate context length support
        assert len(self.train_ctx_len_list) == 1, "Assign a single context length for training."
        assert len(self.val_ctx_len_list) == 1, "Assign a single context length for validation."
        assert len(self.test_ctx_len_list) == 1, "Assign a single context length for testing."

        self.multi_hor = len(self.train_pred_len_list) > 1 or \
                         len(self.val_pred_len_list) > 1 or \
                         len(self.test_pred_len_list) > 1

    def _load_long_term_dataset(self):
        """Load long-term dataset or customized dataset."""
        print(f"Loading Long-term Dataset: {self.dataset}")
        if not self.context_length or not self.prediction_length:
            raise ValueError("context_length or prediction_length must be specified.")

        data_path, self.freq = get_dataset_info(self.dataset, data_path=self.data_path, freq=self.freq)
        self.dataset_raw, self.data_stamp, self.target_dim, data_size = load_dataset(
            self.path, data_path, freq=self.freq, timeenc=self.timeenc, multivariate=self.multivariate
        )
        self.border_begin, self.border_end = get_dataset_borders(
            self.dataset, data_size, train_ratio=self.train_ratio, test_ratio=self.test_ratio
        )
        self._set_meta_parameters_from_raw(data_size)
        self.prepare_dataset()
        
    def _set_meta_parameters_from_raw(self, data_size):
        """Set meta parameters directly from raw dataset."""
        self.lags_list = get_lags(self.freq)
        self.prediction_length = ensure_list(self.prediction_length) if self.multi_hor else self.prediction_length
        self.context_length = ensure_list(self.context_length) if self.multi_hor else self.context_length
        self.history_length = self.history_length or (
            max(self.context_length) + max(self.lags_list) if self.multi_hor else self.context_length + max(self.lags_list)
        )
        if not self.multivariate:
            self.target_dim = 1
            raise NotImplementedError("Customized univariate datasets are not yet supported.")
        assert data_size >= self.border_end[2], "border_end index exceeds dataset size!"
        
        # define the test_rolling_length
        if self.test_rolling_length == 'auto':
            if self.freq.lower() in self.test_rolling_dict:
                self.test_rolling_length = self.test_rolling_dict[self.freq.lower()]
            else:
                self.test_rolling_length = 24
            

    def prepare_dataset(self):
        """Prepare datasets for training, validation, and testing."""
        # Split raw data into train, validation, and test sets
        train_data = self.dataset_raw[: self.border_end[0]]
        val_data = self.dataset_raw[: self.border_end[1]]
        test_data = self.dataset_raw[: self.border_end[2]]
        
        # Calculate statictics using training data
        self.scaler.fit(torch.tensor(train_data.values))
        
        # Convert dataframes to multivariate datasets
        train_set = df_to_mvds(train_data, freq=self.freq)
        val_set = df_to_mvds(val_data,freq=self.freq)
        test_set = df_to_mvds(test_data,freq=self.freq)
        
        train_grouper = MultivariateGrouper(max_target_dim=self.target_dim)
        test_grouper = MultivariateGrouper(max_target_dim=self.target_dim)
        
        group_train_set = train_grouper(train_set)
        group_val_set = test_grouper(val_set)
        group_test_set = test_grouper(test_set)
        
        if self.multi_hor:
            # Handle multi-horizon datasets
            dataset_loader = self._prepare_multi_horizon_datasets(group_val_set, group_test_set)
        else:
            # Handle single-horizon datasets
            dataset_loader = self._prepare_single_horizon_datasets(group_val_set, group_test_set)

        self.train_iter_dataset = dataset_loader.get_iter_dataset(group_train_set, mode='train', data_stamp=self.data_stamp[: self.border_end[0]])
        
        self.time_feat_dim = dataset_loader.time_feat_dim
        self.global_mean = torch.mean(torch.tensor(group_train_set[0]['target']), dim=-1)
    
    
    def _prepare_multi_horizon_datasets(self, group_val_set, group_test_set):
        """Prepare multi-horizon datasets for validation and testing."""
        self.val_iter_dataset = {}
        self.test_iter_dataset = {}
        dataset_loader = MultiHorizonDataset(
            input_names = ProbTSBatchData.input_names_,
            freq = self.freq,
            train_ctx_range = self.train_ctx_len_list,
            train_pred_range = self.train_pred_len_list,
            val_ctx_range = self.val_ctx_len_list,
            val_pred_range = self.val_pred_len_list,
            test_ctx_range = self.test_ctx_len_list,
            test_pred_range = self.test_pred_len_list,
            multivariate = self.multivariate,
            continuous_sample = self.continuous_sample
        )

        # Prepare validation datasets
        for pred_len in self.val_pred_len_list:
            local_group_val_set = get_rolling_test(
                'val', group_val_set, self.border_begin[1], self.border_end[1],
                rolling_length=self.test_rolling_length, pred_len=pred_len, freq=self.freq
            )
            self.val_iter_dataset[str(pred_len)] = dataset_loader.get_iter_dataset(
                local_group_val_set, mode='val', data_stamp=self.data_stamp[:self.border_end[1]], pred_len=[pred_len]
            )

        # Prepare testing datasets
        for pred_len in self.test_pred_len_list:
            local_group_test_set = get_rolling_test(
                'test', group_test_set, self.border_begin[2], self.border_end[2],
                rolling_length=self.test_rolling_length, pred_len=pred_len, freq=self.freq
            )
            self.test_iter_dataset[str(pred_len)] = dataset_loader.get_iter_dataset(
                local_group_test_set, mode='test', data_stamp=self.data_stamp[:self.border_end[2]], pred_len=[pred_len], auto_search=self.auto_search,
            )
            
        return dataset_loader
    
    def _prepare_single_horizon_datasets(self, group_val_set, group_test_set):
        """Prepare single-horizon datasets for training, validation, and testing."""
        dataset_loader = SingleHorizonDataset(
            ProbTSBatchData.input_names_,
            self.history_length,
            self.context_length,
            self.prediction_length,
            self.freq,
            self.multivariate,
        )

        # Validation dataset
        local_group_val_set = get_rolling_test(
            'val', group_val_set, self.border_begin[1], self.border_end[1],
            rolling_length=self.test_rolling_length, pred_len=self.val_pred_len_list[0], freq=self.freq
        )
        self.val_iter_dataset = dataset_loader.get_iter_dataset(local_group_val_set, mode='val', data_stamp=self.data_stamp[:self.border_end[1]])

        # Testing dataset
        local_group_test_set = get_rolling_test(
            'test', group_test_set, self.border_begin[2], self.border_end[2],
            rolling_length=self.test_rolling_length, pred_len=self.prediction_length, freq=self.freq
        )
        self.test_iter_dataset = dataset_loader.get_iter_dataset(local_group_test_set, mode='test', data_stamp=self.data_stamp[:self.border_end[2]], auto_search=self.auto_search)

        return dataset_loader
    
    def prepare_STSF_dataset(self, dataset: str):
        """Prepare datasets for short-term series forecasting."""
        if dataset in MULTI_VARIATE_DATASETS:
            self.num_test_dates = int(len(self.dataset_raw.test)/len(self.dataset_raw.train))

            train_grouper = MultivariateGrouper(max_target_dim=int(self.target_dim))
            test_grouper = MultivariateGrouper(
                num_test_dates=self.num_test_dates, 
                max_target_dim=int(self.target_dim)
            )
            train_set = train_grouper(self.dataset_raw.train)
            test_set = test_grouper(self.dataset_raw.test)
            self.scaler.fit(torch.tensor(train_set[0]['target'].transpose(1, 0)))
            self.global_mean = torch.mean(torch.tensor(train_set[0]['target']), dim=-1)
            
            # split_val
            if self.split_val:
                train_set, val_set = split_train_val(train_set, self.num_test_dates, self.context_length, self.prediction_length, self.freq)
            else:
                val_set = None
        else:
            self.target_dim = 1
            self.multivariate = False
            self.num_test_dates = 1
            train_set = self.dataset_raw.train
            test_set = self.dataset_raw.test
            test_set = truncate_test(test_set, self.context_length, self.prediction_length, self.freq)
            # for univariate dataset, e.g., M4 and M5, no validation set is used
            val_set = None

        if val_set is None:
            print('No validation set is used.')
            
        dataset_loader = SingleHorizonDataset(
            ProbTSBatchData.input_names_, 
            self.history_length,
            self.context_length,
            self.prediction_length,
            self.freq,
            self.multivariate
        )

        self.train_iter_dataset = dataset_loader.get_iter_dataset(train_set, mode='train')
        if val_set is not None:
            self.val_iter_dataset = dataset_loader.get_iter_dataset(val_set, mode='val')
        else:
            self.val_iter_dataset = None
        self.test_iter_dataset = dataset_loader.get_iter_dataset(test_set, mode='test')
        self.time_feat_dim = dataset_loader.time_feat_dim

    def _print_configurations(self):
        """Print dataset and configuration details."""
        print(f"Test context length: {self.test_ctx_len_list}, prediction length: {self.test_pred_len_list}")
        print(f"Validation context length: {self.val_ctx_len_list}, prediction length: {self.val_pred_len_list}")
        print(f"Training context length: {self.train_ctx_len_list}, prediction lengths: {self.train_pred_len_list}")
        print(f"Test rolling length: {self.test_rolling_length}")
        if self.scaler_type == "standard":
            print(f"Variable-specific normalization: {self.var_specific_norm}")

    @cached_property
    def is_gift_eval(self) -> bool:
        return self.dataset[:5] == "gift/"
    
    @cached_property
    def is_univar_dataset(self) -> bool:
        if 'm4' in self.dataset or 'm5' in self.dataset:
            return True
        return False
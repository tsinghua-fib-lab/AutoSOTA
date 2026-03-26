import os
import os.path as osp
import numpy as np
import torch
import logging
import io
import pickle

from pathlib import Path
from typing import List, Tuple, Dict, Union, Literal
from datetime import datetime
from aeon.datasets import load_classification
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from torch.utils.data import random_split, Dataset
from sklearn.preprocessing import StandardScaler

from torch import Tensor

from .base import BaseDataset, BaseDataSubset

from src.utils import DatasetOptions, TrainingMethodOptions
import src.utils.datasets.irregular_sampling as utils
import boto3


__all__ = [
    "P12Dataset"
]


class P12SubDataset(BaseDataSubset):
    """
    A dataset class for handling P19 data subsets.

    This class inherits from `BaseDataSubset` and is specifically designed to handle
    subsets of P19 data. It initializes the dataset with a list of `P12Data` objects.

    Attributes:
        data (List[utils.P12Data]): A list of P12Data objects.

    Methods:
        __init__(data: List[utils.P12Data]) -> None:
            Initializes the dataset with the provided P19 data.
    """

    def __init__(self, data: List[utils.P12Data]) -> None:
        """
        Initializes the dataset with the provided P19 data.

        Args:
            data (List[utils.P12Data]): A list of P12Data objects.
        """
        super().__init__(data)


class P12Dataset(BaseDataset):
    """
    P12Dataset class for handling P19 data.

    This class inherits from BaseDataset and provides functionalities to load,
    process, and split the P19 data for training, validation, and testing purposes.

    Attributes:
        __url (str): URL to download the P19 dataset.
        __raw_zip_filename (str): Filename for the raw P19 dataset zip file.
        __raw_file_names (dict): Dictionary containing filenames for raw data, labels, and irregularity mask.
        __config_filename (str): Filename for the configuration file.
        __splits (dict): Dictionary containing filenames for different data splits.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the data at the specified index.
        download(**kwargs): Downloads the dataset.
        process(**kwargs): Processes the dataset.
        _create_irreg_data(self, irreg_mask_p: float=0.6): Processes sequences to create irregularity in Sequence.
        process_centralized(instance, **kwargs): Processes centralized data.
        process_centralized_load_from_config(instance, config, **kwargs): Processes centralized data from configuration.
        _check_if_centralized_dataset_exists(load_from_config, ds_config, tmp_config): Checks if a centralized dataset configuration exists.
        _log_stats(**kwargs): Logs dataset statistics.
        setup_aws_session(): Sets up AWS required attributes.
        _setup_transforms(): Sets up data transforms.
        scale_data(test_indices=None): Scales the data.
    """

    __url = "https://figshare.com/ndownloader/files/34683085"
    __raw_zip_filename = "P12_raw.zip"
    __raw_file_names = {
        'data': 'PTdict_list.pkl',
        'labels': 'arr_outcomes.pkl',
        'ts_params': 'ts_params.pkl',
    }
    __config_filename = 'config.json'
    __splits = {
        0: 'phy12_split1.pkl',
        1: 'phy12_split2.pkl',
        2: 'phy12_split3.pkl',
        3: 'phy12_split4.pkl',
        4: 'phy12_split5.pkl'
    }

    __max_tmins = 48*60 # 48 hours
    __param_detailed_description = {
        "ALP": "Alkaline phosphatase (IU/L)",
        "ALT": "Alanine transaminase (IU/L)",
        "AST": "Aspartate transaminase (IU/L)",
        "Albumin": "Albumin (g/dL)",
        "BUN": "Blood urea nitrogen (mg/dL)",
        "Bilirubin": "Bilirubin (mg/dL)",
        "Cholesterol": "Cholesterol (mg/dL)",
        "Creatinine": "Serum creatinine (mg/dL)",
        "DiasABP": "Invasive diastolic arterial blood pressure (mmHg)",
        "FiO2": "Fractional inspired O2 (0-1)",
        "GCS": "Glasgow Coma Score (3-15)",
        "Glucose" :"Serum glucose (mg/dL)",
        "HCO3": "Serum bicarbonate (mmol/L)",
        "HCT": "Hematocrit (%)",
        "HR": "Heart rate (bpm)",
        "K": "Serum potassium (mEq/L)",
        "Lactate": "Lactate (mmol/L)",
        "MAP": "Invasive mean arterial blood pressure (mmHg)",
        "MechVent": "Mechanical ventilation respiration (0:false, or 1:true)",
        "Mg": "Serum magnesium (mmol/L)",
        "NIDiasABP": "Non-invasive diastolic arterial blood pressure (mmHg)",
        "NIMAP": "Non-invasive mean arterial blood pressure (mmHg)",
        "NISysABP": "Non-invasive systolic arterial blood pressure (mmHg)",
        "Na": "Serum sodium (mEq/L)", 
        "PaCO2": "partial pressure of arterial CO2 (mmHg)",
        "PaO2": "Partial pressure of arterial O2 (mmHg)",
        "Platelets": "Platelets(cells/nL)",
        "RespRate": "Respiration rate (bpm)",
        "SaO2": "O2 saturation in hemoglobin (%)",
        "SysABP": "Invasive systolic arterial blood pressure (mmHg)",
        "Temp": "Temperature (°C)",
        "TroponinI": "Troponin-I (μg/L)",
        "TroponinT": "Troponin-T (μg/L)",
        # "TropI": "Troponin-I (μg/L)",
        # "TropT": "Troponin-T (μg/L)",
        # "TroponinI": "Troponin-I (μg/L)",
        "Urine": "Urine output (mL)",
        "WBC": "White blood cell count (cells/nL)",
        "pH": "Arterial pH (0-14)",
    }
    

    def __init__(self,
        seed: int, 
        training_method: str=None,
        train_split_index: int=0,
        aws_profile: str=None, 
        s3_bucket_path: Union[str, Path, os.PathLike[str]]=None,
        use_threads: bool = False,
        preprocessing_method: Literal['raindrop', 'vitst']=None,
        min_seq_length: int=None,
        percentile_of_features_used: int=None,
        balance: Literal['random', 'smote']=None,
        upsample_percentage: float=0.0,
        **kwargs):
        """
        Args:
            seed (int): Random seed for reproducibility.
            training_method (str, optional): Method for training. Defaults to None.
            train_split_index (int, optional): Index of the training split to use. Defaults to 0.
            aws_profile (str, optional): AWS profile for accessing S3. Defaults to None.
            s3_bucket_path (Union[str, Path, os.PathLike[str]], optional): S3 bucket path for loading data. Defaults to None.
            use_threads (bool, optional): Whether to use threads for loading data. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        self._dataset_class = P12SubDataset
        self.training_method = training_method
        self.train_split_index = train_split_index
        if train_split_index > 4:
            raise ValueError(f'{train_split_index=}, must be < 5, picke between 0,1,2,3,4')
        self.aws_profile = aws_profile
        self.s3_bucket_path = s3_bucket_path
        self.use_threads = use_threads
        self._preprocessing_method = 'raindrop' if preprocessing_method is None else preprocessing_method
        self._min_seq_length = min_seq_length
        self._percentile_of_features_used = percentile_of_features_used
        self._balance = balance
        self._upsample_percentage = upsample_percentage
        assert 0.0 <= upsample_percentage <= 1.0

        self.setup_aws_session()
        self._setup_transforms()

        super().__init__(dataset_name=DatasetOptions.p12, seed=seed, **kwargs)
    
    def __len__(self,):
        return len(self._targets)
    
    def __getitem__(self, index) -> utils.P12Data:
        return self._data[index]
    
    def download(self, **kwargs):
        # 
        self.dataset_dir_paths['splits'] = osp.join(self.dataset_dir_paths['raw'], 'splits')
        self.dataset_dir_paths['processed_data'] = osp.join(self.dataset_dir_paths['raw'], 'processed_data')

        now = datetime.now()
        if self.training_method.lower() == TrainingMethodOptions.centralized:
            self.dataset_dir_paths['centralized_instance'] = osp.join(self.dataset_dir_paths['centralized'], now.strftime("%Y-%m-%d_%H:%M:%S"))
        elif self.training_method.lower() == TrainingMethodOptions.federated:
            self.dataset_dir_paths['federated_instance'] = osp.join(self.dataset_dir_paths['federated'], now.strftime("%Y-%m-%d_%H:%M:%S"))

        if self.aws_profile is not None or self.s3_bucket_path is not None: # load data from aws bucket
            self.cli_logger.info(f'Downloading {DatasetOptions.p12} from S3 Bucket {self.s3_bucket_path}!')
            
            prefix_X = f"{self.prefix}/raw/processed_data/{self.__raw_file_names['data']}"
            prefix_y = f"{self.prefix}/raw/processed_data/{self.__raw_file_names['labels']}"
            prefix_splits = f"{self.prefix}/raw/splits/{self.__splits[self.train_split_index]}"
            
            self.cli_logger.info('Loading Data')
            s3_object_X = self.s3_client.get_object(Bucket=self.s3_bucket_path, Key=prefix_X)
            self._X = pickle.loads(s3_object_X.get("Body").read())

            self.cli_logger.info('Loading Labels')
            s3_object_y = self.s3_client.get_object(Bucket=self.s3_bucket_path, Key=prefix_y)
            self._y = pickle.loads(s3_object_y.get("Body").read())
            
            self.cli_logger.info('Loading Splits')
            s3_object_splits = self.s3_client.get_object(Bucket=self.s3_bucket_path, Key=prefix_splits)
            self._splits = pickle.loads(s3_object_splits.get("Body").read())
            self.cli_logger.info('Done!')

            self._data_raw = (self._X, self._y)
        
        else:
            if not osp.exists(osp.join(self.dataset_dir_paths['raw'], self.__raw_zip_filename)):
                # Downloading data 
                self.cli_logger.info(f'Downloading raw {DatasetOptions.p12} dataset from {self.__url}!')
                utils.download(
                    dataset_name=DatasetOptions.p12,
                    url=self.__url,
                    folder_path=self.dataset_dir_paths['raw'],
                    file_name=self.__raw_zip_filename,
                    logger=self.cli_logger
                )
                # rename splits files data and labels files 
                check_converted_to_pkl = [f for f in os.listdir(osp.join(
                self.dataset_dir_paths['processed_data'])) if f.endswith('.pkl') and \
                    f.split('/')[-1].split('.')[0] in [k.split('.')[0] for k in self.__raw_file_names.values()]]
                
                if not bool(check_converted_to_pkl):
                    for f in os.listdir(self.dataset_dir_paths['processed_data']):
                        if f.split('/')[-1].replace('npy', 'pkl') in list(self.__raw_file_names.values()):
                            f_arr = np.load(
                                osp.join(self.dataset_dir_paths['processed_data'], f), allow_pickle=True)
                            
                            self.store_asset(format='pickle', 
                                        file_path=osp.join(self.dataset_dir_paths['processed_data'], f.replace('.npy', '.pkl')),
                                        obj=f_arr, write='wb')
                    for f in os.listdir(self.dataset_dir_paths['splits']):
                        if f.split('/')[-1].replace('npy', 'pkl') in list(self.__splits.values()):
                            splits_arr = np.load(
                            osp.join(self.dataset_dir_paths['splits'], f), allow_pickle=True)
                            splits_dict = {
                                'train' : splits_arr[0],
                                'val' : splits_arr[1],
                                'test' : splits_arr[2]
                            }
                            self.store_asset(format='pickle', 
                                            file_path=osp.join(self.dataset_dir_paths['splits'], f.replace('.npy', '.pkl')),
                                            obj=splits_dict, write='wb')
            else:
                self.cli_logger.info(f'Found raw {DatasetOptions.p12} dataset!')
            
            self._X = self.load_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['processed_data'], self.__raw_file_names['data']), read='rb')
            self._y = self.load_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['processed_data'], self.__raw_file_names['labels']), read='rb')[:, -1].reshape(-1,1)

            self._splits = self.load_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['splits'], self.__splits[self.train_split_index]), read='rb')
        
            self._data_raw = (self._X, self._y)
    
    def process(self, **kwargs):
        if self.training_method.lower() == TrainingMethodOptions.centralized:
            #self.dataset_dir_paths['data'] = osp.join(self.dataset_dir_paths['centralized_instance'], self.__data_filename)
            self.dataset_dir_paths['config'] = osp.join(self.dataset_dir_paths['centralized_instance'], self.__config_filename)

            if self.aws_profile is not None or self.s3_bucket_path is not None:
                load_from_config, ds_config = self.check_if_centralized_dataset_exists_on_s3(
                    s3_client=self.s3_client, s3_buckets=self.s3_buckets
                )
            else:   
                load_from_config, ds_config = self.check_if_centralized_dataset_exists(**kwargs)
            
            if load_from_config:
                instance = ds_config['instance']
                params = {'instance': instance, 'config': ds_config}
                self.process_centralized_load_from_config(**params)
            else:
                instance = self.dataset_dir_paths['centralized_instance'].split("/")[-1]
                ds_config = self.process_centralized(instance=instance, **kwargs)
        
        self._max_seq_len = 0
        for sample in self._data:
            if self._max_seq_len <= sample.data.size(0):
                self._max_seq_len = sample.data.size(0)
        
        self._ds_config = ds_config
        self._log_stats(**{
            'method': TrainingMethodOptions.centralized, 
            'cli_logger': self.cli_logger,
            'train_dataset': self.train_dataset, 
            'val_dataset': self.val_dataset,
            'test_dataset': self.test_dataset,
        })
    
    def process_shared(self,):
        """ Shared processing function for load from config and process. """
        if self._preprocessing_method == 'raindrop':
            self.cli_logger.info('Using Raindrop Processing!')
            self.raindrop_processing()
        elif self._preprocessing_method == 'vitst':
            self.vitst_processing()
            self.cli_logger.info('Using ViTST Processing!')
        else:
            raise ValueError(f'{self._preprocessing_method} not implemented. Use "raindrop", "vitst"! ')
        
        self._data = self._train_data + self._val_data + self._test_data

        self.train_dataset = P12SubDataset(data=self._train_data)
        self.val_dataset = P12SubDataset(data=self._val_data)
        self.test_dataset = P12SubDataset(data=self._test_data)

        # balance training sets
        if self._balance == 'random' and self._upsample_percentage != 0.0:
            X_train = [sample for sample in self.train_dataset]
            y_train = [sample.label for sample in self.train_dataset]
            
            unique, counts = np.unique(y_train, return_counts=True)
            
            minority_class_idx = np.argmin(counts)
            majority_class_idx = np.argmax(counts)
            minority_class_indices = np.where(y_train == unique[minority_class_idx])[0]
            assert len(minority_class_indices) == counts[minority_class_idx]
            
            n_samples_to_generate = int((counts[majority_class_idx] - counts[minority_class_idx]) * self._upsample_percentage)
            #print(n_samples_to_generate, counts[majority_class_idx] - counts[minority_class_idx])
            additional_samples = np.random.choice(minority_class_indices, size=n_samples_to_generate, replace=True)
            X_train_resampled = X_train + [X_train[idx] for idx in additional_samples]
            # reinitialize datastet
            self.train_dataset = P12SubDataset(data=X_train_resampled)
        elif self._balance == 'random' and self._upsample_percentage == 0.0:
            pass
        elif self._balance == 'smote' and self._upsample_percentage != 0.0:
            raise NotImplementedError
        elif self._balance is None:
            pass
        else:
            raise NotImplementedError(f'{self._balance=} and {self._upsample_percentage=}')

    def process_centralized(self, instance: str, **kwargs):
        self.cli_logger.info(f'Processing ...!')
        config = {
            'instance': instance,
            'training_method': TrainingMethodOptions.centralized,
            'train_split_index': str(self.train_split_index),
        }
        if not osp.exists(osp.join(self.dataset_dir_paths['centralized_instance'])):
            os.makedirs(osp.join(self.dataset_dir_paths['centralized_instance']))
        
        # processing
        self.process_shared()
        
        # store data and config
        config['indices'] = {
            'train': str(list(self._splits['train'])),
            'val': str(list(self._splits['val'])),
            'test': str(list(self._splits['test'])),
        }
        
        if not osp.exists(self.dataset_dir_paths["centralized_instance"]):
            os.makedirs(self.dataset_dir_paths["centralized_instance"])
        
        #self.store_asset(format='pickle', file_path=self.dataset_dir_paths['data'], obj=self._data, write='wb')
        #print(config)
        self.store_asset(format='json', file_path=self.dataset_dir_paths['config'], obj=config, write='w')   
        return config
    
    def process_centralized_load_from_config(self, instance: str, config: Dict, **kwargs):
        self.cli_logger.info(f'Loading Dataset {instance}!')
        # processing
        self.process_shared()
    
    def raindrop_processing(self,):
        # adjusted from https://github.com/mims-harvard/Raindrop/blob/main/code/Raindrop.py
        X_full = np.concatenate([p['arr'][None, ...]
            for idx, p in enumerate(self._X)
        ]) # (38803, 60, 34)
        train_val_splits = np.concatenate([self._splits['train'], self._splits['val']])
        X_train = X_full[train_val_splits, ...]
        X_full_scaled = self.scale_data(
            X_train=X_train, X_full=X_full
        )
        
        self._train_data = self._raindrop_prepare_data(X_full_scaled, mode='train')
        self._val_data = self._raindrop_prepare_data(X_full_scaled, mode='val')
        self._test_data = self._raindrop_prepare_data(X_full_scaled, mode='test')

    def _raindrop_prepare_data(self, X_full_scaled: np.ndarray, mode: str='train'):
        _zip = zip(
            X_full_scaled[self._splits[mode], ...], # data
            self._y[self._splits[mode], ...], # labels
            self._X[self._splits[mode], ...] # misc
        )
        return [
           utils.P12Data(
            seq_id=p['id'].split("\\")[-1].split(".psv")[0].strip(),
            data=torch.from_numpy(x),
            label=torch.from_numpy(y),
            seq_len=torch.tensor(p['length']),
            time=torch.from_numpy(p['time']),
            extended_static=torch.tensor(p['extended_static']),
            demogr_desc = self._construct_demogr_description(p['extended_static']) 
           ) for x, y, p in tqdm(_zip, total=len(self._splits[mode]),  desc=f"Preparing {mode} data")
        ]
    
    def vitst_processing(self):
        # Adjusted from ViTST https://github.com/Leezekun/ViTST/blob/main/dataset/P12data/process_scripts/ConstructImage.py
        
        # filter short trajectories 
        filtered_indices = [] if self._min_seq_length is None else \
            self.get_filter_indices(self._X, min_length=self._min_seq_length)

        X_full = np.concatenate([p['arr'][:p['length'], :]
            for idx, p in enumerate(self._X) if idx not in filtered_indices
        ]) # (1387075, 34)
        # compose on array accounting for the length of each sample
        non_zero_stats = (X_full > 0).astype(np.int32).sum(axis=0)
        # non_zero_stats = (data > 0).astype(np.int32).sum(axis=0)
        # array([1239908, 1198610,  467216, 1179916, 1205770,  935449, 1156131,
        #       39925,   20699,   57223,  105208,   91890,   73197,   46417,
        #       21634,   94203,   21394,   79539,   61786,   83858,    2504,
        #       235061,   34102,   85742,   53954,  126792,   19957,    9861,
        #       22749,  101870,   40119,   88085,    8588,   81869])
        # filter features with many extreme (non-zero values)
        select_index = np.where(non_zero_stats > np.percentile(non_zero_stats, self._percentile_of_features_used))[0] \
            if (self._percentile_of_features_used != 0 and self._percentile_of_features_used is not None) else np.arange(0, len(non_zero_stats), 1)
        self._setup_transforms(num_features=select_index.shape[0])

        X_full = []
        for gloabl_i, p in enumerate(tqdm(self._X, desc="Preparing data")):
            ts_data = p['arr']
            ts_times = p['time']

            #max_hours, num_params = ts_data.shape[0], ts_data.shape[1] # (60, 34)
            keep_idx = (ts_data > 0) # only consider non-negative values and non-zeros
            max_len_of_features_34 = keep_idx.astype(np.int32).sum(axis=0) # (34,)
            max_len_of_features = max(max_len_of_features_34[select_index]) # find the max, only select relevant features

            #ts_data_non_extrem_interp = np.zeros((max_len_of_features, select_index.shape[0]))
            ts_data_non_extrem_interp = np.zeros((p['length'], select_index.shape[0]))
            
            for local_i, param_idx in enumerate(select_index): # only used selected param indices
                ts_value_non_extreme = ts_data[keep_idx[:, param_idx], param_idx]
                if not bool(list(ts_value_non_extreme)): # if has non_extreme_values
                    continue
                
                # interpolate to get desired output size
                ts_data_non_extrem_interp[:, local_i] = np.interp(
                    np.linspace(0, len(ts_value_non_extreme), ts_data_non_extrem_interp.shape[0]),
                    np.arange(0, len(ts_value_non_extreme)),
                    ts_value_non_extreme
                )

            X_full.append(ts_data_non_extrem_interp)

        train_val_splits = np.concatenate([self._splits['train'], self._splits['val']])
        X_train = np.concatenate([X_full[i] for i in train_val_splits if i not in filtered_indices])

        X_full_scaled = self.scale_data(
            X_train=X_train, X_full=X_full
        )
        
        self._train_data = self._vitst_prepare_data(X_full_scaled, filtered_indices, mode='train')
        self._val_data = self._vitst_prepare_data(X_full_scaled, filtered_indices, mode='val')
        self._test_data = self._vitst_prepare_data(X_full_scaled, filtered_indices, mode='test')

    def _vitst_prepare_data(self, 
                            X_full_scaled: np.ndarray, 
                            filtered_indices: np.ndarray,
                            mode: str='train'):
        _zip = zip(
            [X_full_scaled[idx] for idx in self._splits[mode] if idx not in filtered_indices], # data
            [self._y[idx] for idx in self._splits[mode] if idx not in filtered_indices], # labels
            [self._X[idx] for idx in self._splits[mode] if idx not in filtered_indices] # misc
        )
        return [
           utils.P12Data(
            seq_id=p['id'].split("\\")[-1].split(".psv")[0].strip(),
            data=torch.from_numpy(x),
            label=torch.from_numpy(y),
            seq_len=torch.tensor(p['length']),
            time=torch.from_numpy(p['time']),
            extended_static=torch.tensor(p['extended_static']),
            demogr_desc = self._construct_demogr_description(p['extended_static']) 
           ) for x, y, p in tqdm(_zip, total=len(self._splits[mode]),  desc=f"Preparing {mode} data")
        ]


    @staticmethod
    def get_filter_indices(X, min_length: int=3):
        fi = [] 
        for i, p in enumerate(X):
            if int(p['length']) <= min_length:
                fi.append(i)
        return fi
    

    def _setup_transforms(self, num_features: int=None):
        num_features = len(self.__param_detailed_description) if num_features is None else num_features
        self.transforms = {
            f'{key}': StandardScaler()
            for key in range(num_features) # 34 features
        }
    
    def scale_data(self, X_train, X_full):
        for (key, transform) in self.transforms.items():
            values = X_train[..., int(key)]
            #print(values.reshape(-1,1).shape)
            # previous
            #_ = transform.fit_transform(values)
            # new
            _ = transform.fit_transform(values.reshape(-1,1))
        
        if isinstance(X_full, np.ndarray):
            data_scaled = np.zeros_like(X_full)
            for (key, transform) in tqdm(self.transforms.items(), desc="Scaling features"):
                values = X_full[..., int(key)]
                # old
                #data_scaled[..., int(key)] = transform.transform(values)
                # new 
                N, S = values.shape
                data_scaled[..., int(key)] = transform.transform(
                    values.reshape(-1,1)
                ).reshape(N, S)
        
        elif isinstance(X_full, list):
            X_full_arr = np.concatenate([x
                for x in X_full
            ])
            ptr = [0]
            for i in range(len(X_full)):
                prev = ptr[i]
                ptr.append(
                    prev+X_full[i].shape[0]
                )
            
            data_scaled_arr = np.zeros_like(X_full_arr)
            for (key, transform) in tqdm(self.transforms.items(), desc="Scaling features"):
                values = X_full_arr[:, int(key)] 
                # old
                #data_scaled[..., int(key)] = transform.transform(values)
                # new 
                #N, S = values.shape
                data_scaled_arr[:, int(key)] = transform.transform(
                    values.reshape(-1,1)
                ).reshape(-1)
            
            data_scaled = []
            for i in range(len(ptr)-1):
                j, k = ptr[i], ptr[i+1]
                data_scaled.append(
                    data_scaled_arr[j:k, :]
                )

        return data_scaled

    def _check_if_centralized_dataset_exists(self, 
        load_from_config, ds_config, tmp_config
        ):
        # check 
        # do not include instance key here
        if int(tmp_config['train_split_index']) == self.train_split_index:
            load_from_config = True
            ds_config = tmp_config
            self.cli_logger.info("Dataset already generated.")
        
        return load_from_config, ds_config
    
    def _log_stats(self, **kwargs):
        utils.log_stats(**kwargs)

    def setup_aws_session(self):
        self.session = boto3.Session(profile_name=self.aws_profile)
        self.s3_client = self.session.client('s3')
        self.s3 = self.session.resource('s3')
        if self.s3_bucket_path is not None:
            self.prefix = 'data/p19'
            self.s3_buckets = list(self.s3.Bucket(self.s3_bucket_path).objects.filter(Prefix=f'{self.prefix}/processed/centralized/'))
    
    def _construct_demogr_description(self, static_demogr):
        desc = "A patient is"
        # age
        if static_demogr[0]:
            desc = f"{desc} {int(static_demogr[0])} years old and"

        # gender
        if int(static_demogr[1]) == 1:
            gender = "female"
            pronoun = 'She'
        elif int(static_demogr[2]) == 1:
            gender = "male"
            pronoun = 'He'
        else: 
            gender = ''
            pronoun = 'The patient'
        if gender.startswith('f') or gender.startswith('m'):
            desc = f"{desc} {gender}"

        body = ''
        # height
        if static_demogr[3] > 0:
            height = f"{static_demogr[3]} cm"
            body = body + f"{pronoun} is {height} tall"
        # weight
        if static_demogr[8] > 0:
            weight = f"{static_demogr[8]} kg"
            if body.startswith(f'{pronoun}'):
                body = body + f' and weights {weight}'
            else:
                body = body + f'{pronoun} weights {weight}'
        
        if body.startswith(f'{pronoun}'):
            desc = f"{desc}. {body}"
        
        # icu type
        if int(static_demogr[4]) == 1:
            icu = "coronary care unit"
            icu = f"stayed in {icu}"
        elif int(static_demogr[5]) == 1:
            icu = "cardiac surgery recovery unit"
            icu = f"stayed in {icu}"
        elif int(static_demogr[6]) == 1:
            icu = "medical ICU"
            icu = f"stayed in {icu}"
        elif int(static_demogr[7]) == 1:
            icu = "surgical ICU"
            icu = f"stayed in {icu}"
        else:
            icu = f'stayed not in ICU'

        if desc == "A patient is":
            desc = ""
        else:
            desc = f'{desc}. {pronoun} {icu}.'

        return desc


    @property
    def data_raw(self,):
        return self._data_raw

    @property
    def splits(self,):
        return self._splits



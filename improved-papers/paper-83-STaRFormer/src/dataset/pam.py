import os
import os.path as osp
import numpy as np
import torch
import logging
import io
import pickle

from pathlib import Path
from typing import List, Tuple, Dict, Union
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
    "PAMDataset"
]


class PamSubDataset(BaseDataSubset):
    """
    A dataset class for handling PAM data subsets.

    This class inherits from `BaseDataSubset` and is specifically designed to handle
    subsets of PAM data. It initializes the dataset with a list of `GeoLifeData` objects.

    Attributes:
        data (List[geolife_utils.GeoLifeData]): A list of GeoLifeData objects.

    Methods:
        __init__(data: List[geolife_utils.GeoLifeData]) -> None:
            Initializes the dataset with the provided PAM data.
    """

    def __init__(self, data: List[utils.PAMData]) -> None:
        """
        Initializes the dataset with the provided PAM data.

        Args:
            data (List[geolife_utils.GeoLifeData]): A list of GeoLifeData objects.
        """
        super().__init__(data)


class PAMDataset(BaseDataset):
    """
    PAMDataset class for handling PAM (Physical Activity Monitoring) data.

    This class inherits from BaseDataset and provides functionalities to load,
    process, and split the PAM data for training, validation, and testing purposes.

    Attributes:
        __url (str): URL to download the PAM dataset.
        __raw_zip_filename (str): Filename for the raw PAM dataset zip file.
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

    __url = "https://figshare.com/ndownloader/files/34683103"
    __raw_zip_filename = "pam_raw.zip"
    __raw_file_names = {
        'data': 'PTdict_list.pkl',
        'lables': 'arr_outcomes.pkl',
        'irregularity_mask': 'irregularity_mask.pkl'
    }
    __config_filename = 'config.json'
    __splits = {
        0: 'PAM_split_1.pkl',
        1: 'PAM_split_2.pkl',
        2: 'PAM_split_3.pkl',
        3: 'PAM_split_4.pkl',
        4: 'PAM_split_5.pkl',
    }
    
    def __init__(self,
        seed: int, 
        training_method: str=None,
        train_split_index: int=0,
        aws_profile: str=None, 
        s3_bucket_path: Union[str, Path, os.PathLike[str]]=None,
        use_threads: bool = False,
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
        self._dataset_class = PamSubDataset
        self.training_method = training_method
        self.train_split_index = train_split_index
        if train_split_index > 4:
            raise ValueError(f'{train_split_index=}, must be < 5, picke between 0,1,2,3,4')
        self.aws_profile = aws_profile
        self.s3_bucket_path = s3_bucket_path
        self.use_threads = use_threads
        self.setup_aws_session()
        self._setup_transforms()

        super().__init__(dataset_name=DatasetOptions.pam, seed=seed, **kwargs)
    
    def __len__(self,):
        return len(self._targets)
    
    def __getitem__(self, index) -> utils.PAMData:
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
            self.cli_logger.info(f'Downloading {DatasetOptions.pam} from S3 Bucket {self.s3_bucket_path}!')
            
            prefix_X = f"{self.prefix}/raw/processed_data/{self.__raw_file_names['data']}"
            prefix_y = f"{self.prefix}/raw/processed_data/{self.__raw_file_names['lables']}"
            prefix_irregularity_mask = f"{self.prefix}/raw/processed_data/{self.__raw_file_names['irregularity_mask']}"
            prefix_splits = f"{self.prefix}/raw/splits/{self.__splits[self.train_split_index]}"
            
            self.cli_logger.info('Loading Data')
            s3_object_X = self.s3_client.get_object(Bucket=self.s3_bucket_path, Key=prefix_X)
            self._X = pickle.loads(s3_object_X.get("Body").read())

            self.cli_logger.info('Loading Labels')
            s3_object_y = self.s3_client.get_object(Bucket=self.s3_bucket_path, Key=prefix_y)
            self._y = pickle.loads(s3_object_y.get("Body").read())

            self.cli_logger.info('Loading Irregularity Mask')
            s3_object_irregularity_mask = self.s3_client.get_object(Bucket=self.s3_bucket_path, Key=prefix_irregularity_mask)
            self._irregularity_mask = pickle.loads(s3_object_irregularity_mask.get("Body").read())

            self.cli_logger.info('Loading Irregularity Mask')
            s3_object_splits = self.s3_client.get_object(Bucket=self.s3_bucket_path, Key=prefix_splits)
            self._splits = pickle.loads(s3_object_splits.get("Body").read())
            self.cli_logger.info('Done!')

            self._data_raw = (self._X, self._y)

        else:
            if not osp.exists(osp.join(self.dataset_dir_paths['raw'], self.__raw_zip_filename)):
                # Downloading data 
                self.cli_logger.info(f'Downloading raw {DatasetOptions.pam} dataset from {self.__url}!')
                utils.download(
                    dataset_name=DatasetOptions.pam,
                    url=self.__url,
                    folder_path=self.dataset_dir_paths['raw'],
                    file_name=self.__raw_zip_filename,
                    logger=self.cli_logger
                )
                # rename splits files 
                for file in os.listdir(self.dataset_dir_paths['splits']):
                    file_path = osp.join(self.dataset_dir_paths['splits'], file)
                    idx = file.split('.')[0].split('_')[-1]
                    new_file_name = f'PAM_split_{idx}.pkl'
                    new_file_path = osp.join(self.dataset_dir_paths['splits'], new_file_name)

                    splits_obj = np.load(file_path, allow_pickle=True)
                    splits_dict = {
                        'train' : splits_obj[0],
                        'val' : splits_obj[1],
                        'test' : splits_obj[2]
                    }
                    self.store_asset(format='pickle', file_path=new_file_path, obj=splits_dict, write='wb')   

                # rename data and labels files  
                X = np.load(osp.join(self.dataset_dir_paths['processed_data'], self.__raw_file_names['data'].replace('.pkl', '.npy')), allow_pickle=True)
                y = np.load(osp.join(self.dataset_dir_paths['processed_data'], self.__raw_file_names['lables'].replace('.pkl', '.npy')), allow_pickle=True)
                
                X_path = osp.join(self.dataset_dir_paths['processed_data'], self.__raw_file_names['data'])
                y_path = osp.join(self.dataset_dir_paths['processed_data'], self.__raw_file_names['lables'])
                
                self.store_asset(format='pickle', file_path=X_path, obj=X, write='wb')   
                self.store_asset(format='pickle', file_path=y_path, obj=y, write='wb')

            else:
                self.cli_logger.info(f'Found raw {DatasetOptions.pam} dataset!')
            
            self._X = self.load_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['processed_data'], self.__raw_file_names['data']), read='rb')
            self._y = self.load_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['processed_data'], self.__raw_file_names['lables']), read='rb')

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
    
    def _create_irreg_data(self, irreg_mask_p: float=0.6):
        """
        irreg_mask_p (float): Percentage of elements to be masked, i.e., removed     
        """
        # create irregulary times series 
        N, seq_len, D = self._X.shape
        seq_len_irreg = int(seq_len * (1 - irreg_mask_p))

        X_irreg = np.zeros((N, seq_len_irreg, D))
        for i in tqdm(range(X_irreg.shape[0]), desc="Create irregular time-series"):
            sample = self._X[i] # [600, 17]
            # handling missing value and extreme values
            sample_irreg_interp = np.zeros((seq_len_irreg, D)) # [240, 17]
            kept_index = (sample != 0) # adjusted from ViTST preprocessing, raindrop only considers postive nonzero values (>0)
            for d in range(D):
                # interpolate to get desired output shape via interpolation, i.e. 40% of original data, mask 60% --> 600*0.4=240 
                sample_irreg = sample[kept_index[:, d], d]
                sample_irreg_interp[:, d] = np.interp(
                    np.linspace(0, len(sample_irreg), seq_len_irreg),
                    np.arange(0, len(sample_irreg)),
                    sample_irreg
                )
            X_irreg[i] = sample_irreg_interp
        
        return X_irreg
    
    def process_centralized(self, instance: str, **kwargs):
        self.cli_logger.info(f'Processing ...!')
        config = {
            'instance': instance,
            'training_method': TrainingMethodOptions.centralized,
            'train_split_index': str(self.train_split_index),
        }
        if not osp.exists(osp.join(self.dataset_dir_paths['centralized_instance'])):
            os.makedirs(osp.join(self.dataset_dir_paths['centralized_instance']))
        
        X_irreg = self._create_irreg_data(irreg_mask_p=0.6)
        X_irreg_scaled = self.scale_data(
            X_train=X_irreg[self._splits['train'], :], X_full=X_irreg
        )

        X_train, y_train = torch.from_numpy(X_irreg_scaled[self._splits['train'], :]), torch.from_numpy(self._y[self._splits['train']]).type(torch.long)
        X_val, y_val = torch.from_numpy(X_irreg_scaled[self._splits['val'], :]), torch.from_numpy(self._y[self._splits['val']]).type(torch.long)
        X_test, y_test = torch.from_numpy(X_irreg_scaled[self._splits['test'], :]), torch.from_numpy(self._y[self._splits['test']]).type(torch.long)

        self._train_data = [utils.PAMData(data=x, label=y) for x, y in zip(X_train, y_train)]
        self._val_data = [utils.PAMData(data=x, label=y) for x, y in zip(X_val, y_val)]
        self._test_data = [utils.PAMData(data=x, label=y) for x, y in zip(X_test, y_test)]

        self._data = self._train_data + self._val_data + self._test_data

        self.train_dataset = self._dataset_class(data=self._train_data)
        self.val_dataset = self._dataset_class(data=self._val_data)
        self.test_dataset = self._dataset_class(data=self._test_data)
        
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

        X_irreg = self._create_irreg_data(irreg_mask_p=0.6)
        X_irreg_scaled = self.scale_data(
            X_train=X_irreg[self._splits['train'], :], X_full=X_irreg
        )

        X_train, y_train = torch.from_numpy(X_irreg_scaled[self._splits['train'], :]), torch.from_numpy(self._y[self._splits['train']]).type(torch.long)
        X_val, y_val = torch.from_numpy(X_irreg_scaled[self._splits['val'], :]), torch.from_numpy(self._y[self._splits['val']]).type(torch.long)
        X_test, y_test = torch.from_numpy(X_irreg_scaled[self._splits['test'], :]), torch.from_numpy(self._y[self._splits['test']]).type(torch.long)

        self._train_data = [utils.PAMData(data=x, label=y) for x, y in zip(X_train, y_train)]
        self._val_data = [utils.PAMData(data=x, label=y) for x, y in zip(X_val, y_val)]
        self._test_data = [utils.PAMData(data=x, label=y) for x, y in zip(X_test, y_test)]

        self._data = self._train_data + self._val_data + self._test_data

        self.train_dataset = self._dataset_class(data=self._train_data)
        self.val_dataset = self._dataset_class(data=self._val_data)
        self.test_dataset = self._dataset_class(data=self._test_data)
        
        # store data and config
        config['indices'] = {
            'train': self._splits['train'],
            'val': self._splits['val'],
            'test': self._splits['test'],
        }
    
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
            self.prefix = 'data/pam'
            self.s3_buckets = list(self.s3.Bucket(self.s3_bucket_path).objects.filter(Prefix=f'{self.prefix}/processed/centralized/'))
    
    def _setup_transforms(self):
        self.transforms = {
            f'{key}': StandardScaler()
            for key in range(17) # 17 features
        }

    def scale_data(self, X_train, X_full):
        for (key, transform) in self.transforms.items():
            values = X_train[..., int(key)]
            #print(values.reshape(-1,1).shape)
            _ = transform.fit_transform(values.reshape(-1,1))
            
        data_scaled = np.zeros_like(X_full)
        for (key, transform) in tqdm(self.transforms.items(), desc="Scaling features"):
            values = X_full[..., int(key)]
            N, S = values.shape
            data_scaled[..., int(key)] = transform.transform(
                 values.reshape(-1,1)
            ).reshape(N, S)
        return data_scaled
    
    def raindrop_preprocessing(self):
        def getStats(X_train: np.ndarray, eps: float=1e-7):
            """
            Adjusted from https://github.com/mims-harvard/Raindrop
            """
            N, seq_len, D = X_train.shape

            X_train_features_reshape = X_train.transpose((2,0,1)).reshape(D, -1) # [D, N*S]

            # initialize arrays to store the mean and standard deviation for ech feature
            mean_fts = np.zeros((D, 1))
            std_fts = np.zeros((D, 1))

            # loop through each feature
            for fts in tqdm(range(D), desc="Calculating feature-wise stats"):
                val_fts = X_train_features_reshape[fts, :]
                # only consider non-negative values and non-zeros, considered missing values
                val_fts = val_fts[val_fts != 0]

                # compute mean and std
                mean_fts[fts] = np.mean(val_fts)
                std_fts[fts] = np.std(val_fts)

                # ensure std is at least eps to avoid devision by 0
                std_fts[fts] = np.max([std_fts[fts][0], eps])

            return mean_fts, std_fts

        def _preprocess(X: np.ndarray, mean_fts: np.ndarray, std_fts: np.ndarray, irreg_mask_p: float=0.6):
            """
            Create irregulary times series by only consider non-negative values and non-zeros

            irreg_mask_p: float
                Percentage of elements to be masked, i.e., removed
            """
            # create irregulary times series
            N, seq_len, D = self._X.shape
            seq_len_irreg = int(seq_len * (1 - irreg_mask_p))
            X_irreg = np.zeros((N, seq_len_irreg, D))

            for i in tqdm(range(X_irreg.shape[0]), desc="Create irregular time-series"):
                sample = X[i] # [600, 17]
                sample_non_extrem_interp = np.zeros((seq_len_irreg, D)) # [240, 17]
                # handling missing value and extreme values
                keep_idx = (sample != 0) # only consider non-negative values and non-zeros
                for d in range(D):
                    # interpolate to get desired output, i.e. 40% of original data, mask 60% --> 600*0.4=240
                    sample_non_extrem = sample[keep_idx[:, d], d]
                    # as keep_idx has different lengths, we need to interpolate to desired size
                    if not bool(list(sample_non_extrem)):
                        continue

                    sample_non_extrem_interp[:, d] = np.interp(
                        np.linspace(0, len(sample_non_extrem), seq_len_irreg),
                        np.arange(0, len(sample_non_extrem)),
                        sample_non_extrem
                    )
                    # compute z-score
                    sample_non_extrem_interp[:, d] = ((sample_non_extrem_interp[:, d] - mean_fts[d]) / (std_fts[d] + 1e-18))

                X_irreg[i] = sample_non_extrem_interp
            return X_irreg
        X_train = self._X[self.splits['train'], ...]
        mean_fts, std_fts = getStats(X_train=X_train)
        return _preprocess(X=self._X, mean_fts=mean_fts, std_fts=std_fts, irreg_mask_p=0.6)

    def _setup_transforms(self):
        self.transforms = {
            f'{key}': StandardScaler()
            for key in range(17) # 17 features
        }

    def scale_data(self, X_train, X_full):
        for (key, transform) in self.transforms.items():
            values = X_train[..., int(key)]
            #print(values.reshape(-1,1).shape)
            # previous
            #_ = transform.fit_transform(values)
            # new
            _ = transform.fit_transform(values.reshape(-1,1))
        
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

        return data_scaled
    
    @property
    def data_raw(self,):
        return self._data_raw

    @property
    def splits(self,):
        return self._splits


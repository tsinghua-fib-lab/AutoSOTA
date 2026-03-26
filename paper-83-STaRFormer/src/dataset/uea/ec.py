
import os
import os.path as osp
import pickle
import numpy as np
import torch

from pathlib import Path
from typing import List, Dict, Union
from datetime import datetime
from aeon.datasets import load_classification
from torch.utils.data import random_split
from tqdm.auto import tqdm
from torch import Tensor

from ..base import BaseDataset

from src.utils import DatasetOptions, TrainingMethodOptions
from src.utils.datasets.uea import UcrUeaData, UcrUeaBatchData, UcrUeaSubDataset, log_stats
import boto3

__all__ = [
    "EthanolConcentrationData",
    "EthanolConcentrationBatchData",
    "EthanolConcentrationDataset"
]
"""
refactor
"""

class EthanolConcentrationData(UcrUeaData):
    def __init__(self, data: Tensor=None,  label: Tensor=None,  **kwargs ) -> None:
        super().__init__(data, label, **kwargs)


class EthanolConcentrationBatchData(UcrUeaBatchData):
    def __init__(self, data: Tensor=None,  label: Tensor=None,  seq_len: list |  Tensor=None, **kwargs) -> None:
        super().__init__(data, label, seq_len, **kwargs)



class EthanolConcentrationSubDataset(UcrUeaSubDataset):
    def __init__(self, data: List[UcrUeaData]) -> None:
        super().__init__(data)
        

class EthanolConcentrationDataset(BaseDataset):
    """
    EthanolConcentrationDataset class for handling the Ethanol Concentration dataset.

    This class inherits from BaseDataset and provides functionalities to load and process the
    Ethanol Concentration dataset, including managing AWS S3 access, handling different training 
    methods, and splitting data into training, validation, and test sets.

    Attributes:
        __raw_file_names (dict): Dictionary mapping data types to their respective filenames.
        __config_filename (str): Filename for the configuration file.
        label2idx (dict): Dictionary mapping labels to their respective indices.
        __name (str): Name of the dataset.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the data and target at the specified index.
        download(**kwargs): Downloads the dataset.
        process(**kwargs): Processes the dataset.
        process_centralized(instance, **kwargs): Processes centralized data.
        process_centralized_load_from_config(instance, config, **kwargs): Processes centralized data from configuration.
        _check_if_centralized_dataset_exists(load_from_config, ds_config, tmp_config): Checks if a centralized dataset configuration exists.
        _log_stats(**kwargs): Logs dataset statistics.
        setup_aws_session(): Sets up the AWS session for accessing S3.
    """
    __raw_file_names = {
        'X': 'X.pkl',
        'y': 'y.pkl',
        'X_train': 'X_train.pkl',
        'y_train': 'y_train.pkl',
        'X_test': 'X_test.pkl',
        'y_test': 'y_test.pkl',
    }
    __name = 'EthanolConcentration'
    __config_filename = 'config.json'
    label2idx = {'e35': 0, 'e38': 1, 'e40': 2, 'e45': 3}

    def __init__(self, 
        seed,
        training_method: str=None,
        train_splits: Dict[str, float]={'train': 0.95, 'val': 0.05},
        max_trajectory_length: int=1024,
        aws_profile: str=None, 
        s3_bucket_path: Union[str, Path, os.PathLike[str]]=None,
        use_threads: bool = False,
        **kwargs):
        self.training_method = training_method
        self.train_splits = train_splits
        self.aws_profile = aws_profile
        self.s3_bucket_path = s3_bucket_path
        self.use_threads = use_threads
        self.max_trajectory_length = max_trajectory_length
        self.setup_aws_session()
        """
        Args:
            seed (int): Random seed for reproducibility.
            training_method (str, optional): Method for training. Default: None.
            train_splits (Dict[str, float], optional): Proportions for train, validation, and test splits. Default: {'train': 0.95, 'val': 0.05}.
            max_trajectory_length (int, optional): Maximum length for data trajectories. Default: 1024.
            aws_profile (str, optional): AWS profile for accessing S3. Default: None.
            s3_bucket_path (Union[str, Path, os.PathLike[str]], optional): S3 bucket path for loading data. Default: None.
            use_threads (bool, optional): Whether to use threads for loading data. Default: False.
            **kwargs: Additional keyword arguments.
        """
        
        super().__init__(dataset_name=DatasetOptions.ethanolconcentration, seed=seed, **kwargs)

    def __len__(self, ):
        return len(self._targets)
    
    def __getitem__(self, index):
        return self._data[index]
    
    def download(self):
        now = datetime.now()
        if self.training_method.lower() == TrainingMethodOptions.centralized:
            self.dataset_dir_paths['centralized_instance'] = osp.join(self.dataset_dir_paths['centralized'], now.strftime("%Y-%m-%d_%H:%M:%S"))
        elif self.training_method.lower() == TrainingMethodOptions.federated:
            self.dataset_dir_paths['federated_instance'] = osp.join(self.dataset_dir_paths['federated'], now.strftime("%Y-%m-%d_%H:%M:%S"))

        if self.aws_profile is not None or self.s3_bucket_path is not None: # load data from aws bucket
            self.cli_logger.info(f'Downloading {DatasetOptions.ethanolconcentration} from S3 Bucket {self.s3_bucket_path}!')
            
            prefix_X = f"{self.prefix}/raw/X.pkl"
            prefix_X_train = f"{self.prefix}/raw/X_train.pkl"
            prefix_X_test = f"{self.prefix}/raw/X_test.pkl"
            prefix_y = f"{self.prefix}/raw/y.pkl"
            prefix_y_train = f"{self.prefix}/raw/y_train.pkl"
            prefix_y_test = f"{self.prefix}/raw/y_test.pkl"

            s3_object_X = self.s3_client.get_object(Bucket=self.s3_bucket_path, Key=prefix_X)
            s3_object_X_train = self.s3_client.get_object(Bucket=self.s3_bucket_path, Key=prefix_X_train)
            s3_object_X_test = self.s3_client.get_object(Bucket=self.s3_bucket_path, Key=prefix_X_test)
            s3_object_y = self.s3_client.get_object(Bucket=self.s3_bucket_path, Key=prefix_y)
            s3_object_y_train = self.s3_client.get_object(Bucket=self.s3_bucket_path, Key=prefix_y_train)
            s3_object_y_test = self.s3_client.get_object(Bucket=self.s3_bucket_path, Key=prefix_y_test)
            
            self.cli_logger.info('Loading data form S3 Bucket!')
            self.cli_logger.info('Loading X!')
            X = pickle.loads(s3_object_X['Body'].read())
            self.cli_logger.info('Loading X_train!')
            X_train = pickle.loads(s3_object_X_train['Body'].read())
            self.cli_logger.info('Loading X_test!')
            X_test = pickle.loads(s3_object_X_test['Body'].read())
            self.cli_logger.info('Loading y!')
            y = pickle.loads(s3_object_y['Body'].read())
            self.cli_logger.info('Loading y_train!')
            y_train = pickle.loads(s3_object_y_train['Body'].read())
            self.cli_logger.info('Loading y_test!')
            y_test = pickle.loads(s3_object_y_test['Body'].read())
            self.cli_logger.info('Finished loading S3 Bucket!')

        else:
            #print(self.dataset_dir_paths['raw'], not not self.dataset_dir_paths['raw'])
            if not osp.exists(self.dataset_dir_paths['raw']):
                os.makedirs(self.dataset_dir_paths['raw'])
                
                
            if not osp.exists(osp.join(self.dataset_dir_paths['raw'], 'X.pkl')) and \
                not osp.exists(osp.join(self.dataset_dir_paths['raw'], 'X_train.pkl')) and \
                not osp.exists(osp.join(self.dataset_dir_paths['raw'], 'X_test.pkl')) and \
                not osp.exists(osp.join(self.dataset_dir_paths['raw'], 'y.pkl')) and \
                not osp.exists(osp.join(self.dataset_dir_paths['raw'], 'y_train.pkl')) and \
                not osp.exists(osp.join(self.dataset_dir_paths['raw'], 'y_test.pkl')):
                self.cli_logger.info(f'Downloading {DatasetOptions.ethanolconcentration} from source!')

                X, y, train_meta_data = load_classification(name=self.__name, return_metadata=True)
                X_train, y_train, train_meta_data = load_classification(name=self.__name, split='train', return_metadata=True)
                X_test, y_test, meta_data = load_classification(name=self.__name, split='test', return_metadata=True)
                
                # storing
                self.store_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self.__raw_file_names['X']), obj=X, write='wb')
                self.store_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self.__raw_file_names['y']), obj=y, write='wb')
                self.store_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self.__raw_file_names['X_train']), obj=X_train, write='wb')
                self.store_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self.__raw_file_names['y_train']), obj=y_train, write='wb')
                self.store_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self.__raw_file_names['X_test']), obj=X_test, write='wb')
                self.store_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self.__raw_file_names['y_test']), obj=y_test, write='wb')
            else:
                self.cli_logger.info(f'Found raw {DatasetOptions.ethanolconcentration} dataset!')
                X = self.load_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self.__raw_file_names['X']), read='rb')
                y = self.load_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self.__raw_file_names['y']), read='rb')
                X_train = self.load_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self.__raw_file_names['X_train']), read='rb')
                y_train = self.load_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self.__raw_file_names['y_train']), read='rb')
                X_test = self.load_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self.__raw_file_names['X_test']), read='rb')
                y_test = self.load_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self.__raw_file_names['y_test']), read='rb')
            
        X = X.transpose(0,2,1) # [Size, D, Seq] --> [Size, Seq, D]
        X_train = X_train.transpose(0,2,1) # [Size, D, Seq] --> [Size, Seq, D]
        X_test = X_test.transpose(0,2,1) # [Size, D, Seq] --> [Size, Seq, D]
        
        
        X = torch.from_numpy(X)
        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        
        y = torch.from_numpy(np.array([self.label2idx[k] for k in y]))
        y_train = torch.from_numpy(np.array([self.label2idx[k] for k in y_train]))
        y_test = torch.from_numpy(np.array([self.label2idx[k] for k in y_test]))
        
        self._data = [
            EthanolConcentrationData(data=X[i], label=y[i])
            for i in range(len(X))
        ]
        self._targets = y

        self._train_data = [
            EthanolConcentrationData(data=X_train[i], label=y_train[i])
            for i in range(len(X_train))
        ]
        self._train_targets = y_train

        self._test_data = [
            EthanolConcentrationData(data=X_test[i], label=y_test[i])
            for i in range(len(X_test))
        ]
        self._test_targets = y_test

    
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
            #'ds_config': ds_config
        })

    def process_centralized(self, instance: str, **kwargs):
        self.cli_logger.info(f'Processing ...!')
        config = {
            'instance': instance,
            'training_method': TrainingMethodOptions.centralized,
            'train_splits': dict(self.train_splits),
            'seed': self.seed,
            'max_trajectory_length': self.max_trajectory_length
        }
        if not osp.exists(osp.join(self.dataset_dir_paths['centralized_instance'])):
            os.makedirs(osp.join(self.dataset_dir_paths['centralized_instance']))

        # shorten trajectories
        train_data_short = []
        test_data_short = []
        for sample in tqdm(self._train_data, desc='Shorten Train Data'):
            short_data = shorten_trajectory(sample.data, max_trajectory_length=self.max_trajectory_length)
            sample.data = short_data
            sample.seq_len = torch.tensor(self.max_trajectory_length)
            train_data_short.append(sample)
        self._train_data = train_data_short

        for sample in tqdm(self._test_data, desc='Shorten Test Data'):
            short_data = shorten_trajectory(sample.data, max_trajectory_length=self.max_trajectory_length)
            sample.data = short_data
            sample.seq_len = torch.tensor(self.max_trajectory_length)
            test_data_short.append(sample)
        self._test_data = test_data_short 
        
        train_dataset = EthanolConcentrationSubDataset(data=self._train_data)
        if len(self.train_splits.values()) > 1:
            self.train_dataset, self.val_dataset = random_split(train_dataset, list(self.train_splits.values()))
        else:
            self.train_dataset, self.val_dataset = train_dataset, None

        # store data and config
        config['indices'] = {
            'train': self.train_dataset.indices,
            'val': self.val_dataset.indices if self.val_dataset is not None else None,
        }

        self.train_dataset = EthanolConcentrationSubDataset(data=[self._train_data[idx] for idx in config['indices']['train']])
        self.val_dataset = EthanolConcentrationSubDataset(data=[self._train_data[idx] for idx in config['indices']['val']])
        self.test_dataset = EthanolConcentrationSubDataset(data=self._test_data)
        
        if not osp.exists(self.dataset_dir_paths["centralized_instance"]):
            os.makedirs(self.dataset_dir_paths["centralized_instance"])
        
        #self.store_asset(format='pickle', file_path=self.dataset_dir_paths['data'], obj=self._data, write='wb')
        #print(type(config), type(self.train_splits))
        self.store_asset(format='json', file_path=self.dataset_dir_paths['config'], obj=config, write='w')   
        return config
    
    def process_centralized_load_from_config(self, instance: str, config: Dict, **kwargs):
        self.cli_logger.info(f'Loading Dataset {instance}!')
    
        train_ids = config['indices']['train']
        val_ids = config['indices']['val']

        # shorten trajectories
        train_data_short = []
        test_data_short = []
        for sample in tqdm(self._train_data, desc='Shorten Train Data'):
            short_data = shorten_trajectory(sample.data, max_trajectory_length=self.max_trajectory_length)
            sample.data = short_data
            sample.seq_len = torch.tensor(self.max_trajectory_length)
            train_data_short.append(sample)
        self._train_data = train_data_short

        for sample in tqdm(self._test_data, desc='Shorten Test Data'):
            short_data = shorten_trajectory(sample.data, max_trajectory_length=self.max_trajectory_length)
            sample.data = short_data
            sample.seq_len = torch.tensor(self.max_trajectory_length)
            test_data_short.append(sample)
        self._test_data = test_data_short

        self.train_dataset = EthanolConcentrationSubDataset(data=[self._train_data[idx] for idx in train_ids])
        self.val_dataset = EthanolConcentrationSubDataset(data=[self._train_data[idx] for idx in val_ids])
        self.test_dataset = EthanolConcentrationSubDataset(data=self._test_data)
    
    def _check_if_centralized_dataset_exists(self, 
        load_from_config, ds_config, tmp_config
        ):
        # check 
        # do not include instance key here
        if tmp_config['seed'] == self.seed and \
            tmp_config['train_splits'] == self.train_splits:
            if self.max_trajectory_length: 
                if tmp_config.get('max_trajectory_length', None) is not None:
                    if tmp_config['max_trajectory_length'] == self.max_trajectory_length:
                        load_from_config = True
                        ds_config = tmp_config
                        self.cli_logger.info("Dataset already generated.")
            
            else:
                load_from_config = True
                ds_config = tmp_config
                self.cli_logger.info("Dataset already generated.")
        
        return load_from_config, ds_config
    
    def _log_stats(self, **kwargs):
        log_stats(**kwargs)

    def setup_aws_session(self):
        self.session = boto3.Session(profile_name=self.aws_profile)
        self.s3_client = self.session.client('s3')
        self.s3 = self.session.resource('s3')
        if self.s3_bucket_path is not None:
            self.prefix = 'data/ethanolconcentration'
            self.s3_buckets = list(self.s3.Bucket(self.s3_bucket_path).objects.filter(Prefix=f'{self.prefix}/processed/centralized/'))
        

def shorten_trajectory(trajectory_data: Tensor, max_trajectory_length: int=400):
    """
    Shortens a trajectory sequence to a specified length N, keeping the first and last elements
    and selecting the rest in a special manner.
    
    Args:
        trajectory (list or numpy array): The input trajectory sequence.
        N (int): The desired length of the shortened trajectory.
    
    Returns:
        numpy array: The shortened trajectory sequence.
    """
    # Get the length of the input trajectory
    trajectory_length = trajectory_data.size(0)
    # Check if the trajectory is already shorter than the desired length
    if trajectory_length <= max_trajectory_length:
        return trajectory_data
    
    # Create the shortened trajectory array
    shortened_trajectory_data = torch.zeros(max_trajectory_length, trajectory_data.size(1))
    
    # Set the first and last elements
    shortened_trajectory_data[0, :] = trajectory_data[0, :]
    shortened_trajectory_data[-1, :] = trajectory_data[-1, :]
    
    # Select the middle elements in a special manner
    step_size = (trajectory_length - 2) / (max_trajectory_length - 2)
    for i in range(1, max_trajectory_length - 1):
        index = int(i * step_size)
        shortened_trajectory_data[i] = trajectory_data[index]
    
    return shortened_trajectory_data

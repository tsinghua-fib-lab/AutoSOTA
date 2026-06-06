
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
    'UEAData',
    'UEABatchData',
    'UEASubDataset',
    'UEADataset',
]

class UEAData(UcrUeaData):
    def __init__(self, data: Tensor=None,  label: Tensor=None,  **kwargs ) -> None:
        super().__init__(data, label, **kwargs)


class UEABatchData(UcrUeaBatchData):
    def __init__(self, data: Tensor=None,  label: Tensor=None,  seq_len: list |  Tensor=None, **kwargs) -> None:
        super().__init__(data, label, seq_len, **kwargs)



class UEASubDataset(UcrUeaSubDataset):
    def __init__(self, data: List[UcrUeaData]) -> None:
        super().__init__(data)


class UEADataset(BaseDataset):
    """UEADataset

    A dataset wrapper for the UEA benchamrk datasets. Can handle any dataset in the UEA benchmark.

    Attributes:
        __raw_file_names (dict): Mapping of logical data components to on-disk file
        names used for loading (e.g., features X, labels y, splits).
        __config_filename (str): Name of the per-dataset configuration file.
        _label2idx (dict): Dataset-specific label-to-index mappings. Keys are dataset
        names and values are dictionaries mapping raw label values to integer
        class indices.
    """
    __raw_file_names = {
        'X': 'X.pkl',
        'y': 'y.pkl',
        'X_train': 'X_train.pkl',
        'y_train': 'y_train.pkl',
        'X_test': 'X_test.pkl',
        'y_test': 'y_test.pkl',
    }
    __config_filename = 'config.json'
    _label2idx = {
        'articularywordrecognition': {str(float(k)): i for i, k in enumerate(range(1,26))}, # n_classes + 1 for range
        'atrialfibrillation': {k: i for i, k in enumerate(['n', 's', 't']) },
        'basicmotions': {k: i for i, k in enumerate(['badminton', 'running', 'standing', 'walking'])},
        'charactertrajectories': {str(int(k)): i for i, k in enumerate(range(1,21))},
        'cricket': {str(float(k)): i for i, k in enumerate(range(1,13))},
        'duckduckgeese': {k: i for i, k in enumerate(['black-bellied_whistling_duck', 'canadian_goose', 'greylag_goose','pink-footed_goose', 'white-faced_whistling_duck'])},
        'eigenworms': {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4},
        'epilepsy': {k: i for i, k in enumerate(['epilepsy', 'running', 'sawing', 'walking'])},
        'ering': {k: i for i, k in enumerate(['1', '2', '3', '4', '5', '6'])},
        'ethanolconcentration': {'e35': 0, 'e38': 1, 'e40': 2, 'e45': 3},
        'facedetection': None,
        'fingermovements': {k: i for i, k in enumerate(['left', 'right'])},
        'handmovementdirection': {k: i for i, k in enumerate(['backward', 'forward', 'left', 'right'])},
        'handwriting': {str(float(k)): i for i, k in enumerate(range(1,27))}, # n_classes + 1 for range
        'heartbeat': {'abnormal': 1, 'normal': 0},
        'insectwingbeat': {k: i for i, k in enumerate(['aedes_female', 'aedes_male', 'fruit_flies', 'house_flies', 'quinx_female', 'quinx_male', 'stigma_female', 'stigma_male', 'tarsalis_female', 'tarsalis_male'])},
        'japanesevowels': None,
        'libras': {str(k): i for i, k in enumerate(range(1,16))},
        'lsst': {k: i for i, k in enumerate(['6', '15', '16', '42', '52', '53', '62', '64', '65', '67', '88', '90', '92', '95'])},
        'motorimagery': {'finger': 0, 'tongue': 1},
        'natops': {str(float(k)): i for i, k in enumerate(range(1,7))},
        'pemssf': {str(float(v)): i for i, v in enumerate(range(1,8))},
        'pendigits': {str(v): v for v in range(10)},
        'phonemespectra': {k: i for i, k in enumerate(['aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'eh',
                                                    'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n',
                                                    'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v',
                                                    'w', 'y', 'z', 'zh'])},
        'racketsports': {k: i for i, k in enumerate(['badminton_clear', 'badminton_smash', 'squash_backhandboast', 'squash_forehandboast'])},
        'selfregulationscp1': {'negativity': 0, 'positivity': 1},
        'selfregulationscp2': {'negativity': 0, 'positivity': 1},
        'spokenarabicdigits': {str(i+1): i for i in list(range(10))},
        'standwalkjump': {k: i for i, k in enumerate(['jumping', 'standing', 'walking'])},
        'uwavegesturelibrary': {str(float(i+1)): i for i in list(range(8))},
    }

    def __init__(self, 
        seed,
        dataset_name: str,
        max_trajectory_length: int=None,
        aws_profile: str=None, 
        s3_bucket_path: Union[str, Path, os.PathLike[str]]=None,
        use_threads: bool = False,
        **kwargs):
        self.__check_dataset(dataset_name=dataset_name)
        self._dataset_name = dataset_name
        self.aws_profile = aws_profile
        self.s3_bucket_path = s3_bucket_path
        self.use_threads = use_threads
        self.max_trajectory_length = max_trajectory_length
        self.setup_aws_session()
        """Initializes the UEADataset.

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
        
        super().__init__(dataset_name=dataset_name, seed=seed, **kwargs)
    
    def __len__(self, ):
        return len(self._targets)
    
    def __getitem__(self, index):
        return self._data[index]
    
    def download(self):
        now = datetime.now()
        #if self.training_method.lower() == TrainingMethodOptions.centralized:
        self.dataset_dir_paths['centralized_instance'] = osp.join(self.dataset_dir_paths['centralized'], now.strftime("%Y-%m-%d_%H:%M:%S"))
        #elif self.training_method.lower() == TrainingMethodOptions.federated:
        #    self.dataset_dir_paths['federated_instance'] = osp.join(self.dataset_dir_paths['federated'], now.strftime("%Y-%m-%d_%H:%M:%S"))

        if self.aws_profile is not None or self.s3_bucket_path is not None: # load data from aws bucket
            self.cli_logger.info(f'Downloading {self._dataset_name} from S3 Bucket {self.s3_bucket_path}!')
            
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
                self.cli_logger.info(f'Downloading {self._dataset_name} from source!')

                X, y, train_meta_data = load_classification(name=self._dataset_name, return_metadata=True)
                X_train, y_train, train_meta_data = load_classification(name=self._dataset_name, split='train', return_metadata=True)
                X_test, y_test, meta_data = load_classification(name=self._dataset_name, split='test', return_metadata=True)
                
                # storing
                self.store_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self.__raw_file_names['X']), obj=X, write='wb')
                self.store_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self.__raw_file_names['y']), obj=y, write='wb')
                self.store_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self.__raw_file_names['X_train']), obj=X_train, write='wb')
                self.store_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self.__raw_file_names['y_train']), obj=y_train, write='wb')
                self.store_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self.__raw_file_names['X_test']), obj=X_test, write='wb')
                self.store_asset(format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self.__raw_file_names['y_test']), obj=y_test, write='wb')
            else:
                self.cli_logger.info(f'Found raw {self._dataset_name} dataset!')
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
        
        if self._label2idx[self._dataset_name.lower()] is not None:
            y = torch.from_numpy(np.array([self._label2idx[self._dataset_name.lower()][k] for k in y]))
            y_train = torch.from_numpy(np.array([self._label2idx[self._dataset_name.lower()][k] for k in y_train]))
            y_test = torch.from_numpy(np.array([self._label2idx[self._dataset_name.lower()][k] for k in y_test]))
        
        self._data = [
            UEAData(data=X[i], label=y[i])
            for i in range(len(X))
        ]
        self._targets = y

        self._train_data = [
            UEAData(data=X_train[i], label=y_train[i])
            for i in range(len(X_train))
        ]
        self._train_targets = y_train

        self._test_data = [
            UEAData(data=X_test[i], label=y_test[i])
            for i in range(len(X_test))
        ]
        self._test_targets = y_test

    def process(self, **kwargs):
        #if self.training_method.lower() == TrainingMethodOptions.centralized:
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
            #'training_method': TrainingMethodOptions.centralized,
            'seed': self.seed,
        }
        
        self._process_shared()
        
        if not osp.exists(osp.join(self.dataset_dir_paths['centralized_instance'])):
            os.makedirs(osp.join(self.dataset_dir_paths['centralized_instance']))
        
        self.store_asset(format='json', file_path=self.dataset_dir_paths['config'], obj=config, write='w')   
        return config

    def process_centralized_load_from_config(self, instance: str, config: Dict, **kwargs):
        self.cli_logger.info(f'Loading Dataset {instance}!')
        self._process_shared()
    
    def _process_shared(self):
        """
        Shared processing method!
        """
        if self._dataset_name.lower() in [DatasetOptions.eigenworms, DatasetOptions.cricket, 
                                          DatasetOptions.motorimagery, DatasetOptions.standwalkjump]:
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

        self.train_dataset = UEASubDataset(data=self._train_data)
        self.val_dataset = UEASubDataset(data=self._test_data)
        self.test_dataset = None

    def _check_if_centralized_dataset_exists(self, 
        load_from_config, ds_config, tmp_config
        ):
        # check 
        # do not include instance key here
        if tmp_config['seed'] == self.seed:
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
            self.prefix = f'data/{self._dataset_name.lower()}'
            self.s3_buckets = list(self.s3.Bucket(self.s3_bucket_path).objects.filter(Prefix=f'{self.prefix}/processed/centralized/'))
    
    def __check_dataset(self, dataset_name: str):
        assert dataset_name.lower() in DatasetOptions.uea, f'{dataset_name.lower()} is not in {DatasetOptions.uea}'
    

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
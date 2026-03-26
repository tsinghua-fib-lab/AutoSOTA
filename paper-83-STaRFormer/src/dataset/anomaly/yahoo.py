import os
import os.path as osp
import pickle
import numpy as np
import torch

from pathlib import Path
from typing import List, Dict, Union
from datetime import datetime

from ..base import BaseDataset

from src.utils import TrainingMethodOptions
from src.utils.datasets.anomly import YahooData, YahooSubDataset, log_stats, sliding_window_slice

import boto3
import aeon


__all__ = ['YahooDataset']


class YahooDataset(BaseDataset):
    _url = ""

    _raw_files = {
        'train': None,
        'test': None,
        'processed': 'yahoo.pkl', 
    }
    __config_filename = 'config.json'
    __name = 'Yahoo'

    def __init__(self, 
                 seed, 
                 training_method: str=None,
                 aws_profile: str=None, 
                 s3_bucket_path: Union[str, Path, os.PathLike[str]]=None,
                 use_threads: bool = False,
                 window_size: int=1024,
                 stride: int=512,
                 **kwargs) -> None:
        self.training_method = training_method
        self.aws_profile = aws_profile
        self.s3_bucket_path = s3_bucket_path
        self.use_threads = use_threads
        self._window_size = window_size
        self._stride = stride
        self.setup_aws_session()
        super().__init__(self.__name.lower(), seed, **kwargs)
    

    def __len__(self, ):
        return len(self._targets)
    
    def __getitem__(self, index):
        return self._data[index]
    
    def download(self):
        now = datetime.now()
        if self.training_method.lower() == TrainingMethodOptions.centralized:
            self.dataset_dir_paths['centralized_instance'] = osp.join(self.dataset_dir_paths['centralized'], now.strftime("%Y-%m-%d_%H:%M:%S"))
        elif self.training_method.lower() == TrainingMethodOptions.federated:
            raise NotImplementedError

        if self.aws_profile is not None or self.s3_bucket_path is not None: # load data from aws bucket
            raise NotImplementedError
        
        else:
            if not osp.exists(self.dataset_dir_paths['raw']):
                os.makedirs(self.dataset_dir_paths['raw'])
            
            if not osp.exists(osp.join(self.dataset_dir_paths['raw'], self._raw_files['processed'])):
                raise RuntimeError(f"Processed pickle file does not exist ({osp.join(self.dataset_dir_paths['raw'], self._raw_files['processed'])}). Please download the raw data from https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70 and ensure it has been processed with ../data/yahoo/yahoo_preprocessing.py.")
            
            self._data_pkl = self.load_asset(
                format='pickle', file_path=osp.join(self.dataset_dir_paths['raw'], self._raw_files['processed']),
                read='rb'
            )
            #self._data = self._data_pkl['all_train_data'] + self._data_pkl['all_test_data']
            self._train_data_raw = self._data_pkl['all_train_data']
            self._train_targets_raw = self._data_pkl['all_train_labels']
            self._train_timestamps_raw = self._data_pkl['all_train_timestamps']

            self._test_data_raw = self._data_pkl['all_test_data']
            self._test_targets_raw = self._data_pkl['all_test_labels']
            self._test_timestamps_raw = self._data_pkl['all_test_timestamps']
            
            self._delay = self._data_pkl['delay']

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
        
        elif self.training_method.lower() == TrainingMethodOptions.federated:
            raise NotImplementedError
        
        self._max_seq_len = None
        if self._window_size is not None:
            self._max_seq_len = self._window_size
        
        for sample in self._data:
            if not hasattr(self, '_max_seq_len'):
                self._max_seq_len = sample.data.size(0)
            
            if self._max_seq_len is None:
                self._max_seq_len = sample.data.size(0)
            elif self._max_seq_len <= sample.data.size(0):
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

    def process_shared(self):
        self._train_data = []
        self._test_data = []
        # train 
        for key, values in self._train_data_raw.items():
            if self._window_size is not None:
                train_dws, train_lws, train_tsws = sliding_window_slice(
                    values, self._train_targets_raw[key], self._train_timestamps_raw[key], 
                    window_size=self._window_size, stride=self._stride)
                
                for i in range(len(train_dws)):
                    self._train_data.append(
                        YahooData(
                            seq_id=f'{key}_w{i}',
                            data=torch.from_numpy(train_dws[i]).type(torch.float32), 
                            label=torch.from_numpy(train_lws[i]).type(torch.int32), 
                            timestamps=torch.from_numpy(train_tsws[i]),
                        )
                    )
            else:
                self._train_data.append(
                    YahooData(
                        seq_id=f'{key}',
                        data=torch.from_numpy(values).type(torch.float32), 
                        label=torch.from_numpy(self._train_targets_raw[key]).type(torch.int32), 
                        timestamps=torch.from_numpy(self._train_timestamps_raw[key]),
                    )
                )
        # test        
        for key, values in self._test_data_raw.items():
            if self._window_size is not None:
                test_dws, test_lws, test_tsws = sliding_window_slice(
                    values, self._test_targets_raw[key], self._test_timestamps_raw[key], 
                    window_size=self._window_size, stride=self._window_size)
                for i in range(len(test_dws)):
                    self._test_data.append(
                        YahooData(
                            seq_id=f'{key}_w{i}',
                            data=torch.from_numpy(test_dws[i]).type(torch.float32), 
                            label=torch.from_numpy(test_lws[i]).type(torch.int32), 
                            timestamps=torch.from_numpy(test_tsws[i]),
                        )
                    )
            else:
                self._test_data.append(
                    YahooData(
                        seq_id=f'{key}',
                        data=torch.from_numpy(values).type(torch.float32), 
                        label=torch.from_numpy(self._test_targets_raw[key]).type(torch.int32), 
                        timestamps=torch.from_numpy(self._test_timestamps_raw[key]),
                    )
                )
        
        self._data = self._train_data + self._test_data
    
    def process_centralized(self, instance: str, **kwargs):
        self.cli_logger.info(f'Processing ...!')
        config = {
            'instance': instance,
            'training_method': TrainingMethodOptions.centralized,
            'seed': self.seed,
            'window_size': self._window_size,
            'stride': self._stride,
        }
        if not osp.exists(osp.join(self.dataset_dir_paths['centralized_instance'])):
            os.makedirs(osp.join(self.dataset_dir_paths['centralized_instance']))

        # main processing function
        self.process_shared()

        self.train_dataset = YahooSubDataset(data=self._train_data)
        self.val_dataset = YahooSubDataset(data=self._test_data)
        self.test_dataset = None

        self.store_asset(format='json', file_path=self.dataset_dir_paths['config'], obj=config, write='w')   
        return config
    
    def process_centralized_load_from_config(self, instance: str, config: Dict, **kwargs):
        self.cli_logger.info(f'Loading Dataset {instance}!')

        # main processing function
        self.process_shared()

        self.train_dataset = YahooSubDataset(data=self._train_data)
        self.val_dataset = YahooSubDataset(data=self._test_data)
        self.test_dataset = None

    
    def _check_if_centralized_dataset_exists(self, 
        load_from_config, ds_config, tmp_config
        ):
        # check 
        # do not include instance key here
        if tmp_config['seed'] == self.seed and \
            tmp_config['window_size'] == self._window_size and \
            tmp_config['stride'] == self._stride:
            
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
            self.prefix = f'data/{self._dataset_name}'
            self.s3_buckets = list(self.s3.Bucket(self.s3_bucket_path).objects.filter(Prefix=f'{self.prefix}/processed/centralized/'))


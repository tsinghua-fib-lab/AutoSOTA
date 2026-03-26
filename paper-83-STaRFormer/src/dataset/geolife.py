import os
import os.path as osp
import yupi
import json 
import numpy as np
import random 
import ujson 
import pickle
import logging
import boto3
import pandas as pd

import torch

from pathlib import Path
from typing import List, Tuple, Dict, Union
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from datetime import datetime
from copy import deepcopy

from torch import Tensor
from torch.utils.data import Dataset

from .base import BaseDataset, BaseDataSubset
from src.utils import DatasetOptions, TrainingMethodOptions
import src.utils.datasets.geolife as geolife_utils


__all__ = [
    'GeoLifeSubDataset',
    'GeoLifeDataset'
]


class GeoLifeSubDataset(BaseDataSubset):
    """
    A dataset class for handling GeoLife data subsets.

    This class inherits from `BaseDataSubset` and is specifically designed to handle
    subsets of GeoLife data. It initializes the dataset with a list of `GeoLifeData` objects.

    Attributes:
        data (List[geolife_utils.GeoLifeData]): A list of GeoLifeData objects.

    Methods:
        __init__(data: List[geolife_utils.GeoLifeData]) -> None:
            Initializes the dataset with the provided GeoLife data.
    """

    def __init__(self, data: List[geolife_utils.GeoLifeData]) -> None:
        """
        Initializes the dataset with the provided GeoLife data.

        Args:
            data (List[geolife_utils.GeoLifeData]): A list of GeoLifeData objects.
        """
        super().__init__(data)

class GeoLifeSubMock(BaseDataSubset):
    def __init__(self, data: List[geolife_utils.GeoLifeData], divisor: int = 20) -> None:
        super().__init__(data)
        import logging
        logging.warning("!"*64+"\nWARNING\t\tUSING THE MOCK DATASET (GEOLIFE)\t\tWARNING\n" + "!"*64)
        self._divisor = divisor

    def __len__(self):
        return len(self._targets) // self._divisor

class GeoLifeDataset(BaseDataset):
    """
    GeoLifeDataset class for handling GeoLife trajectory data.

    This class inherits from BaseDataset and provides functionalities to load,
    process, and split the GeoLife trajectory data for training, validation, 
    and testing purposes.

    Attributes:
        _GEOLIFE_URL (str): URL to download the GeoLife dataset.
        _raw_geolife_data_folder (str): Folder name containing raw GeoLife data.
        label2idx (dict): Mapping from labels to indices.
        idx2label (dict): Mapping from indices to labels.
        __data_filename (str): Filename for the dataset file.
        __config_filename (str): Filename for the configuration file.
        __transforms_filename (str): Filename for the transforms file.
        __s3_prefix_processed_centralized (str): S3 prefix for processed centralized data.
        __drop_indices (str): Filename for drop indices.

    Methods:
        __getitem__(index): Returns the data at the specified index.
        __len__(): Returns the length of the dataset.
        download(**kwargs): Downloads the dataset.
        process(**kwargs): Processes the dataset.
        process_centralized_load_from_config(instance, config, **kwargs): Processes centralized data from configuration.
        process_centralized(instance, **kwargs): Processes centralized data.
        _setup_transforms(): Sets up data transforms.
        scale_data(test_indices=None): Scales the data.
        drop_corrupted_indices(**kwargs): Drops corrupted indices from the dataset.
        load_data_from_s3_bucket(): Loads data from the specified S3 bucket.
        _check_if_centralized_dataset_exists(load_from_config, ds_config, tmp_config): Checks if a centralized dataset configuration exists.
        _log_stats(**kwargs): Logs dataset statistics.
    """
    _GEOLIFE_URL = (
        "https://download.microsoft.com/download/F/4/8/"
        "F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip"
    )
    _raw_geolife_data_folder = "Geolife Trajectories 1.3/Data"

    label2idx = {k: v for k, v in geolife_utils.LABEL2IDX.items() if k is not None}
    idx2label = {k: v for k, v in geolife_utils.IDX2LABEL.items() if k is not None}

    __data_filename = 'geolife_data.pkl'
    __config_filename = 'config.json'
    __transforms_filename = 'transforms.pkl'

    __s3_prefix_processed_centralized = 'data/geolife/processed/centralized'
    __drop_indices = 'drop_indices.json'

    def __init__(self, 
        seed, 
        training_method: str=None,
        train_test_split: Dict[str, float]={'train': 0.7, 'val': 0.1, 'test': 0.2},
        identical_training_class_label_distribution: bool=True,
        max_trajectory_length: int = 1024,
        aws_profile: str=None, 
        s3_bucket_path: Union[str, Path, os.PathLike[str]]=None,
        use_threads: bool = False,
        **kwargs) -> None:
        """
        Args:
            seed (int): Random seed for reproducibility.
            training_method (str, optional): Method for training. Defaults to None.
            train_test_split (dict, optional): Proportions for train, validation, and test splits.
                Defaults to {'train': 0.7, 'val': 0.1, 'test': 0.2}.
            identical_training_class_label_distribution (bool, optional): Whether to maintain identical training class label distribution.
                Defaults to True.
            max_trajectory_length (int, optional): Maximum length of a trajectory. Defaults to 1024.
            aws_profile (str, optional): AWS profile for accessing S3. Defaults to None.
            s3_bucket_path (Union[str, Path, os.PathLike[str]], optional): S3 bucket path for loading data.
                Defaults to None.
            use_threads (bool, optional): Whether to use threads for loading data. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        self._dataset_class = GeoLifeSubDataset 
        self.training_method = training_method
        self.train_test_split = train_test_split
        self.identical_training_class_label_distribution = identical_training_class_label_distribution
        self.max_trajectory_length = max_trajectory_length
        self.aws_profile = aws_profile
        self.s3_bucket_path = s3_bucket_path
        if aws_profile is not None or s3_bucket_path is not None:
            self.use_threads = use_threads

        self._setup_transforms()
        super().__init__(dataset_name=DatasetOptions.geolife, seed=seed, **kwargs)
       
    def __getitem__(self, index) -> Tuple[str, Tensor, List | Tensor]:
        return self.data[index]
    
    def __len__(self) -> int:
        return len(self.targets)

    def download(self, **kwargs):
        """Implement from github repo!"""
        # general setups
        now = datetime.now()
        if self.training_method.lower() == TrainingMethodOptions.centralized:
            self.dataset_dir_paths['centralized_instance'] = osp.join(self.dataset_dir_paths['centralized'], now.strftime("%Y-%m-%d_%H:%M:%S"))
        elif self.training_method.lower() == TrainingMethodOptions.federated:
            self.dataset_dir_paths['federated_instance'] = osp.join(self.dataset_dir_paths['federated'], now.strftime("%Y-%m-%d_%H:%M:%S"))
        
        if self.aws_profile is not None or self.s3_bucket_path is not None: 
            # When AWS is used (either aws_profile or s3_bucket_path is provided), only processed data is loaded.
            # There is no need to download and run preprocessing.
            pass
        else:
            if not osp.exists(self.dataset_dir_paths['raw']):
                os.makedirs(self.dataset_dir_paths['raw'])

            if not osp.exists(osp.join(self.dataset_dir_paths['raw'], 'geolife.zip')):
                self.cli_logger.info(f'Downloading {DatasetOptions.geolife} from source!')
                geolife_utils.download(
                    url=self._GEOLIFE_URL,
                    dataset_name=DatasetOptions.geolife,
                    dataset_file_path=self.dataset_dir_paths['raw'] + '/geolife.zip',
                    dataset_path=self.dataset_dir_paths['raw'],
                    uncompress=True

                )
            else:
                self.cli_logger.info(f'Found raw {DatasetOptions.geolife} dataset!')
            
            if not osp.exists(osp.join(self.dataset_dir_paths['raw'], 'geolife.pkl.gzip')):
                self.cli_logger.info('Preprocessing dataset!')
                self.df_raw = geolife_utils.preprocess(
                    data_folder= osp.join(self.dataset_dir_paths['raw'], self._raw_geolife_data_folder),
                    raw_data_file_path=osp.join(self.dataset_dir_paths['raw'], 'geolife.pkl.gzip'),
                )
            else:
                self.cli_logger.info(f'Found preprocessed {DatasetOptions.geolife} dataset!')
                self.df_raw = pd.read_pickle(osp.join(self.dataset_dir_paths['raw'], 'geolife.pkl.gzip'), compression='gzip')

    def process(self, **kwargs):
        if self.training_method.lower() == TrainingMethodOptions.centralized:
            self.dataset_dir_paths['data'] = osp.join(self.dataset_dir_paths['centralized_instance'], self.__data_filename)
            self.dataset_dir_paths['config'] = osp.join(self.dataset_dir_paths['centralized_instance'], self.__config_filename)
            self.dataset_dir_paths['transforms'] = osp.join(self.dataset_dir_paths['centralized_instance'], self.__transforms_filename)

            if self.aws_profile is not None or self.s3_bucket_path is not None:
                _, s3_client, _, s3_buckets = self.load_data_from_s3_bucket()
                load_from_config, ds_config = self.check_if_centralized_dataset_exists_on_s3(
                    s3_client=s3_client, s3_buckets=s3_buckets
                )
            else:   
                load_from_config, ds_config = self.check_if_centralized_dataset_exists(**kwargs)
            
            if load_from_config:
                instance = ds_config['instance']
                params = {'instance': instance, 'config': ds_config}
                if self.aws_profile is not None or self.s3_bucket_path is not None:
                    params['s3_client'] = s3_client
                self.process_centralized_load_from_config(**params)
            else:
                instance = self.dataset_dir_paths['centralized_instance'].split("/")[-1]
                ds_config = self.process_centralized(instance=instance, **kwargs)
            
            # scale data
            drop_params = {}
            if self.aws_profile is not None or self.s3_bucket_path is not None:
                drop_params['instance'] = instance
                drop_params['s3_client'] = s3_client
            
            self.drop_corrupted_indices(**drop_params)
            self.set_data()
            #self.scale_data()
        
        self._max_seq_len = 0
        for sample in self._data:
            if self._max_seq_len <= sample.data.size(0):
                self._max_seq_len = sample.data.size(0)
        
        self._ds_config = ds_config
        self._log_stats(**{
            'method': TrainingMethodOptions.centralized, 
            'cli_logger': self.cli_logger,
            'data': self._data,
            'ds_config': ds_config
        })
    
    def process_centralized_load_from_config(self, instance: str, config: Dict, **kwargs):
        if self.aws_profile is not None or self.s3_bucket_path is not None: # load data from aws bucket
            self.cli_logger.info(f'Loading Dataset {instance} from S3 Bucket!')
            assert kwargs.get('s3_client', None) is not None
    
            prefix = f"{self.__s3_prefix_processed_centralized}/{instance}/{self.__data_filename}"
            s3_object = kwargs['s3_client'].get_object(Bucket=self.s3_bucket_path, Key=prefix)
            pkl = s3_object['Body'].read()
            data = pickle.loads(pkl)

            prefix = f"{self.__s3_prefix_processed_centralized}/{instance}/{self.__transforms_filename}"
            s3_object = kwargs['s3_client'].get_object(Bucket=self.s3_bucket_path, Key=prefix)
            pkl = s3_object['Body'].read()
            self.transforms = pickle.loads(pkl) 

        else: # load locally
            self.cli_logger.info(f'Loading Dataset {instance}!')
        
            # load datafile
            self.dataset_dir_paths['centralized_instance'] = osp.join(self.dataset_dir_paths['centralized'], instance)
            self.dataset_dir_paths['data'] = osp.join(self.dataset_dir_paths['centralized_instance'], self.__data_filename)
            self.dataset_dir_paths['transforms'] = osp.join(self.dataset_dir_paths['centralized_instance'], self.__transforms_filename)
            data = self.load_asset(format='pickle', file_path=self.dataset_dir_paths['data'], read='rb')
            self.transforms = self.load_asset(format='pickle', file_path=self.dataset_dir_paths['transforms'], read='rb')

        # set data in class
        self._data = data
        self.set_data()

        # initialize datasets
        if isinstance(config['indices']['train'][0], str):
            train_data = [sample for sample in data if sample.traj_id in config['indices']['train']]
            val_data = [sample for sample in data if sample.traj_id in config['indices']['val']] if config['indices'].get('val', None) is not None else None
            test_data = [sample for sample in data if sample.traj_id in config['indices']['test']]

        self.train_dataset = self._dataset_class(data=train_data)
        self.val_dataset = self._dataset_class(data=val_data) if val_data is not None else None
        self.test_dataset = self._dataset_class(data=test_data)

    def process_centralized(self, instance: str, **kwargs):
        self.cli_logger.info(f'Processing ...!')
        config = {
            'instance': instance,
            'training_method': TrainingMethodOptions.centralized,
            'train_test_split': dict(self.train_test_split),
            'identical_training_class_label_distribution': self.identical_training_class_label_distribution,
            'seed': self.seed,
        }
        if self.max_trajectory_length is not None:
            config['max_trajectory_length'] = self.max_trajectory_length
        
        if not osp.exists(osp.join(self.dataset_dir_paths['centralized_instance'])):
            os.makedirs(osp.join(self.dataset_dir_paths['centralized_instance']))
        
        df = deepcopy(self.df_raw)
        
        # process raw data 
        df = geolife_utils.create_new_labels_for_training(
                geolife_utils.filter_trajectories_velocity(
                    geolife_utils.create_features(
                        geolife_utils.filter_utm_50(
                            geolife_utils.filter_1(df=df)
                        )
                    )
                )
            )
        
        self._data = geolife_utils.prepare_df(df, 
            feature_columns=['x (m)', 'y (m)', 'v (m/s)', 'vx (m/s)', 'vy (m/s)', \
            'a (m/s^2)', 'ax (m/s^2)', 'ay (m/s^2)', 'time', 'dt']
        )
        if self.max_trajectory_length:
            #data_short = []
            for sample in tqdm(self._data, desc="Shorten Data"):
                traj_short = geolife_utils.shorten_trajectory(sample.data, max_trajectory_length=self.max_trajectory_length)
                sample.data = traj_short
                sample.seq_len = traj_short.size(0)
            # sanity check
            for sample in tqdm(self._data, desc="Shorten Data - Sanity Check"):
                assert len(sample.data) <= self.max_trajectory_length

        self.set_data()

        datasets = geolife_utils.create_train_test_split(
            data=self._data, targets=self._targets, train_test_split=self.train_test_split, 
            identical_training_class_label_distribution=self.identical_training_class_label_distribution
        )
        
        self.train_dataset = self._dataset_class(data=datasets['train_data'])
        self.val_dataset = self._dataset_class(data=datasets['val_data']) if datasets.get('val_data', None) is not None else None
        self.test_dataset = self._dataset_class(data=datasets['test_data'])

        # scale data
        if not osp.exists(self.dataset_dir_paths["centralized_instance"]):
            os.makedirs(self.dataset_dir_paths["centralized_instance"])

        self.scale_data()

        # store data and config
        config['indices'] = {
            'train': self.train_dataset.sequence_ids,
            'val': self.val_dataset.sequence_ids if datasets.get('val_data', None) is not None else None,
            'test': self.test_dataset.sequence_ids
        }
    
        if not osp.exists(self.dataset_dir_paths["centralized_instance"]):
            os.makedirs(self.dataset_dir_paths["centralized_instance"])
        
        self.store_asset(format='pickle', file_path=self.dataset_dir_paths['data'], obj=self._data, write='wb')
        self.store_asset(format='json', file_path=self.dataset_dir_paths['config'], obj=config, write='w')   
        return config

    def _setup_transforms(self):
        self.transforms = {
            'x': StandardScaler(),
            'y': StandardScaler(),
            'v': StandardScaler(),
            'vx': StandardScaler(),
            'vy': StandardScaler(),
            'a': StandardScaler(),
            'ax': StandardScaler(),
            'ay': StandardScaler(),
            #'dx': StandardScaler(),
            #'dy': StandardScaler(),
            #'dvx': StandardScaler(),
            #'dvy': StandardScaler(),
            't': StandardScaler(),
            'dt': StandardScaler(),
        }

    def scale_data(self, test_indices: set=None):
        if test_indices is not None:
            # scale data based on real train sequences
            train_trajs = []
            for sample in self._data:
                traj_id = "_".join(sample.traj_id.split('_')[:-2]) if sample.traj_id.split('_')[-2] == 'A' else sample.traj_id
                if traj_id not in test_indices:
                    train_trajs.append(sample.data)
        else:
            train_trajs = self.train_dataset.sequences + self.val_dataset.sequences if self.val_dataset is not None else self.train_dataset.sequences
        trajectories_merge = torch.vstack(train_trajs)
        for idx, (key, transform) in enumerate(self.transforms.items()):
            if transform != None:
                values = trajectories_merge[:, idx]
                # fit transform 
                _ = transform.fit_transform(values.reshape(-1, 1)) 

        self.store_asset(format='pickle', file_path=self.dataset_dir_paths['transforms'], obj=self.transforms, write='wb')

        if test_indices is not None:
            data_scaled = []

            for sample in tqdm(self._data, desc="Scaling features"):
                data_item_scaled = []
                for idx, (key, transform) in enumerate(self.transforms.items()):   
                    if transform != None:
                        data_item_scaled.append(transform.transform(sample.data[:, idx].reshape(-1, 1)))
                    else:
                        data_item_scaled.append(sample.data[:, idx].reshape(-1,1))

                data_scaled.append((
                    geolife_utils.GeoLifeData(
                        traj_id=sample.traj_id, 
                        data=torch.from_numpy(np.concatenate(data_item_scaled, axis=1)), 
                        label=sample.label
                    )
                ))                
            self._data = data_scaled
        
        else:

            data_scaled = []
            train_data_scaled = []
            val_data_scaled = []
            test_data_scaled = []
            train_ids = self.train_dataset.sequence_ids
            val_ids = self.val_dataset.sequence_ids if self.val_dataset is not None else None
            test_ids = self.test_dataset.sequence_ids

            for sample in tqdm(self.data, desc="Scaling features"):
                data_item_scaled = []
                for idx, (key, transform) in enumerate(self.transforms.items()):   
                    if transform != None:
                        data_item_scaled.append(transform.transform(sample.data[:, idx].reshape(-1, 1)))
                    else:
                        data_item_scaled.append(sample.data[:, idx].reshape(-1,1))

                data_scaled.append((
                    geolife_utils.GeoLifeData(
                        traj_id=sample.traj_id, 
                        data=torch.from_numpy(np.concatenate(data_item_scaled, axis=1)), 
                        label=sample.label
                    )
                ))
                if sample.traj_id in train_ids:
                    train_data_scaled.append((
                        geolife_utils.GeoLifeData(
                            traj_id=sample.traj_id, 
                            data=torch.from_numpy(np.concatenate(data_item_scaled, axis=1)), 
                            label=sample.label
                        )
                    ))
                if self.val_dataset is not None:
                    if sample.traj_id in val_ids:
                        val_data_scaled.append((
                            geolife_utils.GeoLifeData(
                                traj_id=sample.traj_id, 
                                data=torch.from_numpy(np.concatenate(data_item_scaled, axis=1)), 
                                label=sample.label
                            )
                        ))
                if sample.traj_id in test_ids:
                    test_data_scaled.append((
                        geolife_utils.GeoLifeData(
                            traj_id=sample.traj_id, 
                            data=torch.from_numpy(np.concatenate(data_item_scaled, axis=1)), 
                            label=sample.label
                        )
                    ))
        
            self._data = data_scaled
            self.train_dataset=self._dataset_class(data=train_data_scaled)
            self.val_dataset=self._dataset_class(data=val_data_scaled)
            self.test_dataset=self._dataset_class(data=test_data_scaled)

    def drop_corrupted_indices(self, **kwargs):
        if self.aws_profile is not None or self.s3_bucket_path is not None: # load data from aws bucket
            assert kwargs.get('instance', None) is not None
            assert kwargs.get('s3_client', None) is not None
    
            try:
                prefix = f"{self.__s3_prefix_processed_centralized}/{kwargs['instance']}/{self.__drop_indices}"
                s3_object = kwargs['s3_client'].get_object(Bucket=self.s3_bucket_path, Key=prefix)
                binary = s3_object['Body'].read()
                self.drop_indices = ujson.loads(binary.decode('utf-8'))
                cleaned_data = [
                    data 
                    for data in tqdm(self._data, desc="Dropping corrupt indices")
                    if data.traj_id not in self.drop_indices['drop_indices']
                ]
                self._data = cleaned_data
                self.train_dataset=self._dataset_class(data=[data for data in self.train_dataset if data.traj_id not in self.drop_indices['drop_indices']])
                self.val_dataset=self._dataset_class(data=[data for data in self.val_dataset if data.traj_id not in self.drop_indices['drop_indices']])
                self.test_dataset=self._dataset_class(data=[data for data in self.test_dataset if data.traj_id not in self.drop_indices['drop_indices']])
            except Exception as e:
                self.cli_logger.info(prefix)
                self.cli_logger.warning(e)

        else:

            file_path = osp.join(self.dataset_dir_paths["centralized_instance"], self.__drop_indices)
            self.drop_indices = None

            if osp.exists(file_path):
                self.drop_indices = self.load_asset(format='json', file_path=file_path, read='r')
                cleaned_data = [
                    data 
                    for data in tqdm(self._data, desc="Dropping corrupt indices")
                    if data.traj_id not in self.drop_indices['drop_indices']
                ]
                self._data = cleaned_data
                self.train_dataset=self._dataset_class(data=[data for data in self.train_dataset if data.traj_id not in self.drop_indices['drop_indices']])
                self.val_dataset=self._dataset_class(data=[data for data in self.val_dataset if data.traj_id not in self.drop_indices['drop_indices']])
                self.test_dataset=self._dataset_class(data=[data for data in self.test_dataset if data.traj_id not in self.drop_indices['drop_indices']])
                
    def load_data_from_s3_bucket(self):
        session = boto3.Session(profile_name=self.aws_profile) if self.aws_profile is not None else boto3.Session()
        s3_client = session.client('s3')
        s3 = session.resource('s3')
        return (
            session, 
            s3_client, 
            s3, 
            list(s3.Bucket(self.s3_bucket_path).objects.filter(Prefix=self.__s3_prefix_processed_centralized))
        )
    
    def _check_if_centralized_dataset_exists(self, 
        load_from_config, ds_config, tmp_config
        ):
        if tmp_config['seed'] == self.seed and \
            tmp_config['train_test_split'] == self.train_test_split and \
            tmp_config['identical_training_class_label_distribution'] == self.identical_training_class_label_distribution:
            if self.max_trajectory_length is not None:
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
        geolife_utils.log_stats(**kwargs)

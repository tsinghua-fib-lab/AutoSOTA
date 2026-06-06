import os

import torch
#from torch.nn.utils.rnn import 

from pathlib import Path
from typing import Dict, Literal, Union
from torch.nn.utils.rnn import pack_sequence, pad_sequence

from src.dataset.anomaly import KPIDataset, YahooDataset
from src.utils.datasets.anomly import KPIData, KPIBatchData, YahooData, YahooBatchData
from src.utils import DatasetOptions

from .base import BaseDatamodule


__all__ = ['AnomalyDatamodule']


def collate_packed_kpi(batch):
    ids = []
    data = []
    label = []
    timestamps = []
    seq_len = []
    for sample in batch:
        # sample is TSRData Type
        ids.append(sample.seq_id)
        data.append(sample.data.unsqueeze(1)) 
        label.append(sample.label.unsqueeze(0)) 
        seq_len.append(sample.seq_len.unsqueeze(0)) 
        timestamps.append(sample.timestamps.unsqueeze(0)) 

    data_packed = pack_sequence(data, enforce_sorted=False)
    label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    timestamps = torch.concat(timestamps)
    
    return KPIBatchData(
        seq_id=ids,
        data=data_packed, 
        label=label, 
        seq_len=seq_len, 
        timestamps=timestamps
    )

def collate_padded_kpi(batch):
    ids = []
    data = []
    label = []
    timestamps = []
    seq_len = []
    for sample in batch:
        # sample is TSRData Type
        ids.append(sample.seq_id)
        data.append(sample.data.unsqueeze(1)) 
        label.append(sample.label) 
        seq_len.append(sample.seq_len.unsqueeze(0)) 
        timestamps.append(sample.timestamps) 

    data_padded = pad_sequence(sequences=data, batch_first=False, padding_value=0) # [bs, N, D] 
    label_padded = pad_sequence(sequences=label, batch_first=True, padding_value=-1) # [bs, N, D] 
    #label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    #timestamps = torch.concat(timestamps)

    return KPIBatchData(
        seq_id=ids,
        data=data_padded, 
        label=label_padded, 
        seq_len=seq_len, 
        timestamps=timestamps
    )

def collate_packed_yahoo(batch):
    ids = []
    data = []
    label = []
    timestamps = []
    seq_len = []
    for sample in batch:
        # sample is TSRData Type
        ids.append(sample.seq_id)
        data.append(sample.data.unsqueeze(1)) 
        label.append(sample.label.unsqueeze(0)) 
        seq_len.append(sample.seq_len.unsqueeze(0)) 
        timestamps.append(sample.timestamps.unsqueeze(0)) 

    data_packed = pack_sequence(data, enforce_sorted=False)
    label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    timestamps = torch.concat(timestamps)
    
    return YahooBatchData(
        seq_id=ids,
        data=data_packed, 
        label=label, 
        seq_len=seq_len, 
        timestamps=timestamps
    )

def collate_padded_yahoo(batch):
    ids = []
    data = []
    label = []
    timestamps = []
    seq_len = []
    for sample in batch:
        # sample is TSRData Type
        ids.append(sample.seq_id)
        data.append(sample.data.unsqueeze(1)) 
        label.append(sample.label) 
        seq_len.append(sample.seq_len.unsqueeze(0)) 
        timestamps.append(sample.timestamps) 

    data_padded = pad_sequence(sequences=data, batch_first=False, padding_value=0) # [bs, N, D] 
    label_padded = pad_sequence(sequences=label, batch_first=True, padding_value=-1) # [bs, N, D] 
    #label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    #timestamps = torch.concat(timestamps)

    return YahooBatchData(
        seq_id=ids,
        data=data_padded, 
        label=label_padded, 
        seq_len=seq_len, 
        timestamps=timestamps
    )


class AnomalyDatamodule(BaseDatamodule):
    """AnomalyDatamodule

    A Lightning DataModule that provides dataset-specific loading for anomaly
    detection experiments. It supports KPI and Yahoo datasets (as of this
    implementation) and exposes configurable batching, windowing, and
    collation strategies. 

    Attributes:
        dataset (object): The chosen dataset class instance (e.g., KPI or Yahoo
        dataset) with loaded train/val/test splits.
        collate_fn (callable): The collate function used to batch data for the
        selected dataset (padded or packed sequences depending on pad_sequence).
        num_train (int|None): Optional limit on the number of training samples.
        num_val (int|None): Optional limit on the number of validation samples.
        num_test (int|None): Optional limit on the number of test samples.
    """
    def __init__(self, 
                 dataset: str,
                 batch_size: int, 
                 num_workers: int, 
                 seed: int,
                 training_method: str=None,
                 aws_profile: str=None, 
                 s3_bucket_path: Union[str, Path, os.PathLike[str]]=None,
                 use_threads: bool = False,
                 window_size: int=1024,
                 stride: int=512,
                 pad_sequence: bool=True,
                 val_batch_size: int=None,
                 test_batch_size: int=None, 
                 num_train: int=None,
                 num_val: int=None,
                 num_test: int=None,
                 **kwargs
        ) -> None:
        """Initialize the AnomalyDatamodule.

        Args:
            dataset (str): Identifier of the dataset to load. Supported values are
            DatasetOptions.kpi and DatasetOptions.yahoo.
            batch_size (int): Batch size for training/validation/testing.
            num_workers (int): Number of worker processes for data loading.
            seed (int): Random seed for reproducibility.
            training_method (str, optional): Training method/configuration key.
            aws_profile (str, optional): AWS profile name for S3 data access.
            s3_bucket_path (str | Path | os.PathLike, optional): S3 bucket path to load data from.
            use_threads (bool, optional): Whether to load data using threads. Default: False.
            window_size (int, optional): Size of the sliding window for sequence data.
            stride (int, optional): Stride between windows.
            pad_sequence (bool, optional): If True, use padded collate; otherwise, packed.
            val_batch_size (int, optional): Batch size for the validation set (overrides default if provided).
            test_batch_size (int, optional): Batch size for the test set (overrides default if provided).
            num_train (int, optional): Limit on number of training samples.
            num_val (int, optional): Limit on number of validation samples.
            num_test (int, optional): Limit on number of test samples.
            **kwargs: Additional keyword arguments passed to the base DataModule.
        """
        dm_params = {
            'seed': seed,
            'training_method': training_method,
            'aws_profile': aws_profile,
            's3_bucket_path': s3_bucket_path,
            'use_threads': use_threads,
            'window_size': window_size,
            'stride': stride,
        }

        dataset_mapping = {
            DatasetOptions.kpi: (KPIDataset, collate_padded_kpi if pad_sequence else collate_packed_kpi),
            DatasetOptions.yahoo: (YahooDataset, collate_padded_yahoo if pad_sequence else collate_packed_yahoo),
        }

        if dataset not in dataset_mapping:
            raise NotImplementedError(f'{dataset} is not implemented')
        
        dataset_class, collate_fn = dataset_mapping[dataset]
        
        self.dataset = dataset_class(**dm_params)
        self.collate_fn = collate_fn

        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test

        super().__init__(batch_size, num_workers, seed, val_batch_size, test_batch_size, **kwargs)
    
    def setup(self, stage: str=None) -> None:
        if self.num_train is not None:
            self._train_dataset = self.dataset.train_dataset[:self.num_train]
        else:
            self._train_dataset = self.dataset.train_dataset
        
        if self.num_val is not None:
            self._val_dataset = self.dataset.val_dataset[:self.num_val]
        else:
            self._val_dataset = self.dataset.val_dataset
        
        if self.num_test is not None and self.dataset.test_dataset is not None:
            self._test_dataset = self.dataset.test_dataset[:self.num_test]
        else:
            self._test_dataset = self.dataset.test_dataset
        
        return super().setup(stage=stage)

import os
import os.path as osp
from pathlib import Path
from typing import List, Dict, Union, Literal

import torch 
from torch.nn.utils.rnn import pad_sequence

from .base import BaseDatamodule
from src.dataset.p12 import P12Dataset

from src.utils import DatasetOptions
import src.utils.datasets.irregular_sampling as utils


__all__ = [
    "P12Datamodule",
]


def collate_fn(batch):
    """
    Collate function for packing sequences in a batch.

    This function processes a batch of data items by extracting data sequences,
    labels, and sequence lengths. It then packs the sequences using PyTorch's `pack_sequence` 
    function, which is useful for handling variable-length sequences in RNNs.

    Args:
        batch (list): List of data items, where each item is an object containing `traj_id`, `data`, 
                      `label`, and `seq_len` attributes.

    Returns:
        utils.P12BatchData: An object containing packed data, labels, sequence lengths, 
                                        and trajectory IDs.
    """
    seq_id = []
    data = []
    label = []
    seq_len = []
    extended_static = []
    time = []
    demogr_desc = []
    for sample in batch:
        seq_id.append(sample.seq_id)
        data.append(sample.data[:sample.seq_len, ...]) 
        label.append(sample.label.unsqueeze(0)) # add additional dimension for concatenation
        seq_len.append(sample.seq_len.unsqueeze(0)) # add additional dimension for concatenation
        extended_static.append(sample.extended_static.unsqueeze(0))
        time.append(sample.time.reshape(1,-1))
        demogr_desc.append(sample.demogr_desc)
    
    data = torch.concat(data, dim=0).permute(1, 0, 2) # [bs, N, D] --> [N, D, bs]
    label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    extended_static = torch.concat(extended_static)
    time = torch.concat(time)
    return utils.P12BatchData(seq_id=seq_id, data=data, label=label, 
                              seq_len=seq_len, extended_static=extended_static, 
                              time=time, demogr_desc=demogr_desc)


def collate_fn_padded_seq(batch: List[utils.P12Data]):
    """
    Collate function for padding sequences in a batch.

    This function processes a batch of data items by extracting data sequences,
    labels, and sequence lengths. It then pads the sequences using PyTorch's `pad_sequence` 
    function, which is useful for handling variable-length sequences in RNNs by padding them to 
    the same length.

    Args:
        batch (list): List of data items, where each item is an object containing `traj_id`, `data`, 
                      `label`, and `seq_len` attributes.

    Returns:
        utils.P12BatchData: An object containing padded data, labels, sequence lengths, 
    """
    seq_id = []
    data = []
    label = []
    seq_len = []
    extended_static = []
    time = []
    demogr_desc = []
    for sample in batch:
        seq_id.append(sample.seq_id) # only select length for with data was recorded
        data.append(sample.data[:sample.seq_len, ...]) 
        label.append(sample.label.unsqueeze(0)) # add additional dimension for concatenation
        seq_len.append(sample.seq_len.unsqueeze(0)) # add additional dimension for concatenation
        extended_static.append(sample.extended_static.unsqueeze(0))
        time.append(sample.time.reshape(1,-1))
        demogr_desc.append(sample.demogr_desc)
    
    data_padded = pad_sequence(sequences=data, batch_first=False, padding_value=0)
    #data_padded = data_padded.permute(1, 0, 2) # [bs, N, D] --> [N, D, bs]

    label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    extended_static = torch.concat(extended_static)
    time = torch.concat(time)
    return utils.P12BatchData(seq_id=seq_id, data=data_padded, label=label, 
                              seq_len=seq_len, extended_static=extended_static, 
                              time=time, demogr_desc=demogr_desc)



class P12Datamodule(BaseDatamodule):
    """
    P12Datamodule class for handling P12 dataset within the PyTorch Lightning framework.

    This class inherits from BaseDatamodule and provides functionalities to set up the P12 dataset,
    configure data loaders, and manage batch sizes and worker threads for training, validation, 
    and testing purposes.

    Attributes:
        dataset (P12Dataset): The P12 dataset instance.
        collate_fn (callable): Function to merge a list of samples to form a mini-batch.
        num_train (int): Number of training samples.
        num_val (int): Number of validation samples.
        num_test (int): Number of test samples.

    Methods:
        setup(stage=None): Sets up the datasets for the specified stage.
        train_dataloader(): Returns the data loader for the training dataset.
        val_dataloader(): Returns the data loader for the validation dataset.
        test_dataloader(): Returns the data loader for the test dataset.
        configure_dataloader(batch_size, num_workers, seed, mode='train'): Configures and returns a data loader for the specified mode.
        configure_cli_logger(log_level=logging.INFO): Configures the command-line interface logger.
        train_dataset(): Returns the training dataset.
        val_dataset(): Returns the validation dataset.
        test_dataset(): Returns the test dataset.
    """
    def __init__(
        self,
        batch_size: int, 
        num_workers: int, 
        seed: int,
        training_method: str,
        train_split_index: int,
        aws_profile: str=None,
        s3_bucket_path: Union[str, Path, os.PathLike[str]]=None,
        preprocessing_method: Literal['raindrop', 'vitst']=None,
        min_seq_length: int=None,
        percentile_of_features_used: int=None, 
        balance: Literal['random', 'smote']=None,
        upsample_percentage: float=1.0,
        use_threads: bool = False,
        val_batch_size: int=None,
        test_batch_size: int=None, 
        num_train: int=None,
        num_val: int=None,
        num_test: int=None,
        pad_sequence: bool=True,
        **kwargs
        ) -> None:
        """
        Args:
            batch_size (int): The batch size for training data.
            num_workers (int): The number of worker threads for data loading.
            seed (int): Random seed for reproducibility.
            training_method (str): Method for training.
            train_split_index (int): Index for splitting the training data.
            aws_profile (str, optional): AWS profile for accessing S3. Default: None.
            s3_bucket_path (Union[str, Path, os.PathLike[str]], optional): S3 bucket path for loading data. Default: None.
            use_threads (bool, optional): Whether to use threads for loading data. Default: False.
            val_batch_size (int, optional): The batch size for validation data. Default: None.
            test_batch_size (int, optional): The batch size for test data. Default: None.
            num_train (int, optional): Number of training samples. Default: None.
            num_val (int, optional): Number of validation samples. Default: None.
            num_test (int, optional): Number of test samples. Default: None.
            pad_sequence (bool, optional): Whether to pad sequences. Default: True.
            **kwargs: Additional keyword arguments.
        """

        dm_params = {
            'seed': seed,
            'training_method': training_method,
            'train_split_index': train_split_index, 
            'aws_profile': aws_profile,
            's3_bucket_path': s3_bucket_path,
            'use_threads': use_threads,
            'preprocessing_method': preprocessing_method,
            'min_seq_length': min_seq_length,
            'percentile_of_features_used': percentile_of_features_used,
            'balance': balance,
            'upsample_percentage': upsample_percentage,
        }
        self.dataset = P12Dataset(**dm_params)
        if pad_sequence:
            self.collate_fn = collate_fn_padded_seq
        else:
            self.collate_fn = collate_fn
            
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        super().__init__(batch_size, num_workers, seed=seed, val_batch_size=val_batch_size, test_batch_size=test_batch_size, **kwargs) 
        
    def setup(self, stage: str=None) -> None:
        if self.num_train is not None:
            self._train_dataset = self.dataset.train_dataset[:self.num_train]
        else:
            self._train_dataset = self.dataset.train_dataset
        
        if self.num_train is not None:
            self._val_dataset = self.dataset.val_dataset[:self.num_val]
        else:
            self._val_dataset = self.dataset.val_dataset
        
        if self.num_train is not None:
            self._test_dataset = self.dataset.test_dataset[:self.num_test]
        else:
            self._test_dataset = self.dataset.test_dataset
        #self._test_dataset = None
        
        return super().setup(stage=stage)

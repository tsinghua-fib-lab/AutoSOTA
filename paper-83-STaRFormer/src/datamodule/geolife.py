import os
import os.path as osp
from pathlib import Path

from typing import List, Dict, Union

import torch 
from torch.nn.utils.rnn import pack_sequence, pad_sequence

from .base import BaseDatamodule
from src.dataset import GeoLifeDataset, GeoLifeSubDataset
import src.utils.datasets.geolife as geolife_utils


__all__ = [
    "GeoLifeDatamodule",
]


def collate_fn_packed_seq(batch):
    """
    Collate function for packing sequences in a batch.

    This function processes a batch of data items by extracting trajectory IDs, data sequences,
    labels, and sequence lengths. It then packs the sequences using PyTorch's `pack_sequence` 
    function, which is useful for handling variable-length sequences in RNNs.

    Args:
        batch (list): List of data items, where each item is an object containing `traj_id`, `data`, 
                      `label`, and `seq_len` attributes.

    Returns:
        geolife_utils.GeoLifeBatchData: An object containing packed data, labels, sequence lengths, 
                                        and trajectory IDs.
    """
    ids = [item.traj_id for item in batch]
    data = [item.data for item in batch]
    data = pack_sequence(data, enforce_sorted=False)
    targets = torch.concat([item.label.reshape(1,-1) for item in batch])
    seq_lens = torch.concat([item.seq_len.reshape(1,-1) for item in batch]).squeeze()
    
    return geolife_utils.GeoLifeBatchData(
        traj_id=ids,
        data=data,
        label=targets,
        seq_len=seq_lens
    )

def collate_fn_padded_seq(batch):
    """
    Collate function for padding sequences in a batch.

    This function processes a batch of data items by extracting trajectory IDs, data sequences,
    labels, and sequence lengths. It then pads the sequences using PyTorch's `pad_sequence` 
    function, which is useful for handling variable-length sequences in RNNs by padding them to 
    the same length.

    Args:
        batch (list): List of data items, where each item is an object containing `traj_id`, `data`, 
                      `label`, and `seq_len` attributes.

    Returns:
        geolife_utils.GeoLifeBatchData: An object containing padded data, labels, sequence lengths, 
    """
    ids = [item.traj_id for item in batch]
    data = [item.data for item in batch]
    data = pad_sequence(sequences=data, batch_first=False, padding_value=0)
    targets = torch.concat([item.label.reshape(1,-1) for item in batch])
    seq_lens = torch.concat([item.seq_len.reshape(1,-1) for item in batch]).squeeze()
    return geolife_utils.GeoLifeBatchData(
        traj_id=ids,
        data=data,
        label=targets,
        seq_len=seq_lens
    )

class GeoLifeDatamodule(BaseDatamodule):
    """
    GeoLifeDatamodule class for handling GeoLife trajectory data within the PyTorch Lightning framework.

    This class inherits from BaseDatamodule and provides functionalities to set up the GeoLife dataset,
    configure data loaders, and manage batch sizes and worker threads for training, validation, 
    and testing purposes.

    Attributes:
        dataset (GeoLifeDataset): The GeoLife dataset instance.
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
        train_test_split: Dict[str, float],
        identical_training_class_label_distribution: bool=True,
        max_trajectory_length: int=None, 
        aws_profile: str=None,
        s3_bucket_path: Union[str, Path, os.PathLike[str]]=None,
        use_threads: bool = False,
        val_batch_size: int=None,
        test_batch_size: int=None, 
        pad_sequence: bool=False,
        num_train: int=None,
        num_val: int=None,
        num_test: int=None,
        **kwargs
        ) -> None:
        """
        Args:
            batch_size (int): The batch size for training data.
            num_workers (int): The number of worker threads for data loading.
            seed (int): Random seed for reproducibility.
            training_method (str): Method for training.
            train_test_split (Dict[str, float]): Proportions for train, validation, and test splits.
            identical_training_class_label_distribution (bool, optional): Whether to maintain identical training class label distribution. Default: True.
            max_trajectory_length (int, optional): Maximum length of a trajectory. Default: None.
            aws_profile (str, optional): AWS profile for accessing S3. Default: None.
            s3_bucket_path (Union[str, Path, os.PathLike[str]], optional): S3 bucket path for loading data. Default: to None.
            use_threads (bool, optional): Whether to use threads for loading data. Default: to False.
            val_batch_size (int, optional): The batch size for validation data. Default: to None.
            test_batch_size (int, optional): The batch size for test data. Default: to None.
            pad_sequence (bool, optional): Whether to pad sequences. Default: to False.
            num_train (int, optional): Number of training samples. Default: to None.
            num_val (int, optional): Number of validation samples. Default: to None.
            num_test (int, optional): Number of test samples. Default: to None.
            **kwargs: Additional keyword arguments.
        """
        self.dataset = GeoLifeDataset(
            seed=seed,
            training_method=training_method,
            train_test_split=train_test_split,
            identical_training_class_label_distribution=identical_training_class_label_distribution,
            max_trajectory_length=max_trajectory_length,
            aws_profile=aws_profile,
            s3_bucket_path=s3_bucket_path,
            use_threads=use_threads,
        )

        if pad_sequence:
            self.collate_fn = collate_fn_padded_seq
        else:
            self.collate_fn = collate_fn_packed_seq
        
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        super().__init__(batch_size, num_workers, seed, val_batch_size=val_batch_size, test_batch_size=test_batch_size, **kwargs) 
    
    def setup(self, stage: str=None) -> None:
        if self.num_train is None:
            self.num_train = len(self.dataset.ds_config['indices']['train'])
        if self.num_val is None:
            self.num_val = len(self.dataset.ds_config['indices']['val'])
        if self.num_test is None:
            self.num_test = len(self.dataset.ds_config['indices']['test'])
        
        if isinstance(self.dataset.ds_config['indices']['train'][0], str):
            #self._train_dataset = self.dataset.train_dataset
            self._train_dataset = GeoLifeSubDataset(
                data=[
                    sample for j, sample in enumerate(self.dataset.train_dataset.data) if j <= self.num_train],
            )
            self._val_dataset = GeoLifeSubDataset(
                data=[
                    sample for j, sample in enumerate(self.dataset.val_dataset.data) if j <= self.num_val],
            )
            self._test_dataset = GeoLifeSubDataset(
                data=[
                    sample for j, sample in enumerate(self.dataset.test_dataset.data) if j <= self.num_test],
            )
            
        else:
            raise RuntimeError

        
        return super().setup(stage=stage)
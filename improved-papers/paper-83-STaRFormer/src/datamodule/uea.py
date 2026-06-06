import os
import os.path as osp
from pathlib import Path
from typing import List, Dict, Union

import torch 

from .base import BaseDatamodule
from src.dataset.uea import (
    JapaneseVowelsDataset, JapaneseVowelsBatchData,
    FaceDetectionDataset, FaceDetectionBatchData,  
    EthanolConcentrationDataset, EthanolConcentrationBatchData,
    EigenWormsDataset, EigenWormsBatchData,
    HandwritingDataset, HandwritingBatchData,
    HeartbeatDataset, HeartbeatBatchData,
    PenDigitsDataset, PenDigitsBatchData,
    PEMSSFDataset, PEMSSFBatchData,
    SCP1Dataset, SCP1BatchData,
    SCP2Dataset, SCP2BatchData,
    SpokenArabicDigitsDataset, SpokenArabicDigitsBatchData,
    UWaveGestureLibraryDataset, UWaveGestureLibraryBatchData,
)

from src.dataset.uea.base import UEADataset, UEABatchData

from src.utils import DatasetOptions


__all__ = [
    "UEADatamodule",
]


def collate_jv(batch):
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
        JapaneseVowelsBatchData: An object containing padded data, labels, sequence lengths, 
    """
    data = []
    label = []
    seq_len = []
    for sample in batch:
        data.append(sample.data.unsqueeze(0)) # add additional dimension for concatenation
        label.append(sample.label.unsqueeze(0)) # add additional dimension for concatenation
        seq_len.append(sample.seq_len.unsqueeze(0)) # add additional dimension for concatenation
    
    data = torch.concat(data, dim=0).permute(1, 0, 2) # [bs, N, D] --> [N, D, bs]
    label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    return JapaneseVowelsBatchData(data=data, label=label, seq_len=seq_len)

def collate_fd(batch):
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
        FaceDetectionBatchData: An object containing padded data, labels, sequence lengths, 
    """
    data = []
    label = []
    seq_len = []
    for sample in batch:
        data.append(sample.data.unsqueeze(0)) # add additional dimension for concatenation
        label.append(sample.label.unsqueeze(0)) # add additional dimension for concatenation
        seq_len.append(sample.seq_len.unsqueeze(0)) # add additional dimension for concatenation
    
    data = torch.concat(data, dim=0).permute(1, 0, 2) # [bs, N, D] --> [N, D, bs]
    label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    return FaceDetectionBatchData(data=data, label=label, seq_len=seq_len)

def collate_ec(batch):
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
        EthanolConcentrationBatchData: An object containing padded data, labels, sequence lengths, 
    """
    data = []
    label = []
    seq_len = []
    for sample in batch:
        data.append(sample.data.unsqueeze(0)) # add additional dimension for concatenation
        label.append(sample.label.unsqueeze(0)) # add additional dimension for concatenation
        seq_len.append(sample.seq_len.unsqueeze(0)) # add additional dimension for concatenation
    
    data = torch.concat(data, dim=0).permute(1, 0, 2) # [bs, N, D] --> [N, D, bs]
    label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    return EthanolConcentrationBatchData(data=data, label=label, seq_len=seq_len)

def collate_ew(batch):
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
        EigenWormsBatchData: An object containing padded data, labels, sequence lengths, 
    """
    data = []
    label = []
    seq_len = []
    for sample in batch:
        data.append(sample.data.unsqueeze(0)) # add additional dimension for concatenation
        label.append(sample.label.unsqueeze(0)) # add additional dimension for concatenation
        seq_len.append(sample.seq_len.unsqueeze(0)) # add additional dimension for concatenation
    
    data = torch.concat(data, dim=0).permute(1, 0, 2) # [bs, N, D] --> [N, D, bs]
    label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    return EigenWormsBatchData(data=data, label=label, seq_len=seq_len)

def collate_hb(batch):
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
        HeartbeatBatchData: An object containing padded data, labels, sequence lengths, 
    """
    data = []
    label = []
    seq_len = []
    for sample in batch:
        data.append(sample.data.unsqueeze(0)) # add additional dimension for concatenation
        label.append(sample.label.unsqueeze(0)) # add additional dimension for concatenation
        seq_len.append(sample.seq_len.unsqueeze(0)) # add additional dimension for concatenation
    
    data = torch.concat(data, dim=0).permute(1, 0, 2) # [bs, N, D] --> [N, D, bs]
    label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    return HeartbeatBatchData(data=data, label=label, seq_len=seq_len)

def collate_hw(batch):
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
        HandwritingBatchData: An object containing padded data, labels, sequence lengths, 
    """
    data = []
    label = []
    seq_len = []
    for sample in batch:
        data.append(sample.data.unsqueeze(0)) # add additional dimension for concatenation
        label.append(sample.label.unsqueeze(0)) # add additional dimension for concatenation
        seq_len.append(sample.seq_len.unsqueeze(0)) # add additional dimension for concatenation
    
    data = torch.concat(data, dim=0).permute(1, 0, 2) # [bs, N, D] --> [N, D, bs]
    label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    return HandwritingBatchData(data=data, label=label, seq_len=seq_len)

def collate_ps(batch):
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
        PEMSSFBatchData: An object containing padded data, labels, sequence lengths, 
    """
    data = []
    label = []
    seq_len = []
    for sample in batch:
        data.append(sample.data.unsqueeze(0)) # add additional dimension for concatenation
        label.append(sample.label.unsqueeze(0)) # add additional dimension for concatenation
        seq_len.append(sample.seq_len.unsqueeze(0)) # add additional dimension for concatenation
    
    data = torch.concat(data, dim=0).permute(1, 0, 2) # [bs, N, D] --> [N, D, bs]
    label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    return PEMSSFBatchData(data=data, label=label, seq_len=seq_len)

def collate_pd(batch):
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
        PenDigitsBatchData: An object containing padded data, labels, sequence lengths, 
    """
    data = []
    label = []
    seq_len = []
    for sample in batch:
        data.append(sample.data.unsqueeze(0)) # add additional dimension for concatenation
        label.append(sample.label.unsqueeze(0)) # add additional dimension for concatenation
        seq_len.append(sample.seq_len.unsqueeze(0)) # add additional dimension for concatenation
    
    data = torch.concat(data, dim=0).permute(1, 0, 2) # [bs, N, D] --> [N, D, bs]
    label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    return PenDigitsBatchData(data=data, label=label, seq_len=seq_len)

def collate_sad(batch):
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
        SpokenArabicDigitsBatchData: An object containing padded data, labels, sequence lengths, 
    """
    data = []
    label = []
    seq_len = []
    for sample in batch:
        data.append(sample.data.unsqueeze(0)) # add additional dimension for concatenation
        label.append(sample.label.unsqueeze(0)) # add additional dimension for concatenation
        seq_len.append(sample.seq_len.unsqueeze(0)) # add additional dimension for concatenation
    
    data = torch.concat(data, dim=0).permute(1, 0, 2) # [bs, N, D] --> [N, D, bs]
    label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    return SpokenArabicDigitsBatchData(data=data, label=label, seq_len=seq_len)

def collate_scp1(batch):
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
        SCP1BatchData: An object containing padded data, labels, sequence lengths, 
    """
    data = []
    label = []
    seq_len = []
    for sample in batch:
        data.append(sample.data.unsqueeze(0)) # add additional dimension for concatenation
        label.append(sample.label.unsqueeze(0)) # add additional dimension for concatenation
        seq_len.append(sample.seq_len.unsqueeze(0)) # add additional dimension for concatenation
    
    data = torch.concat(data, dim=0).permute(1, 0, 2) # [bs, N, D] --> [N, D, bs]
    label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    return SCP1BatchData(data=data, label=label, seq_len=seq_len)

def collate_scp2(batch):
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
        SCP2BatchData: An object containing padded data, labels, sequence lengths, 
    """
    data = []
    label = []
    seq_len = []
    for sample in batch:
        data.append(sample.data.unsqueeze(0)) # add additional dimension for concatenation
        label.append(sample.label.unsqueeze(0)) # add additional dimension for concatenation
        seq_len.append(sample.seq_len.unsqueeze(0)) # add additional dimension for concatenation
    
    data = torch.concat(data, dim=0).permute(1, 0, 2) # [bs, N, D] --> [N, D, bs]
    label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    return SCP2BatchData(data=data, label=label, seq_len=seq_len)

def collate_uw(batch):
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
        UWaveGestureLibraryBatchData: An object containing padded data, labels, sequence lengths, 
    """
    data = []
    label = []
    seq_len = []
    for sample in batch:
        data.append(sample.data.unsqueeze(0)) # add additional dimension for concatenation
        label.append(sample.label.unsqueeze(0)) # add additional dimension for concatenation
        seq_len.append(sample.seq_len.unsqueeze(0)) # add additional dimension for concatenation
    
    data = torch.concat(data, dim=0).permute(1, 0, 2) # [bs, N, D] --> [N, D, bs]
    label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    return UWaveGestureLibraryBatchData(data=data, label=label, seq_len=seq_len)

def collate_uea(batch):
    """
    Collate function for UEADatasets sequences in a batch.

    Args:
        batch (list): List of data items, where each item is an object containing `traj_id`, `data`, 
                      `label`, and `seq_len` attributes.

    Returns:
        UWaveGestureLibraryBatchData: An object containing padded data, labels, sequence lengths, 
    """
    data = []
    label = []
    seq_len = []
    for sample in batch:
        data.append(sample.data.unsqueeze(0)) # add additional dimension for concatenation
        label.append(sample.label.unsqueeze(0)) # add additional dimension for concatenation
        seq_len.append(sample.seq_len.unsqueeze(0)) # add additional dimension for concatenation
    
    data = torch.concat(data, dim=0).permute(1, 0, 2) # [bs, N, D] --> [N, D, bs]
    label = torch.concat(label)
    seq_len = torch.concat(seq_len)
    return UEABatchData(data=data, label=label, seq_len=seq_len)

class UEADatamodule(BaseDatamodule):
    """
    UcrUeaDatamodule class for handling various UCR/UEA datasets within the PyTorch Lightning framework.

    This class inherits from BaseDatamodule and provides functionalities to set up different UCR/UEA datasets,
    configure data loaders, and manage batch sizes and worker threads for training, validation, 
    and testing purposes.

    Attributes:
        dataset (Dataset): The dataset instance.
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
        dataset_name: str,
        batch_size: int, 
        num_workers: int, 
        seed: int,
        max_trajectory_length: int=None,
        aws_profile: str=None,
        s3_bucket_path: Union[str, Path, os.PathLike[str]]=None,
        use_threads: bool = False,
        val_batch_size: int=None,
        test_batch_size: int=None, 
        num_train: int=None,
        num_val: int=None,
        num_test: int=None,
        **kwargs
        ) -> None:
        """
        Args:
            dataset (str): The name of the dataset to be used.
            batch_size (int): The batch size for training data.
            num_workers (int): The number of worker threads for data loading.
            seed (int): Random seed for reproducibility.
            training_method (str): Method for training.
            train_splits (Dict[str, float]): Proportions for train, validation, and test splits.
            aws_profile (str, optional): AWS profile for accessing S3. Default: None.
            s3_bucket_path (Union[str, Path, os.PathLike[str]], optional): S3 bucket path for loading data. Default: None.
            use_threads (bool, optional): Whether to use threads for loading data. Default: False.
            val_batch_size (int, optional): The batch size for validation data. Default: None.
            test_batch_size (int, optional): The batch size for test data. Default: None.
            num_train (int, optional): Number of training samples. Default: None.
            num_val (int, optional): Number of validation samples. Default: None.
            num_test (int, optional): Number of test samples. Default: None.
            **kwargs: Additional keyword arguments.
        """

        dm_params = {
            'seed': seed,
            'dataset_name': dataset_name,
            'max_trajectory_length': max_trajectory_length,
            'aws_profile': aws_profile,
            's3_bucket_path': s3_bucket_path,
            'use_threads': use_threads,
        }

        #dataset_mapping = {
        #    DatasetOptions.japanesevowels: (JapaneseVowelsDataset, collate_jv),
        #    DatasetOptions.facedetection: (FaceDetectionDataset, collate_fd),
        #    DatasetOptions.ethanolconcentration: (EthanolConcentrationDataset, collate_ec),
        #    DatasetOptions.eigenworms: (EigenWormsDataset, collate_ew),
        #    DatasetOptions.heartbeat: (HeartbeatDataset, collate_hb),
        #    DatasetOptions.handwriting: (HandwritingDataset, collate_hw),
        #    DatasetOptions.pemssf: (PEMSSFDataset, collate_ps),
        #    DatasetOptions.pendigits: (PenDigitsDataset, collate_pd),
        #    DatasetOptions.selfregulationscp1: (SCP1Dataset, collate_scp1),
        #    DatasetOptions.selfregulationscp2: (SCP2Dataset, collate_scp2),
        #    DatasetOptions.spokenarabicdigits: (SpokenArabicDigitsDataset, collate_sad),
        #    DatasetOptions.uwavegesturelibrary: (UWaveGestureLibraryDataset, collate_uw),
        #}
#
        #if dataset not in dataset_mapping:
        #    raise NotImplementedError(f'{dataset} is not implemented')
        dataset_class, collate_fn = UEADataset, collate_uea
        self.dataset: UEADataset = dataset_class(**dm_params)
        self.collate_fn = collate_fn

        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        super().__init__(batch_size, num_workers, seed, val_batch_size=val_batch_size, test_batch_size=test_batch_size, **kwargs) 
        
    def setup(self, stage: str=None) -> None:
        if self.num_train is not None:
            self._train_dataset = self.dataset.train_dataset[:self.num_train]
        else:
            self._train_dataset = self.dataset.train_dataset
        
        if self.dataset.val_dataset is not None and self.num_val is not None:
            self._val_dataset = self.dataset.val_dataset[:self.num_val]
        else:
            self._val_dataset = self.dataset.val_dataset
        
        if self.dataset.test_dataset is not None and self.num_test is not None:
            self._test_dataset = self.dataset.test_dataset[:self.num_test]
        else:
            self._test_dataset = self.dataset.test_dataset
        
        return super().setup(stage=stage)

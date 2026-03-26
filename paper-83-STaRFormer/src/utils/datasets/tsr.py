import os
import logging
import requests
import patoolib

import torch
from torch import Tensor
from torch.utils.data import Dataset

from typing import List, Tuple, Dict

from .utils import BaseData
from src.utils import TrainingMethodOptions

__all__ = [
    "UcrUeaData",
    "UcrUeaBatchData"
    "UcrUeaSubDataset",
    "log_stats"
]


class TSRData(BaseData):
    def __init__(
        self,
        data: Tensor=None, 
        label: Tensor=None,
        target: Tensor=None, 
        **kwargs
        ) -> None:
        self.data = data
        self.label = label
        self.target = target
        self.seq_len = torch.tensor(data.size(0))
        super().__init__(**kwargs)


class TSRBatchData(BaseData):
    def __init__(
        self,
        data: Tensor=None, 
        label: Tensor=None, 
        target: Tensor=None, 
        seq_len: list |  Tensor=None,
        **kwargs
        ) -> None:
        self.data = data
        self.label = label
        self.target = target
        self.batch_size=len(label)
        self.seq_len = seq_len.reshape(-1, 1)
        self.ptr = [torch.tensor([0], device=data.device)]
        for idx, sl in enumerate(self.seq_len):
            self.ptr.extend([self.ptr[idx] + sl])
        self.ptr = torch.concat(self.ptr)
        super().__init__(**kwargs)

class TSRSubDataset(Dataset):
    def __init__(self, data: List[TSRData]) -> None:
        super().__init__()
        self._data = data
        self.set_data()

    def __getitem__(self, index) -> TSRData:
        return self.data[index]
    
    def __len__(self) -> int:
        return len(self.targets)
    
    def set_data(self) -> None:
        """
        Sets the sequences IDs, sequences, and targets, and labels from the dataset.

        This method processes the dataset stored in `self._data` and extracts the 
        sequence IDs, sequences, and targets. It then assigns these extracted values 
        to the corresponding attributes: `self._sequences_ids`, `self._sequences`, 
        and `self._targets`.

        The dataset (`self._data`) is expected to be a collection of objects where 
        each object has the attributes `seq_id`, `data`, and `label`.

        Attributes:
            self._sequences_ids (list): A list of sequence IDs extracted from the dataset.
            self._sequences (list): A list of sequences extracted from the dataset.
            self._targets (list): A list of targets (labels) extracted from the dataset.
        
        Raises:
            AssertionError: If the first element of `_data` is not an instance of `BaseData`.
        """
        #assert isinstance(self._data[0], BaseData), f'{type(self._data[0])} {BaseData}'
        #self._sequences_ids = [d.seq_id if hasattr(d, 'seq_id') else d.traj_id for d in self._data]
        self._sequences = [d.data for d in self._data]
        self._targets = [d.target for d in self._data]
        self._labels = [d.label for d in self._data]
    
    @property
    def seq_ids(self):
        return self._sequences_ids
    
    @property
    def data(self):
        return self._data
    
    @property
    def targets(self):
        return self._targets
    
    @property
    def labels(self,):
        return self._labels
    

############
# download #
############

def _get_progress_log(part, total, progress_bar_length: int=50):
    """ If the total is unknown, just return the part """
    if total == -1:
        return f"Downloaded: {part / 1024 ** 2:.2f} MB"

    passed = "=" * int(progress_bar_length * part / total)
    rest = " " * (progress_bar_length - len(passed))
    p_bar = f"[{passed}{rest}] {part * 100/total:.2f}%"
    if part == total:
        p_bar += "\n"
    return p_bar

def start_download(url, dataset_name: str='Geolife'):
    """ """
    response = requests.get(url, allow_redirects=True, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download dataset {dataset_name}")

    return response

from pathlib import Path


def download_until_finish(url: str, response: requests.Response, dataset_path: Path, chunk_size: int=4096) -> Path:
    """ """
    data_length = int(response.headers.get("content-length", -1))
    size_mb_msg = (
        f"    Size: {data_length / 1024 ** 2:.2f} MB" if data_length != -1 else ""
    )
    dataset_file_path = dataset_path
    with open(dataset_file_path, "wb") as ds_file:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                ds_file.write(chunk)
                downloaded += len(chunk)
                print(
                    _get_progress_log(downloaded, data_length) + size_mb_msg,
                    end="\r",
                    flush=True,
                )
    return dataset_file_path

def _download(url: str, dataset_name: str, dataset_file_path: Path) -> Path:
    """ Check if the dataset is already downloaded
    """
    if os.path.exists(dataset_file_path):
        return dataset_file_path
    
    # Make the download request
    response = start_download(url, dataset_name)

    # Download the dataset to a zip file
    return download_until_finish(url, response, dataset_file_path)

def download(
    url: str, 
    dataset_name: str, 
    dataset_file_path: Path | str,
    dataset_path: Path,
    uncompress: bool=True
    ):
    _download(url, dataset_name, dataset_file_path)

    if uncompress:
        patoolib.extract_archive(
            str(dataset_file_path),
            outdir=str(dataset_path),
            verbosity=1,
            interactive=False,
        )
    
    return dataset_file_path


###########
# logging #
###########

def log_stats(
    method: str,
    cli_logger: logging.Logger,
    train_dataset: TSRSubDataset=None,
    val_dataset: TSRSubDataset=None,
    test_dataset: TSRSubDataset=None,
    ds_config: Dict=None
    ):
    assert train_dataset is not None, f'{train_dataset} cannot be None!'
    #assert test_dataset is not None, f'{test_dataset} cannot be None!'

    if method == TrainingMethodOptions.centralized:
        log_stats_centralized(cli_logger=cli_logger, train_dataset=train_dataset, 
            val_dataset=val_dataset, test_dataset=test_dataset)
    #elif method == TrainingMethodOptions.federated:
        #log_stats_federated()
    else:
        RuntimeError

def log_stats_centralized(
    cli_logger: logging.Logger, 
    train_dataset: TSRSubDataset=None,
    val_dataset: TSRSubDataset=None,
    test_dataset: TSRSubDataset=None,
    ):
    cli_logger.info("-" * 50)
    cli_logger.info("Statistics")
    cli_logger.info("-" * 50)

    cli_logger.info(f"Size of Train Data:\t{len(train_dataset)}")
    if val_dataset is not None: 
        cli_logger.info(f"Size of Val Data:\t {len(val_dataset)}")
    if test_dataset is not None: 
        cli_logger.info(f"Size of Test Data:\t{len(test_dataset)}")
    cli_logger.info("-" * 50)
    if test_dataset is not None and val_dataset is not None:
        cli_logger.info(f"Total Data Size:\t{len(train_dataset)+len(val_dataset)+len(test_dataset)}")
    elif val_dataset is None and test_dataset is not None:
        cli_logger.info(f"Total Data Size:\t{len(train_dataset)+len(test_dataset)}")
    elif test_dataset is None and val_dataset is not None:
        cli_logger.info(f"Total Data Size:\t{len(train_dataset)+len(val_dataset)}")
    else: # val_dataset is None and test_dataset is None
        cli_logger.info(f"Total Data Size:\t{len(train_dataset)}")
    cli_logger.info("-" * 50)
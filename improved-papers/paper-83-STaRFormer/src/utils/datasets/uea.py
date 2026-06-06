import logging


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


class UcrUeaData(BaseData):
    def __init__(
        self,
        data: Tensor=None, 
        label: Tensor=None, 
        **kwargs
        ) -> None:
        self.data = data
        self.label = label
        self.seq_len = torch.tensor(data.size(0))
        super().__init__(**kwargs)


class UcrUeaBatchData(BaseData):
    def __init__(
        self,
        data: Tensor=None, 
        label: Tensor=None, 
        seq_len: list |  Tensor=None,
        **kwargs
        ) -> None:
        self.data = data
        self.label = label
        self.batch_size=len(label)
        self.seq_len = seq_len.reshape(-1, 1)
        self.ptr = [torch.tensor([0])]
        for idx, sl in enumerate(self.seq_len):
            self.ptr.extend([self.ptr[idx] + sl])
        self.ptr = torch.concat(self.ptr)
        super().__init__(**kwargs)

class UcrUeaSubDataset(Dataset):
    def __init__(self, data: List[UcrUeaData]) -> None:
        super().__init__()
        self._data = data
        self.set_data()

    def __getitem__(self, index) -> UcrUeaData:
        return self.data[index]
    
    def __len__(self) -> int:
        return len(self.targets)
    
    def set_data(self) -> None:
        self._trajectories = [d.data for d in self._data]
        self._targets = [d.label for d in self._data]
    
    @property
    def data(self):
        return self._data
    
    @property
    def targets(self):
        return self._targets


###########
# logging #
###########

def log_stats(
    method: str,
    cli_logger: logging.Logger,
    train_dataset: UcrUeaSubDataset=None,
    val_dataset: UcrUeaSubDataset=None,
    test_dataset: UcrUeaSubDataset=None,
    ds_config: Dict=None
    ):
    assert train_dataset is not None, f'{train_dataset} cannot be None!'
    assert val_dataset is not None, f'{val_dataset} cannot be None!'

    if method == TrainingMethodOptions.centralized:
        log_stats_centralized(cli_logger=cli_logger, train_dataset=train_dataset, 
            val_dataset=val_dataset, test_dataset=test_dataset)
    #elif method == TrainingMethodOptions.federated:
        #log_stats_federated()
    else:
        RuntimeError

def log_stats_centralized(
    cli_logger: logging.Logger, 
    train_dataset: UcrUeaSubDataset=None,
    val_dataset: UcrUeaSubDataset=None,
    test_dataset: UcrUeaSubDataset=None,
    ):
    cli_logger.info("-" * 50)
    cli_logger.info("Statistics")
    cli_logger.info("-" * 50)

    cli_logger.info(f"Size of Train Data:\t{len(train_dataset)}")
    if val_dataset is not None: 
        cli_logger.info(f"Size of Val Data:\t {len(val_dataset)}")
    if test_dataset is not None: cli_logger.info(f"Size of Test Data:\t{len(test_dataset)}")
    cli_logger.info("-" * 50)
    if val_dataset is not None and test_dataset is None: 
        cli_logger.info(f"Total Data Size:\t{len(train_dataset)+len(val_dataset)}")
    elif val_dataset is None and test_dataset is not None: 
        cli_logger.info(f"Total Data Size:\t{len(train_dataset)+len(val_dataset)}")
    elif val_dataset is None and test_dataset is not None: 
        cli_logger.info(f"Total Data Size:\t{len(train_dataset)}")
    else:
        cli_logger.info(f"Total Data Size:\t{len(train_dataset)+len(val_dataset)+len(test_dataset)}")
    cli_logger.info("-" * 50)
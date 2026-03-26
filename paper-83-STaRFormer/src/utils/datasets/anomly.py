import logging
import torch
import numpy as np

from torch.utils.data import Dataset
from torch import Tensor
from typing import List, Dict

from src.utils import BaseData, TrainingMethodOptions


__all__ = [
    "KPIData",
    "KPIBatchData"
    "KPISubDataset",
    "YahooData",
    "YahooBatchData"
    "YahooSubDataset",
    "log_stats",
    "sliding_window_slice"
]


class KPIData(BaseData):
    def __init__(
        self,
        seq_id: str=None,
        data: Tensor=None, 
        label: Tensor=None, 
        timestamps: Tensor=None,
        **kwargs
        ) -> None:
        self.seq_id = seq_id
        self.data = data
        self.label = label
        self.timestamps = timestamps
        self.seq_len = torch.tensor(data.size(0))
        super().__init__(**kwargs)


class KPIBatchData(BaseData):
    def __init__(
        self,
        seq_id: str=None,
        data: Tensor=None, 
        label: Tensor=None, 
        seq_len: list | Tensor=None,
        timestamps: Tensor=None,
        **kwargs
        ) -> None:
        self.seq_id = seq_id
        self.data = data
        self.label = label
        self.timestamps = timestamps
        self.batch_size=len(seq_len)
        self.seq_len = seq_len.reshape(-1, 1)
        self.ptr = [torch.tensor([0])]
        for idx, sl in enumerate(self.seq_len):
            self.ptr.extend([self.ptr[idx] + sl])
        self.ptr = torch.concat(self.ptr)
        super().__init__(**kwargs)


class KPISubDataset(Dataset):
    def __init__(self, data: List[KPIData]) -> None:
        super().__init__()
        self._data = data
        self.set_data()

    def __getitem__(self, index) -> KPIData:
        return self.data[index]
    
    def __len__(self) -> int:
        return len(self.targets)
    
    def set_data(self) -> None:
        self._seq_ids = [d.seq_id for d in self._data]
        self._sequences = [d.data for d in self._data]
        self._targets = [d.label for d in self._data]
        self._timestamps = [d.timestamps for d in self._data]
    
    @property
    def data(self):
        return self._data
    
    @property
    def targets(self):
        return self._targets

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def seq_ids(self):
        return self._seq_ids


class YahooData(KPIData):
    def __init__(self, seq_id = None, data = None, label = None, timestamps = None, **kwargs):
        super().__init__(seq_id, data, label, timestamps, **kwargs)
    #def __init__(self, seq_id = None, data = None, label = None, timestamps = None, **kwargs):
    #    super().__init__(seq_id, data, label, timestamps, **kwargs)

class YahooBatchData(KPIBatchData):
    def __init__(self, 
                 seq_id: str=None,
                 data: Tensor=None, 
                 label: Tensor=None, 
                 seq_len: list | Tensor=None,
                 timestamps: Tensor=None,
                 **kwargs):
        super().__init__(seq_id, data, label, seq_len, timestamps, **kwargs)

class YahooSubDataset(KPISubDataset):
    def __init__(self, data):
        super().__init__(data)

###########
# logging #
###########

def log_stats(
    method: str,
    cli_logger: logging.Logger,
    train_dataset: KPISubDataset=None,
    val_dataset: KPISubDataset=None,
    test_dataset: KPISubDataset=None,
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
    train_dataset: KPISubDataset=None,
    val_dataset: KPISubDataset=None,
    test_dataset: KPISubDataset=None,
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


def sliding_window_slice(data, labels, timestamps, window_size: int=1024, stride: int=512):
    """
    
    """
    num_windows = (len(data) - window_size) // stride + 1
    data_windows = [
        data[window*stride : window*stride+window_size]
        for window in range(num_windows)
    ]
    label_windows = [
        labels[window*stride : window*stride+window_size]
        for window in range(num_windows)
    ]
    ts_windows = [
        timestamps[window*stride : window*stride+window_size]
        for window in range(num_windows)
    ]
    remainder_start = num_windows * stride
    
    if remainder_start < len(data):
        partial_data_window = data[remainder_start:]
        partial_label_window = labels[remainder_start:]
        partial_ts_window = timestamps[remainder_start:]

        if np.any(partial_label_window == 1):
            data_windows.append(partial_data_window)
            label_windows.append(partial_label_window)
            ts_windows.append(partial_ts_window)
    
    for i, (dw, lw, tsw) in enumerate(zip(data_windows, label_windows, ts_windows)):
        assert dw.shape[0] <= window_size, f'{dw.shape[0]} | {i+1} / {len(data_windows)} {remainder_start} {len(data)}, {len(data) - remainder_start}'
        assert lw.shape[0] <= window_size, f'{lw.shape[0]} | {i+1} / {len(label_windows)} {remainder_start} {len(data)}, {len(data) - remainder_start}'
        assert tsw.shape[0] <= window_size, f'{tsw.shape[0]} | {i+1} / {len(ts_windows)} {remainder_start} {len(data)}, {len(data) - remainder_start}'
    return data_windows, label_windows, ts_windows
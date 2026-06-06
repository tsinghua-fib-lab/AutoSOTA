from typing import Any, Dict

import logging
import torch
import torch
from torch.utils.data import DataLoader

from lightning import LightningDataModule

from src.utils import seed_worker, DatasetOptions


class BaseDatamodule(LightningDataModule):
    """
    BaseDatamodule class for handling data loading and processing in a PyTorch Lightning framework.

    This class inherits from LightningDataModule and provides functionalities to set up datasets,
    configure data loaders, and manage batch sizes and worker threads for training, validation, 
    and testing purposes.

    Attributes:
        batch_size (int): The batch size for training data.
        val_batch_size (int): The batch size for validation data.
        test_batch_size (int): The batch size for test data.
        num_workers (int): The number of worker threads for data loading.
        seed (int): Random seed for reproducibility.
        collate_fn (callable, optional): Function to merge a list of samples to form a mini-batch.
        _train_dataset (Dataset, optional): The training dataset.
        _val_dataset (Dataset, optional): The validation dataset.
        _test_dataset (Dataset, optional): The test dataset.

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
    def __init__(self, 
        batch_size: int,
        num_workers: int,
        seed: int,
        val_batch_size: int=None,
        test_batch_size: int=None,
        **kwargs
        ) -> None:
        """
        Args:
            batch_size (int): The batch size for training data.
            num_workers (int): The number of worker threads for data loading.
            seed (int): Random seed for reproducibility.
            val_batch_size (int, optional): The batch size for validation data. Default: None, in which case `batch_size` is used.
            test_batch_size (int, optional): The batch size for test data. Default: None, in which case `batch_size` is used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.test_batch_size = batch_size if test_batch_size is None else test_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.configure_cli_logger()

        if not hasattr(self, 'collate_fn'):
            self.cli_logger.info("No collate_fn attribute found, setting to None!")
            self.collate_fn = None
        
        self._train_dataset, self._val_dataset, self._test_dataset = None, None, None
    
    def setup(self, stage: str=None) -> None:
        assert self._train_dataset != None #and self._test_dataset != None

    def train_dataloader(self) -> Any:
        return self.configure_dataloader(batch_size=self.batch_size, num_workers=self.num_workers, seed=self.seed, mode='train')

    def val_dataloader(self) -> Any:
        return self.configure_dataloader(batch_size=self.val_batch_size, num_workers=self.num_workers, seed=self.seed, mode='val')
    
    def test_dataloader(self) -> Any:
        return self.configure_dataloader(batch_size=self.test_batch_size, num_workers=self.num_workers, seed=self.seed, mode='test')        
        #return self.configure_dataloader(batch_size=1, num_workers=self.num_workers, seed=self.seed, mode='test')

    def configure_dataloader(self, batch_size: int, num_workers: int, seed: int, mode: str='train'):
        g = torch.Generator()
        g.manual_seed(seed)

        if mode == 'train':
            return DataLoader(
                self._train_dataset, batch_size=batch_size, shuffle=True, 
                num_workers=num_workers, worker_init_fn=seed_worker, generator=g,
                collate_fn=self.collate_fn, pin_memory=True
            )
        elif mode == 'val':
            if self._val_dataset is not None:
                return DataLoader(
                    self._val_dataset, batch_size=batch_size, shuffle=False, 
                    num_workers=num_workers, worker_init_fn=seed_worker, generator=g,
                    collate_fn=self.collate_fn, pin_memory=True
                )
            return None
            
        elif mode == 'test':
            if self._test_dataset is not None:
                return DataLoader(
                    self._test_dataset, batch_size=batch_size, shuffle=False, 
                    num_workers=num_workers, worker_init_fn=seed_worker, generator=g,
                    collate_fn=self.collate_fn, pin_memory=True
                )
            return None
        else:
            raise ValueError
    
    def configure_cli_logger(self, log_level=logging.INFO):
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(log_level)
        self.cli_logger = logger

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def val_dataset(self):
        return self._val_dataset
    
    @property
    def test_dataset(self):
        return self._test_dataset

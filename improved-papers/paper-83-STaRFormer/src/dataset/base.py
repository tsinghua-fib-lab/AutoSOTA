import os
import os.path as osp
import logging
import pickle
import ujson
from typing import List, Any, Dict, Tuple, Literal
from tqdm.auto import tqdm
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.utils.data import load_asset, store_asset
from src.utils import BaseData

class BaseDataset(Dataset, ABC):
    """
    Base class for datasets.

    This class provides a base implementation for datasets, including methods 
    for processing, storing, and loading dataset assets, as well as checking 
    for the existence of dataset configurations. It inherits from 
    both `Dataset` and `ABC` (Abstract Base Class).
    
    Attributes:
        data_dir (str): The root directory for storing dataset files.
        __config_filename (str): The filename for the dataset configuration file.
        

    Methods:
        __init__(dataset_name: str, seed: int=42, **kwargs): Initializes the dataset 
            with the given name and seed, and processes the dataset.
        check_if_centralized_dataset_exists_on_s3(s3_client, s3_buckets: List, **kwargs): 
            Checks if a centralized dataset configuration exists on S3.
        check_if_centralized_dataset_exists(**kwargs): Checks if a centralized dataset 
            configuration exists locally.
        _check_if_centralized_dataset_exists(**kwargs): Abstract method to check if a 
            centralized dataset configuration exists.
        configure_cli_logger(log_level=logging.INFO): Configures the command-line interface (CLI) logger.
        store_asset(format: str='pickle', file_path: str=None, obj: object=None, write: str='w'): 
            Stores an object to a file in the specified format.
        load_asset(format: str='pickle', file_path: str=None, buffer: bool=False, read: str='r'): 
            Loads an object from a file in the specified format.
        data: Property to get the dataset data.
        targets: Property to get the dataset targets.
        ds_config: Property to get the dataset configuration.
    """
    data_dir = osp.join("/".join(osp.abspath(__file__).split("/")[:-3]), "data")
    __config_filename = 'config.json'
    def __init__(self, dataset_name: str, seed: int=42, **kwargs) -> None:
        """
        Initializes the dataset with the given name and seed, and processes the dataset.

        Args:
            dataset_name (str): The name of the dataset.
            seed (int): The seed for random number generation. Default is 42.
            **kwargs: Additional keyword arguments for dataset processing.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.seed = seed
        self._data = None
        self._targets = None
        self._ds_config = None
        self.configure_cli_logger()
        self._process(**kwargs)
    
    @abstractmethod
    def __getitem__(self, idx):
        return NotImplementedError

    @abstractmethod
    def __len__(self):
        return len(self.targets)

    @abstractmethod
    def download(self, **kwargs) -> None | Any:
        """ 
        Abstract method to download the dataset.

        This method should be implemented by any subclass to handle the downloading
        of the dataset. The implementation can vary depending on the source and 
        format of the dataset. This method may accept various keyword arguments 
        (**kwargs) to customize the download process, such as specifying the 
        download URL, file path, authentication details, etc.

        Returns:
            None: This method is expected to perform the download operation and 
            does not return any value. It should raise an appropriate exception 
            if the download fails 
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def process(self, **kwargs) -> None | Any:
        """
        Abstract method to process the dataset.

        This method should be implemented by any subclass to handle the processing
        of the dataset. The implementation can vary depending on the nature and 
        requirements of the dataset. This method may accept various keyword 
        arguments (**kwargs) to customize the processing steps, such as data 
        cleaning, transformation, normalization, feature extraction, etc.

        Returns:
            None: This method is expected to perform the processing operation and 
            does not return any value. It should raise an appropriate exception 
            if the processing fails.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _process(self, **kwargs) -> None:
        """
        Internal method to set up dataset directories and initiate the download and processing.

        This method creates the necessary directory structure for the dataset if it 
        does not already exist. It then calls the `download` and `process` methods 
        to handle the downloading and processing of the dataset, respectively.

        Args:
            **kwargs: Additional keyword arguments to be passed to the `download` 
            and `process` methods for customizing the download and processing steps.

        Directory Structure:
            - root: The root directory for the dataset.
            - raw: Directory for storing raw dataset files.
            - processed: Directory for storing processed dataset files.
            - centralized: Directory for storing centralized processed data.
            - federated: Directory for storing federated processed data.

        Returns:
            None: This method performs the setup, download, and processing operations 
            and does not return any value. It should raise an appropriate exception 
            if any step fails.
        """
        dataset_dir = osp.join(self.data_dir, self.dataset_name)
        # check if dir exists
        if not osp.exists(dataset_dir):
            self.cli_logger.info("\nCreating dataset directory.\n")
            os.makedirs(dataset_dir)
        
        # setup directories
        raw = osp.join(dataset_dir, "raw")
        processed = osp.join(dataset_dir, "processed")
        centralized = osp.join(processed, "centralized")
        federated = osp.join(processed, "federated")

        self.dataset_dir_paths = {
            "root": dataset_dir,
            "raw": raw,
            "processed": processed,
            "centralized": centralized,
            "federated": federated,
        }

        self.download(**kwargs)
        self.process(**kwargs)
    
    def store_asset(self, format: Literal['pickle', 'pt', 'json'] = 'pickle', file_path: str = None, obj: object = None, write: Literal['w', 'wb'] = 'w'):
        return store_asset(format=format, file_path=file_path, obj=obj, write=write)

    def load_asset(self, format: Literal['pickle', 'pt', 'json'] = 'pickle', file_path: str=None, buffer: bool=False, read: Literal['r', 'rb']='r'):
        return load_asset(format=format, file_path=file_path, buffer=buffer, read=read)
    
    def check_if_centralized_dataset_exists_on_s3(
        self, 
        s3_client,
        s3_buckets: List,
        **kwargs) -> Tuple[bool, Dict[str, Any]]:
        """
        Checks if a centralized dataset configuration exists on S3.

        This method iterates through the provided S3 buckets to check if the 
        centralized dataset configuration file exists. If found, it loads the 
        configuration and updates the status.

        Args:
            s3_client: The S3 client to interact with the S3 service.
            s3_buckets (List): A list of S3 bucket keys to check for the configuration file.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing:
                - load_from_config (bool): Indicates if the configuration was loaded.
                - ds_config (dict): The loaded dataset configuration.
        """
        load_from_config = False
        ds_config = None

        # load configs from s3
        for key_summary in tqdm(s3_buckets, desc="Loading Configs from S3 Bucket"):
            if key_summary.key.split('/')[-1] == self.__config_filename: # check that config is loaded
                s3_object = s3_client.get_object(Bucket=self.s3_bucket_path, Key=key_summary.key)
                binary = s3_object['Body'].read()
                tmp_config = ujson.loads(binary.decode('utf-8'))
                load_from_config, ds_config = self._check_if_centralized_dataset_exists(
                    load_from_config, ds_config, tmp_config,
                )
        
        return load_from_config, ds_config

    def check_if_centralized_dataset_exists(self, **kwargs) -> Tuple[bool, Dict[str, Any]]:
        """
        Checks if a centralized dataset configuration exists locally.

        This method walks through the directory structure of the centralized dataset 
        to check if the configuration file exists. If found, it loads the configuration 
        and updates the status.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing:
                - load_from_config (bool): Indicates if the configuration was loaded.
                - ds_config (dict): The loaded dataset configuration.
        """
        load_from_config = False
        ds_config = None
        
        for dir, folders, file in os.walk(self.dataset_dir_paths['centralized']):
            for folder in folders:
                if osp.exists(osp.join(dir, folder, self.__config_filename)):
                    with open(osp.join(dir, folder, self.__config_filename), 'r') as f:
                        tmp_config = ujson.load(f)
                        
                        load_from_config, ds_config = self._check_if_centralized_dataset_exists(
                            load_from_config, ds_config, tmp_config,
                        )
                    

        return load_from_config, ds_config
    
    @abstractmethod
    def _check_if_centralized_dataset_exists(self, **kwargs):
        """
        Abstract method to check if a centralized dataset configuration exists.

        This method should be implemented by any subclass to handle the specific 
        logic for checking if a centralized dataset configuration exists.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def set_data(self) -> None:
        """
        Sets the sequences IDs, sequences, and targets from the dataset.

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
        if not isinstance(self._data[0], BaseData):
            self.cli_logger.warning('Potentially Wrong DataClass given!')
    
        self._sequences_ids = [d.seq_id if hasattr(d, 'seq_id') else d.traj_id for d in self._data]
        self._sequences = [d.data for d in self._data]
        self._targets = [d.label for d in self._data]

    def configure_cli_logger(self, log_level=logging.INFO):
        """
        Configures the command-line interface (CLI) logger.

        This method sets up the CLI logger with the specified log level.

        Args:
            log_level: The logging level to set for the logger. Default is logging.INFO.

        Returns:
            None: This method performs the logger configuration and does not return any value.
        """
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(log_level)
        self.cli_logger = logger

    @property
    def data(self,) -> List[BaseData]:
        """
        This property provides access to the dataset data.

        Returns:
            The dataset data.
        """
        return self._data

    @property
    def targets(self,) -> Tensor:
        """
        This property provides access to the dataset targets.

        Returns:
            The dataset targets.
        """
        return self._targets

    @property
    def sequences(self):
        """
        This property provides access to the dataset sequences.

        Returns:
            The dataset sequences.
        """
        return self._sequences

    @property
    def ds_config(self,) -> Dict[str, Any]:
        """
        Returns the dataset configuration.

        This property provides access to the dataset configuration.

        Returns:
            The dataset configuration.
        """
        return self._ds_config

    
class BaseDataSubset(Dataset, ABC):
    """
    A dataset subclass for handling data and targets.

    This class provides an implementation for a dataset that holds data and 
    corresponding targets. It also supports optional transformations for 
    both data and targets.

    Attributes:
        data (Tensor): The data tensor.
        targets (Tensor): The targets tensor.
        transform (callable, optional): A function/transform to apply to the data.
        target_transform (callable, optional): A function/transform to apply to the targets.

    Methods:
        __init__(data: Tensor, targets: Tensor, transform=None, target_transform=None) -> None:
            Initializes the dataset with data, targets, and optional transformations.
        __len__() -> int:
            Returns the number of samples in the dataset.
        __getitem__(index) -> Any:
            Retrieves the data and target at the specified index.
    """
    def __init__(self, data: List[BaseData]) -> None:
        """
        Initializes the dataset with data, targets, and optional transformations.

        Args:
            data (Tensor): The data tensor.
            targets (Tensor): The targets tensor.
            transform (callable, optional): A function/transform to apply to the data.
            target_transform (callable, optional): A function/transform to apply to the targets.
        """
        super().__init__()
        self._data = data
        self.set_data()

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.targets)
    
    def __getitem__(self, index) -> BaseData:
        """
        Retrieves the data and target at the specified index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            Any: The data and target at the specified index.
        
        Raises:
            NotImplementedError: This method should be implemented to return the 
                                 actual data and target at the specified index.
        """
        return self._data[index]
    
    
    def set_data(self) -> None:
        """
        Sets the sequences IDs, sequences, and targets from the dataset.

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
        self._sequences_ids = [d.seq_id if hasattr(d, 'seq_id') else d.traj_id for d in self._data]
        self._sequences = [d.data for d in self._data]
        self._targets = [d.label for d in self._data]

    @property
    def data(self,) -> List[BaseData]:
        """
        This property provides access to the dataset data.

        Returns:
            The dataset data.
        """
        return self._data
    
    @property
    def sequence_ids(self,) -> List[BaseData]:
        """
        This property provides access to the dataset data.

        Returns:
            The dataset data.
        """
        return self._sequences_ids

    @property
    def sequences(self):
        """
        This property provides access to the dataset sequences.

        Returns:
            The dataset sequences.
        """
        return self._sequences

    @property
    def targets(self,) -> Tensor:
        """
        This property provides access to the dataset targets.

        Returns:
            The dataset targets.
        """
        return self._targets

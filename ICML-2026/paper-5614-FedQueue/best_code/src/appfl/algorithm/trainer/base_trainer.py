import abc
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, Tuple, Union, OrderedDict


class BaseTrainer:
    """
    BaseTrainer:
        Abstract base trainer for FL clients.
    Args:
        model: torch neural network model to train
        loss_fn: loss function for the model training
        metric: metric function for the model evaluation
        train_dataset: training dataset
        val_dataset: validation dataset
        train_configs: training configurations
        logger: logger for the trainer
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
        metric: Optional[Any] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
        client_id: Optional[Any] = None,
        **kwargs,
    ):
        self.round = 0
        self.model = model
        self.loss_fn = loss_fn
        self.metric = metric
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_configs = train_configs
        self.logger = logger
        self.client_id = client_id
        self.__dict__.update(kwargs)

    @abc.abstractmethod
    def get_parameters(
        self,
    ) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """Return local model parameters and optional metadata."""
        pass

    @abc.abstractmethod
    def train(self, **kwargs):
        pass

    def load_parameters(
        self,
        params: Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict], Any],
    ):
        """Load model parameters."""
        self.model.load_state_dict(params, strict=False)

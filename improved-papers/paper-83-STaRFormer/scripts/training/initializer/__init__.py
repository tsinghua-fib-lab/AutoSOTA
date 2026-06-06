from omegaconf import DictConfig
from lightning.pytorch.utilities import rank_zero_only

from .dataset import initialize_datamodule
from .model import initialize_lightning_model
from .utils import initialize_exp_logger

class Initializer(object):
    def __init__(self, config: DictConfig) -> None:
        self.config = config
    
    def init_datamodule(self):
        datamodule, instance, config = initialize_datamodule(self.config)
        return datamodule, instance, config
    
    def init_model(self, max_seq_len: int=None, **kwargs):
        """ Initialize model and lightning model """
        model = initialize_lightning_model(config=self.config, max_seq_len=max_seq_len, **kwargs)
        return model

#    @rank_zero_only    
    def init_exp_logger(self, **kwargs):
        return initialize_exp_logger(config=self.config, **kwargs)
        

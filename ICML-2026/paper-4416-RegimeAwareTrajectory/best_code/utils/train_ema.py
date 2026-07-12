from ema_pytorch import EMA
from pytorch_lightning.callbacks import Callback
import torch

class EMACallback(Callback):
    
    def __init__(self, decay=0.999, update_after_step: int = 0, update_every: int = 1):
        super().__init__()
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.ema = None
        self._original_state_dict = None
    
    def on_fit_start(self, trainer, pl_module):
        self.ema = EMA(
            pl_module, 
            beta=self.decay, 
            update_after_step=self.update_after_step, 
            update_every=self.update_every
        )
        self.ema.to(pl_module.device)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.ema.update()
    
    def on_validation_start(self, trainer, pl_module):
        # save original state dict
        self._original_state_dict = {
            k: v.clone() for k, v in pl_module.state_dict().items()
        }
        # switch to EMA weights
        self.ema.copy_params_from_ema_to_model()
        
    def on_validation_end(self, trainer, pl_module):
        # switch back to original weights
        if self._original_state_dict is not None:
            pl_module.load_state_dict(self._original_state_dict)
            self._original_state_dict = None
    
    def on_test_start(self, trainer, pl_module):
        # Initialize EMA if not yet done (test-only mode)
        if self.ema is None:
            self.ema = EMA(
                pl_module, 
                beta=self.decay, 
                update_after_step=self.update_after_step, 
                update_every=self.update_every
            )
            self.ema.to(pl_module.device)
        # switch to EMA weights
        self._original_state_dict = {
            k: v.clone() for k, v in pl_module.state_dict().items()
        }
        self.ema.copy_params_from_ema_to_model()
    
    def on_test_end(self, trainer, pl_module):
        # switch back to original weights
        if self._original_state_dict is not None:
            pl_module.load_state_dict(self._original_state_dict)
            self._original_state_dict = None
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # save EMA weights
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.ema_model.state_dict()
    
    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        # load EMA weights
        if 'ema_state_dict' in checkpoint:
            if self.ema is None:
                # Test-only mode: initialize EMA first
                self.ema = EMA(
                    pl_module, 
                    beta=self.decay, 
                    update_after_step=self.update_after_step, 
                    update_every=self.update_every
                )
                self.ema.to(pl_module.device)
            self.ema.ema_model.load_state_dict(checkpoint['ema_state_dict'])

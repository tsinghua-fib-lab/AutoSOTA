import torch
from torch.optim.lr_scheduler import _LRScheduler


class ManualScheduler(_LRScheduler):
    '''
    Example:
    config = {
        # epoch: learning rate
        0: 1e-3,
        10: 5e-4,
        20: 1e-4
    }
    '''
    def __init__(self, optimizer, config, verbose=False, **kwargs):
        self.config = config
        self.verbose = verbose
        super().__init__(optimizer, **kwargs)

    def get_lr(self):
        if not self.last_epoch in self.config.keys():
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            new_lr = [
                self.config[self.last_epoch] for group in self.optimizer.param_groups]
            if self.verbose:
                print(f'learning rate -> {new_lr}')
            return new_lr

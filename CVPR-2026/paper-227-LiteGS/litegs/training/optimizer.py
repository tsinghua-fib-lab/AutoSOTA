import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

from .. import arguments
from ..utils.wrapper import sparse_adam_update

class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps):
        super().__init__(params=params, lr=lr, eps=eps)
    
    @torch.no_grad()
    def step(self, visible_chunk):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)


            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"].view(-1,param.shape[-2],param.shape[-1])
            exp_avg_sq = stored_state["exp_avg_sq"].view(-1,param.shape[-2],param.shape[-1])
            param_view=param.data.view(-1,param.shape[-2],param.shape[-1])
            sparse_adam_update(param_view, param.grad._values().reshape(param_view.shape[0],visible_chunk.shape[0],param_view.shape[-1]), exp_avg, exp_avg_sq, visible_chunk, lr, 0.9, 0.999, eps)

class Scheduler(_LRScheduler):
    def __init__(self, optimizer:torch.optim.Adam,lr_init, lr_final,max_epochs=10000, last_epoch=-1):
        self.max_epochs=max_epochs
        self.lr_init=lr_init
        self.lr_final=lr_final
        super(Scheduler, self).__init__(optimizer, last_epoch)
        return
    
    def __helper(self):
        if self.last_epoch < 0 or (self.lr_init == 0.0 and self.lr_final == 0.0):
            # Disable this parameter
            return 0.0
        delay_rate = 1.0
        t = np.clip(self.last_epoch / self.max_epochs, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return delay_rate * log_lerp

    def get_lr(self):
        lr_list=[]
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                lr_list.append(self.__helper())
            else:
                lr_list.append(group['initial_lr'])

        return lr_list


def get_optimizer(xyz:torch.nn.Parameter,scale:torch.nn.Parameter,rot:torch.nn.Parameter,
                  sh_0:torch.nn.Parameter,sh_rest:torch.nn.Parameter,opacity:torch.nn.Parameter,
                  spatial_lr_scale:float,
                  opt_setting:arguments.OptimizationParams,pipeline_setting:arguments.PipelineParams):
    
    l = [
        {'params': [xyz], 'lr': opt_setting.position_lr_init * spatial_lr_scale, "name": "xyz"},
        {'params': [sh_0], 'lr': opt_setting.feature_lr, "name": "sh_0"},
        {'params': [sh_rest], 'lr': opt_setting.feature_lr / 20.0, "name": "sh_rest"},
        {'params': [opacity], 'lr': opt_setting.opacity_lr, "name": "opacity"},
        {'params': [scale], 'lr': opt_setting.scaling_lr, "name": "scale"},
        {'params': [rot], 'lr': opt_setting.rotation_lr, "name": "rot"}
    ]
    if pipeline_setting.cluster_size and pipeline_setting.sparse_grad:
        optimizer = SparseGaussianAdam(l, lr=0, eps=1e-15)
    else:
        optimizer = torch.optim.Adam(l, lr=0, eps=1e-15)
    scheduler = Scheduler(optimizer,opt_setting.position_lr_init*spatial_lr_scale,
              opt_setting.position_lr_final*spatial_lr_scale,
              max_epochs=opt_setting.position_lr_max_steps)
    
    return optimizer,scheduler
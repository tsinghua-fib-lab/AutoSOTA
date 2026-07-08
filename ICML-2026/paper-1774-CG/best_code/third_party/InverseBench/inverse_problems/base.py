from abc import ABC, abstractmethod
from torch.autograd import grad

import torch
from typing import Dict


class BaseOperator(ABC):
    def __init__(self, sigma_noise=0.0, unnorm_shift=0.0, unnorm_scale=1.0, device='cuda'):
        self.sigma_noise = sigma_noise
        self.unnorm_shift = unnorm_shift
        self.unnorm_scale = unnorm_scale
        self.device = device

    @abstractmethod
    def forward(self, inputs, **kwargs):
        '''
        inputs : torch.tensor with shape (batch_size, ...). 
                 Note that inputs have been normalized to the input range of pre-trained diffusion models.
        '''
        pass

    def __call__(self, 
                 inputs: Dict,
                 **kwargs):

        target = inputs['target']
        # calculate A(x)
        out = self.forward(target, **kwargs)
        # add noise
        return out + self.sigma_noise * torch.randn_like(out)

    def gradient(self, pred, observation, return_loss=False):
        """
            Use torch.autograd to compute gradient w.r.t. predicted parameters, 
                i.e., \nabla_{pred} loss(pred, observation). 
            Note that some inverse problems may not support torch.autograd.grad or do not allow for gradient access. 
        Args:
            - pred (torch.tensor): predicted parameters (not measurement), shape (batch_size, ...)
            - observation (torch.tensor): observed data, shape (1, ...)
            - return_loss (bool): whether to return loss scale
        Returns:
            - pred_grad (torch.tensor): gradient of loss w.r.t. predicted parameters, shape (batch_size, ...)
            - loss (torch.tensor): loss value, shape (batch_size, ) if return_loss is True
        """
        pred_tmp = pred.clone().detach().requires_grad_(True)
        loss = self.loss(pred_tmp, observation).sum()
        pred_grad = grad(loss, pred_tmp)[0]
        if return_loss:
            return pred_grad, loss
        else:
            return pred_grad

    def loss(self, pred, observation, **kwargs):
        """
            data consistency loss between prediction and given observation
            default as L2 loss (summation over batches)
        Args:
            - pred (torch.tensor): predicted parameters (not measurement), shape (batch_size, ...)
            - observation (torch.tensor): observed data, shape (1, ...)
        Returns:
            - loss (torch.tensor): loss value, shape (batch_size, )
        """
        return (self.forward(pred) - observation).square().flatten(start_dim=1).sum(dim=1)
    
    def loss_m(self, measurements, observation):
        '''
        Calculate the loss function for a batch of measurements
        Args:
            - measurements (torch.tensor): measurements of predicted signal
            - observation (torch.tensor): actual observation
        '''
        return (measurements - observation).square().flatten(start_dim=1).sum(dim=1)
    
    @torch.enable_grad()
    def gradient_m(self, measurements, observation):
        '''
        Gradient of the loss function w.r.t. measurements
        Args:
            - measurements (torch.tensor): measurements of predicted signal, (batch_size, ...)
            - observation (torch.tensor): actual observation, (batch_size, ...)
        '''
        mea_tmp = measurements.clone().detach().requires_grad_(True)
        loss = self.loss_m(mea_tmp, observation).sum()
        grad_m = grad(loss, mea_tmp)[0]
        return grad_m   
        
    def unnormalize(self, inputs):
        return (inputs + self.unnorm_shift) * self.unnorm_scale
    
    def normalize(self, inputs):
        return inputs / self.unnorm_scale - self.unnorm_shift
    
    def close(self):
        # release resources if necessary
        pass
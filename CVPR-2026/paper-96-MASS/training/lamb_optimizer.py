"""LAMB optimizer registration for MASS training.

The optimizer is used by the pretraining config for stable large-batch
optimization and is registered under the ``Lamb`` key for YAML construction.
"""

import math
import torch
from torch.optim import Optimizer
from typing import List, Dict, Optional, Tuple, Union, Any, Callable

from utils.registry import register_optimizer


@register_optimizer("Lamb")
class Lamb(Optimizer):
    r"""Implements Lamb algorithm for large batch optimization.
    
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
            
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """
    
    def __init__(
        self, 
        params, 
        lr: float = 1e-3, 
        betas: Tuple[float, float] = (0.9, 0.999), 
        eps: float = 1e-6,
        weight_decay: float = 0, 
        adam: bool = False
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(
            lr=lr, 
            betas=betas, 
            eps=eps,
            weight_decay=weight_decay
        )
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instead.')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Paper v3 does not use debiasing.
                step_size = group['lr']
                
                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                
                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])
                
                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                
                if self.adam:
                    trust_ratio = 1
                
                p.data.add_(adam_step, alpha=-step_size * trust_ratio)
        
        return loss

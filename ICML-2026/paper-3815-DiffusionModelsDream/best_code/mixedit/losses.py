import torch
import torch.optim as optim
import numpy as np
from .model import utils as mutils
import random
from functools import partial
import torch.nn.functional as F
from .sde import ContinuousSDE

_LOSSES = {}


def register_loss_fn(cls=None, *, name=None):
    """A decorator for registering loss classes."""
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _LOSSES:
            raise ValueError(f'Already registered model with name: {local_name}')
        _LOSSES[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_loss_fn(name):
    return _LOSSES[name]


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(
            params, lr=config.optim.lr, 
            betas=(config.optim.beta1, config.optim.beta2), 
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay
        )
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            params, lr=config.optim.lr, 
            betas=(config.optim.beta1, config.optim.beta2), 
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay
        )
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!'
        )

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""
    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(loss_type, sde, train, optimize_fn, accum, **kwargs):
    loss_fn = get_loss_fn(loss_type)(sde, train, **kwargs)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch, cond=cond).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn


# ELBO loss function
@register_loss_fn(name="elbo")
def elbo_loss_fn(sde, train, sampling_eps=1e-4, **kwargs):
    simul_steps = kwargs.get("simul_steps", 0)
    interpolant_fn = sde.interpolant if simul_steps == 0 else \
        partial(sde.interpolant_simul, simul_steps=simul_steps)

    def loss_fn(model, batch, cond=None):
        """
        Batch shape: [B, L, D]
        """
        drift_fn = mutils.get_drift_fn(model, sde, train=train, sampling=False)

        if sde.scheduler.weight_type == "default" or not train:
            t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device)
        else:
            t = sde.scheduler.importance_weighted_time((batch.shape[0],), batch.device)

        prior_sample = sde.prior_sample(batch.shape, batch.device, t=t)
        interpolant = interpolant_fn(prior_sample, batch, t)

        drift = sde.drift(batch, interpolant, t)
        loss = sde.manifold.squared_norm(drift_fn(interpolant, t) - drift)
        loss = 0.5 * loss.sum(dim=-1) * sde.scheduler.importance_weight(t, train) / sde.diffusion(interpolant, t).square()

        return loss
     
    return loss_fn


# Cross-entropy loss function
@register_loss_fn(name="ce")
def ce_loss_fn(sde, train, sampling_eps=1e-4, **kwargs):
    simul_steps = kwargs.get("simul_steps", 0)
    interpolant_fn = sde.interpolant if simul_steps == 0 else \
        partial(sde.interpolant_simul, simul_steps=simul_steps)

    def loss_fn(model, batch, cond=None):
        """
        Batch shape: [B, L, D]
        """
        model_fn = mutils.get_model_fn(model, train=train)

        if sde.scheduler.weight_type == "default" or not train:
            t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device)
        else:
            t = sde.scheduler.importance_weighted_time((batch.shape[0],), batch.device)

        prior_sample = sde.prior_sample(batch.shape, batch.device, t=t)
        interpolant = interpolant_fn(prior_sample, batch, t)

        output = model_fn(interpolant, t)
        loss = torch.vmap(torch.nn.CrossEntropyLoss(reduction='sum'))(
            output.to(torch.float32), batch[...,:output.shape[-1]].argmax(-1)
        )
        loss = loss * sde.scheduler.importance_weight(t, train)

        return loss
     
    return loss_fn

@register_loss_fn(name="focal")
def focal_loss_fn(sde, train, gamma=2.0, alpha=1.0, sampling_eps=1e-4, **kwargs):
    # Handle simulation steps for interpolant if needed
    simul_steps = kwargs.get("simul_steps", 0)
    interpolant_fn = sde.interpolant if simul_steps == 0 else \
        partial(sde.interpolant_simul, simul_steps=simul_steps)

    def loss_fn(model, batch, cond=None):
        """
        Computes Focal Loss.
        Batch is expected to be One-Hot encoded [B, L, V].
        """
        model_fn = mutils.get_model_fn(model, train=train)

        # Sample time t
        if sde.scheduler.weight_type == "default" or not train:
            t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device)
        else:
            t = sde.scheduler.importance_weighted_time((batch.shape[0],), batch.device)

        # Get interpolant (x_t)
        prior_sample = sde.prior_sample(batch.shape, batch.device, t=t)
        interpolant = interpolant_fn(prior_sample, batch, t)

        # Model forward pass -> logits
        logits = model_fn(interpolant, t) # [B, L, V]
        
        # Get integer targets from one-hot batch
        # Slicing matches dimensions if mask token logic requires it
        targets = batch[..., :logits.shape[-1]].argmax(dim=-1) # [B, L]

        # --- Focal Loss Calculation ---
        # 1. Calculate standard Cross Entropy (log_pt)
        #    log_softmax gives log(p) for all classes
        log_p = F.log_softmax(logits, dim=-1) 
        
        # 2. Gather log_pt for the correct classes
        #    gather requires indices to have same dim as input, so we unsqueeze
        log_pt = log_p.gather(-1, targets.unsqueeze(-1)).squeeze(-1) # [B, L]
        
        # 3. Calculate pt and the focal weight
        pt = log_pt.exp()
        focal_weight = alpha * (1 - pt) ** gamma

        # 4. Final weighted loss
        loss = -focal_weight * log_pt # [B, L]
        
        # Sum over sequence length (L) to match 'reduction=sum' behavior of the original code
        loss = loss.sum(dim=-1) # [B]

        # Apply importance weighting from SDE scheduler
        loss = loss * sde.scheduler.importance_weight(t, train)

        return loss

    return loss_fn


# Updated loss function to handle joint discrete/continuous training
@register_loss_fn(name="mixed_joint")
def mixed_loss_fn(sde_discrete, sde_continuous, train, sampling_eps=1e-4, gamma=0.1, **kwargs):
    # sde_discrete: RDLM/LogBridge SDE
    # sde_continuous: A standard Gaussian/VP SDE for continuous data
    
    simul_steps = kwargs.get("simul_steps", 0)
    interpolant_fn = sde_discrete.interpolant if simul_steps == 0 else \
        partial(sde_discrete.interpolant_simul, simul_steps=simul_steps)

    def loss_fn(model, batch, cond=None):
        """
        Batch is now a tuple: (discrete_indices [B, L], continuous_values [B, L, C])
        """
        discrete_x, continuous_x = batch
        
        # --- 1. Sample Independent Timesteps ---
        # Discrete Timestep
        if sde_discrete.scheduler.weight_type == "default" or not train:
            t_disc = (1 - sampling_eps) * torch.rand(discrete_x.shape[0], device=discrete_x.device)
        else:
            t_disc = sde_discrete.scheduler.importance_weighted_time((discrete_x.shape[0],), discrete_x.device)

        # Continuous Timestep (Standard Uniform)
        t_cont = (1 - sampling_eps) * torch.rand(continuous_x.shape[0], device=continuous_x.device)

        # --- 2. Forward Diffusion (Noising) ---
        # Discrete Noising (Existing Logic)
        prior_sample_d = sde_discrete.prior_sample(discrete_x.shape, discrete_x.device, t=t_disc)
        # Ensure discrete_x is one-hot for the interpolant if required
        # discrete_x_oh = F.one_hot(discrete_x, num_classes=sde_discrete.vocab_size).float() 
        interpolant_disc = interpolant_fn(prior_sample_d, discrete_x, t_disc)
        
        # Continuous Noising (Gaussian Diffusion)
        eps = torch.randn_like(continuous_x)
        mean, std = sde_continuous.marginal_prob(continuous_x, t_cont)
        interpolant_cont = mean + std * eps

        # --- 3. Joint Model Forward Pass ---
        # Model now accepts both interpolants and both timesteps
        model_fn = mutils.get_model_fn_mixed(model, train=train)
        # unsqueeze so both are "1 SL"
        out_disc, out_cont = model_fn(interpolant_disc, interpolant_cont.unsqueeze(1), t_disc, t_cont)

        # --- 4. Calculate Combined Loss ---
        # Discrete Loss (Cross Entropy)
        loss_d = torch.vmap(torch.nn.CrossEntropyLoss(reduction='sum'))(
            out_disc.to(torch.float32), discrete_x
        )
        loss_d = loss_d * sde_discrete.scheduler.importance_weight(t_disc, train)

        # Continuous Loss (MSE on noise)
        if out_cont.shape != eps.shape and eps.ndim == 2:
            eps = eps.unsqueeze(1)
        eps = eps.to(dtype=out_cont.dtype)
             
        loss_c = F.mse_loss(out_cont, eps, reduction='none').sum(dim=-1).sum(dim=-1)
        
        return (1.-gamma)*loss_d + gamma * loss_c, loss_d, loss_c

    return loss_fn
import torch
from typing import Optional
from torch import nn, Tensor

@torch.no_grad()
def uniform_quantize(x, scale, zero_point, nbits):    
    x_int = torch.clamp(torch.round(x / scale) + zero_point, 0, 2**nbits - 1)
    return (x_int - zero_point) * scale

def quantize_zfold(weight, scale, zero, zeta, nbits):
    return uniform_quantize(weight, scale * zeta, zero, nbits)

@torch.no_grad()
def compute_loss_perturb(weight, scale, zero, zeta, nbits, H):
    delta_w = quantize_zfold(weight, scale, zero, zeta, nbits) - weight
    if len(H.shape) == 3:
        num_heads = H.shape[0]
        delta_w = delta_w.view(num_heads, -1, delta_w.shape[-1])
    
    return ((delta_w @ H) * delta_w).sum()

def damping(H, percdamp=.01):  
    # Calculate the mean of diagonals across all heads  
    mean_diags = torch.mean(torch.diagonal(H, dim1=-2, dim2=-1), dim=-1)  

    # Add the damping values back into the original tensor along the diagonals  
    H.diagonal(dim1=-2, dim2=-1).add_(mean_diags.view(-1, *[1]*(len(H.shape)-2)), alpha=percdamp)  

    return H

def filter_dead_neuron(W, H, replace=1/2048, percdamp=.01, apply_damping=True):
    if len(H.shape) == 2:  
        H = H.unsqueeze(0)  
    num_heads, in_features = H.shape[0], H.shape[-1]  
    W = W.view(num_heads, -1, in_features)  

    # Extract the diagonals of H and find indices where they are equal to 0  
    diagonals = torch.diagonal(H, dim1=-2, dim2=-1)  
    idx_dead = (diagonals < 1e-8)  

    # Set the corresponding columns of W to 0 and replace the dead neurons in H with the given value  
    mask = ~idx_dead.unsqueeze(-2)
    W *= mask  
    H.diagonal(dim1=-2, dim2=-1)[idx_dead] = replace  

    if apply_damping:
        H = damping(H, percdamp)

    W = W.view(-1, in_features)  
    H = H.squeeze()

    return W, H

@torch.jit.script
def _grid_search_impl_H(
    w_flat: Tensor,
    zeta: Tensor,
    w_min: Tensor,
    w_max: Tensor,
    n_bits: int,
    symmetric: bool,
    H: Tensor,
    cov_G: Optional[Tensor],
    Hinv: Optional[Tensor],
    correction: Optional[Tensor],
):
    # torchscriptable code
    best_score = torch.full_like(w_min, 1e10)
    best_scale = torch.ones_like(best_score)

    H_from_Hinv = torch.zeros_like(H)

    if Hinv is not None:
        eye = torch.eye(Hinv.shape[-1], device=Hinv.device, dtype=Hinv.dtype)
        H_from_Hinv = torch.cholesky_solve(eye, Hinv, upper=True)
    
    if symmetric:
        best_zero_point = torch.full_like(best_score, 2**(n_bits-1))
    else:
        best_zero_point = torch.zeros_like(best_score)

    for clip_ratio in torch.arange(1.0, 0.0, -0.01):
        new_max, new_min = w_max * clip_ratio, w_min * clip_ratio
        new_scale = (new_max - new_min) / (2**n_bits - 1)

        if symmetric: # Disabled
            assert False, "Symmetric quantization is not supported."
            w_hat = new_scale * (torch.clamp((w_flat / new_scale + best_zero_point).round(), 0, 2**n_bits - 1) - best_zero_point)

            # Reshape for zeta broadcasting
            delta_w = (w_hat - w_flat) * zeta
            score = torch.sum((delta_w @ H) * delta_w, dim=-1, keepdim=True)

            best_scale = torch.where(score < best_score, new_scale, best_scale)
            best_score = torch.minimum(score, best_score)

        else:
            for round in ("floor", "ceil"):
                new_zeropoint = torch.floor(-new_min / new_scale) if round == "floor" else torch.ceil(-new_min / new_scale)
                # new_zeropoint = torch.round(-new_min / new_scale)
                w_hat = new_scale * (torch.clamp((w_flat / new_scale + new_zeropoint).round(), 0, 2**n_bits - 1) - new_zeropoint)
                delta_w = (w_hat - w_flat) * zeta

                if correction is not None:
                    delta_w = delta_w - correction
                    
                if Hinv is not None:
                    if cov_G is not None:
                        score = torch.sum(cov_G * (delta_w @ H_from_Hinv @ delta_w.transpose(-1, -2)), dim=-1, keepdim=True)
                    else:
                        score = torch.sum((delta_w @ H_from_Hinv) * delta_w, dim=-1, keepdim=True)
                else:
                    if cov_G is None:
                        score = torch.sum((delta_w @ H) * delta_w, dim=-1, keepdim=True)
                    else:
                        score = torch.sum(cov_G * (delta_w @ H @ delta_w.transpose(-1, -2)), dim=-1, keepdim=True)

                best_scale = torch.where(score < best_score, new_scale, best_scale)
                best_zero_point = torch.where(score < best_score, new_zeropoint, best_zero_point)
                best_score = torch.minimum(score, best_score)

    return best_scale, best_zero_point

@torch.no_grad()
def find_quant_params(weight, zeta, n_bits, symmetric, H, cov_G=None, Hinv=None, correction=None):
    assert H is not None, "Hessian should be given."

    target_dim = (*weight.shape[:-1], 1)
    w_flat = weight
    in_features = w_flat.shape[-1]
    
    w_flat = w_flat / zeta
    if len(H.shape) == 2:
        H = H.unsqueeze(0)
    zeta = zeta.view(1, 1, in_features)

    tmp = torch.zeros((*w_flat.shape[:-1], 1), device=w_flat.device)
    w_max = torch.maximum(torch.max(w_flat, dim=-1, keepdim=True).values, tmp)
    w_min = torch.minimum(torch.min(w_flat, dim=-1, keepdim=True).values, tmp)

    if symmetric:
        w_max = torch.maximum(torch.abs(w_min), w_max)
        tmp = w_min < 0
        if torch.any(tmp):
            w_min[tmp] = -w_max[tmp]

    tmp = (w_min == 0) & (w_max == 0)
    w_max[tmp] = 1
    w_min[tmp] = -1

    scale, zero_point = _grid_search_impl_H(w_flat, zeta, w_min, w_max, n_bits, symmetric, H, cov_G, Hinv, correction)
    
    return scale.view(target_dim), zero_point.view(target_dim)


def refine_qparams_zfold(wrappers: dict, zeta_share_list: list, hyperparams: dict):
    import time 
    tick = time.time()
    
    for i, name in enumerate(zeta_share_list):
        if i > 0:
            break
        in_features = wrappers[name].layer.weight.data.shape[-1]
        n_bits = wrappers[name].quantizer.nbits
        sym = wrappers[name].quantizer.sym

    # pre-processing
    W_group, H_group = {}, {}
    for name in zeta_share_list:
        W, H = wrappers[name].layer.weight.data.clone(), wrappers[name].H.clone()
        W, H = filter_dead_neuron(W, H, replace=hyperparams['replace'], percdamp=hyperparams['percdamp'], apply_damping=True)
        W_group[name], H_group[name] = W, H
     
    # initialize qparams
    zeta = torch.ones([1, in_features], device=W.device)
    scale_group, zero_group = {}, {}
    for name in zeta_share_list:
        quantizer = wrappers[name].quantizer
        scale_group[name], zero_group[name] = quantizer.scale.view([-1, 1]), quantizer.zero.view([-1, 1])
        
    # compute initial loss perturbation incurred by quantization
    loss_perturb_initial = 0
    for name in zeta_share_list:
        loss_perturb_initial += compute_loss_perturb(W_group[name], scale_group[name], zero_group[name], zeta, n_bits, H_group[name])

    loss_perturb_before = loss_perturb_initial
    best_scale_group, best_zero_group, best_zeta = scale_group.copy(), zero_group.copy(), zeta

    # update scale/zero and zeta alternatively
    count_update = 0
    while count_update < 30:
        count_update += 1

        # update zeta
        zeta = find_zeta(W_group, scale_group, zero_group, zeta, n_bits).view(zeta.shape)
        zeta = torch.where(zeta==0., torch.ones(1).cuda(), zeta)

        # update scale and zero-point
        for name in zeta_share_list:
            scale_group[name], zero_group[name] = find_quant_params(W_group[name], zeta, n_bits, sym, H_group[name])
        
        # compute loss perturbation after update
        loss_perturb_after = 0
        for name in zeta_share_list:
            loss_perturb_after += compute_loss_perturb(W_group[name], scale_group[name], zero_group[name], zeta, n_bits, H_group[name])
        
        if loss_perturb_after > loss_perturb_before:
            break
        else:
            loss_perturb_before = loss_perturb_after
            best_scale_group, best_zero_group, best_zeta = scale_group.copy(), zero_group.copy(), zeta
    
    delta_loss_improvement = loss_perturb_initial - loss_perturb_before
    num_updates = count_update - 1

    for name in zeta_share_list:
        quantizer = wrappers[name].quantizer
        quantizer.scale.data = best_scale_group[name].view(quantizer.scale.shape)
        quantizer.zero.data = best_zero_group[name].view(quantizer.zero.shape)
        quantizer.zeta.data = best_zeta
    
    return delta_loss_improvement, num_updates, time.time() - tick


def find_zeta(W_group: dict, scale_group: dict, zero_group: dict, zeta, n_bits, eps=1e-10):
    W_stack = torch.cat(list(W_group.values()), dim=0)
    scale = torch.cat(list(scale_group.values()), dim=0)
    zero = torch.cat(list(zero_group.values()), dim=0)

    W_hat = uniform_quantize(W_stack / zeta, scale, zero, n_bits)
    p = torch.bmm(W_hat.T.unsqueeze(-2), W_stack.T.unsqueeze(-1))
    q = torch.bmm(W_hat.T.unsqueeze(-2), W_hat.T.unsqueeze(-1))
    q = eps * torch.ones_like(q) + q

    return p / q


def refine_qparams_with_hessian(wrappers, idx_block, model_type, use_zfold, hyperparams, llm_config):
    if use_zfold:
        from model_utils import set_zfold_layers
        zfold_list, zeta_share_lists = set_zfold_layers(model_type, llm_config)  # define layers where Z-Fold can be applied
    else:
        zfold_list, zeta_share_lists = [], {}

    print('+---------------------------+---------------------------+--------+-----------+')
    print('|           Layer           |   delta-loss-improvement  |  time  | num_iters |')
    print('+===========================+===========================+========+===========+')

    compute_zeta_list = []
    # compute shared zeta first
    for share_name, zeta_share_list in zeta_share_lists.items():
        delta_loss_improvement, num_updates, refine_time = refine_qparams_zfold(wrappers, zeta_share_list, hyperparams)
        print(f'|{idx_block}: {share_name : <24}| {delta_loss_improvement:.3f}\t| {refine_time:.2f}\t| {num_updates : <2}|')
        compute_zeta_list += zeta_share_list

    for name, wrapper in wrappers.items():
        if name in compute_zeta_list:
            continue
        else:
            delta_loss_improvement, num_updates, refine_time = wrapper.refine_quant_params(name in zfold_list, hyperparams)
        
        print(f'|{idx_block}: {name : <24}| {delta_loss_improvement:.3f}\t| {refine_time:.2f}\t| {num_updates : <2}|')
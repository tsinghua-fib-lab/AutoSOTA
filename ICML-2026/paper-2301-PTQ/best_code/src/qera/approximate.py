from copy import deepcopy
import logging
from functools import partial

import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
import pandas as pd

from .utils import (
    find_matched_pattern,
    get_layer_by_name,
    get_full_device_map,
    move_module_to_device,
    load_and_process_hessian,

)
from .quantize import get_quantizer
from .approximate_with_init import _compute_scale_inv_dot_U
from .approximate_with_init import get_lr_initializer

from pathlib import Path

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_q_loss(weight_orig: torch.Tensor, 
                   weight_q: torch.Tensor, 
                   hess: torch.Tensor=None
                  ):
    '''
    Compute actaware if hess exists. (hess = 2x.Tx)
    weight_q can be either W_q or W_q + A@B
    '''
    W_o, W_q = weight_orig, weight_q
    H = (hess/2).to(W_o.device) if hess is not None else None
    
    q_loss_act, q_loss = 0, 0
    if H is not None:
        q_loss_act = (torch.trace((W_o - W_q) @ H @ (W_o - W_q).T) / torch.trace((W_o) @ H @ (W_o.T))).item()
    q_loss = torch.nn.functional.mse_loss(W_o,W_q).item()
    
    return q_loss, q_loss_act

@torch.no_grad()
def compute_AB_and_approximation_error(
    model,
    layers_to_approximate: list[str],
    quant_config: dict,
    scale_dict: dict[str, torch.Tensor],
    hess_dict: dict[str, torch.Tensor] = None,
    move_model_back: bool = True,
    srr_seed: int = 42,
):
    LR_dict = {}
    W_q_dict = {}
    df = pd.DataFrame(columns=["layer_name", "mse", "act_mse", "rank"])

    full_device_map = get_full_device_map(model)
    model = model.to("cpu")

    torch.manual_seed(srr_seed)
    torch.cuda.manual_seed_all(srr_seed)

    for layer_name in tqdm(layers_to_approximate, desc="Computing low-rank A and B"):
        torch.cuda.empty_cache()
        # device
        layer = get_layer_by_name(model, layer_name)
        layer.to(full_device_map[layer_name])
    
        # scale
        scale = scale_dict[layer_name].clone() if scale_dict is not None else None
        hess = hess_dict[layer_name].clone() if hess_dict is not None else None
        
        # qera config
        matched_entry = find_matched_pattern(layer_name, quant_config.keys())
        if isinstance(matched_entry, str):
            matched_entry = quant_config[matched_entry]
        layer_quant_config = deepcopy(quant_config[matched_entry])        

        layer_LR_dict, layer_W_q_dict, (mse, act_mse) = _compute_scales_and_error_for_fc(
            layer_name, layer, scale, layer_quant_config, hess=hess
        )
        if layer_LR_dict is not None:
            layer_LR_dict = {k: v.to("cpu") for k, v in layer_LR_dict.items()}
            LR_dict.update(layer_LR_dict)
        W_q_dict.update(layer_W_q_dict)
        df.loc[len(df)] = [layer_name, mse, act_mse, layer_quant_config["rank"]]
        layer.to("cpu")

    if move_model_back:
        move_module_to_device(model, full_device_map)

    return LR_dict, W_q_dict, df


def _compute_scales_and_error_for_fc(
    layer_name: str,
    layer: torch.nn.Linear,
    scale: torch.Tensor,
    layer_quant_config: dict,
    hess: torch.Tensor = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], float]:
    """

    q_error_T = W^T - W_q^T
    SVD(S @ q_error_T) = U @ S @ V^T

    A = S^-1 @ U_k
    B = S_k @ V_k^T

    y_hat = x @ (W_q^T + AB)
          = x @ (W_q^T + S^-1 @ U_k @ S_k @ V_k^T)
          = x @ (W_q^T + S^-1 @ (S @ q_error_T)_k)
          ~ x @ (W_q^T + W^T - W_q^T)
          = x @ W^T

    """
    cfg = layer_quant_config
    rank = cfg["rank"]
    num_iter = cfg['iter']
    device, dtype  = layer.weight.device, layer.weight.dtype

    w_q_config = cfg["w_quantizer"]
    w_quantizer_name = w_q_config.pop("name")
    if w_quantizer_name in cfg['hess_require_quantizer']:
        assert isinstance(layer, nn.Linear) or isinstance(layer, transformers.Conv1D), "Currently only support Linear"
        w_q_config["hess"] = hess.to(dtype).to(device)
        
    w_quantizer = partial(
        get_quantizer(w_quantizer_name),
        **w_q_config,
    )

    ab_q_config = cfg["x_quantizer"]
    ab_quantizer = partial(get_quantizer(ab_q_config.pop("name")), **ab_q_config)

    #  lr init
    lr_it_config = cfg["lr_initializer"]
    lr_it_name = lr_it_config.pop("name")
    if lr_it_name in cfg['hess_require_lr_init']:
        lr_it_config["hess"] = hess.to(dtype).to(device)

    lr_it_config["layer_name"] = layer_name # TODO:
    weight_orig = layer.weight.detach().clone()
    
    if scale is None: 
        #This is for w-only
        weight_q = w_quantizer(weight_orig).clone()
        q_loss, q_loss_act = compute_q_loss(weight_orig, weight_q, hess=hess)
        
        if q_loss > 1e-3:
            logger.warning(f"Mean squared error for {layer_name}: {q_loss}")
        
        print(f"||W-Q||: {q_loss:.6f} ||(W-Q)X||: {q_loss_act:.6f} | {layer_name}") 
        W_q_dict = {layer_name + ".W_q": weight_q.detach().cpu()}    
        return None, W_q_dict, (q_loss, q_loss_act)
    else:
        init_fn = get_lr_initializer(lr_it_name)
        L, R, k_star = init_fn(layer.weight, scale, rank, lr_it_config, return_ada=True)
        
        A = L 
        B = R 
        A_name = layer_name + ".L"
        B_name = layer_name + ".R"
        scale = scale.to(dtype).to(device) 
        
        if num_iter == 1:
            w_res = weight_orig - (A @ B).T 
            weight_q = w_quantizer(w_res).to(device)

            if scale.ndim == 1:
                assert (
                    scale.shape[0] == weight_orig.shape[1]
                ), "Scale must have the same number of elements as the weight"
                scaled_q_error_T = torch.diag(scale) @ ((weight_orig - weight_q).transpose(0, 1))
            elif scale.ndim == 2:
                assert scale.shape[0] == scale.shape[1], "Scale must be a square matrix"
                scaled_q_error_T = scale @ ((weight_orig - weight_q).transpose(0, 1))
            else:
                raise ValueError("Scale must be either a vector (diagonal) or a matrix")
            # svd
            U, S, V_T = torch.linalg.svd(scaled_q_error_T)
        
            U = U[:, :rank]
            S = S[:rank]
            V_T = V_T[:rank, :]

            if scale.ndim == 1:
                A = _compute_scale_inv_dot_U(scale, U)
                B = ab_quantizer(torch.diag(S) @ V_T)
            elif scale.ndim == 2:
                A = _compute_scale_inv_dot_U(scale, U)
                B = ab_quantizer(torch.diag(S) @ V_T)
            else:
                raise ValueError("Scale must be either a vector (diagonal) or a matrix")

            q_loss, q_loss_act = compute_q_loss(weight_orig, weight_q + (A@B).T, hess=hess)

            if q_loss > 1e-3:
                logger.warning(f"Mean squared error for {layer_name}: {q_loss}")
            
            print(f"||W-Q-LR||: {q_loss:.6f} ||(W-Q-LR)X||: {q_loss_act:.6f} | {layer_name}") 
        else:
            for i in range(num_iter):
                w_res = weight_orig - (A @ B).transpose(0,1)
                weight_q = w_quantizer(w_res).to(device)
                
                if scale.ndim == 1:
                    assert (
                        scale.shape[0] == weight_orig.shape[1]
                    ), "Scale must have the same number of elements as the weight"
                    scaled_q_error_T = torch.diag(scale) @ (weight_orig - weight_q).transpose(0, 1)
                elif scale.ndim == 2:
                    assert scale.shape[0] == scale.shape[1], "Scale must be a square matrix"
                    scaled_q_error_T = scale @ (weight_orig - weight_q).transpose(0, 1)
                else:
                    raise ValueError("Scale must be either a vector (diagonal) or a matrix")

                # svd      
                U, S, V_T = torch.linalg.svd(scaled_q_error_T)
                U = U[:, :rank]
                S = S[:rank]
                V_T = V_T[:rank, :]

                # scale^-1
                if scale.ndim == 1:
                    A = _compute_scale_inv_dot_U(scale, U)
                    B = ab_quantizer(torch.diag(S) @ V_T)
                elif scale.ndim == 2:
                    A = _compute_scale_inv_dot_U(scale, U)
                    B = ab_quantizer(torch.diag(S) @ V_T)
                else:
                    raise ValueError("Scale must be either a vector (diagonal) or a matrix")
                
                q_loss, q_loss_act = compute_q_loss(weight_orig, weight_q + (A@B).T, hess=hess)
                
                if q_loss > 1e-3:
                    logger.warning(f"Mean squared error for {layer_name}: {q_loss}")
                print(f"Num iter: {i+1}")
                print(f"||W-Q-LR||: {q_loss:.6f} ||(W-Q-LR)X||: {q_loss_act:.6f} | {layer_name}") 
                
        W_q_dict = {layer_name + ".W_q": weight_q.detach().cpu()}    
        return {A_name: A, B_name: B}, W_q_dict, (q_loss, q_loss_act)


def attach_LR(model, layers_to_approximate, AB_dict: dict[str, torch.Tensor]):
    if AB_dict:
        for layer_name in layers_to_approximate:
            A = AB_dict[layer_name + ".L"]
            B = AB_dict[layer_name + ".R"]

            layer: torch.nn.Linear = get_layer_by_name(model, layer_name)
            device = layer.weight.device
            dtype = layer.weight.dtype
            A = A.to(dtype).to(device)
            B = B.to(dtype).to(device)
            
            layer.A = torch.nn.Parameter(A)
            layer.B = torch.nn.Parameter(B)

    return model


def attach_quantized_weight(model, layers_to_approximate, W_q_dict: dict[str, torch.Tensor]):
    for layer_name in layers_to_approximate:
        W_q = W_q_dict[layer_name + ".W_q"]

        layer: torch.nn.Linear = get_layer_by_name(model, layer_name)
        device = layer.weight.device
        dtype = layer.weight.dtype

        layer.weight.data = W_q.to(device).to(dtype)
        layer._w_is_quantized = True
        
    return model

import tqdm
import torch
import bitsandbytes as bnb
from peft.utils.loftq_utils import _SafetensorLoader, NFQuantizer
from peft.tuners.lora import Linear4bit
from peft.tuners.lora.layer import Linear as LoraLinear

from ..quantize.quantizers import mxint_quantizer
from qera.approximate_with_init import get_lr_initializer
import torch.nn as nn


#################################loftQ####################################


def _low_rank_decomposition_loftq(weight, reduced_rank=32):
    """
    :param weight: The matrix to decompose, of shape (H, W) :param reduced_rank: the final rank :return:
    """
    matrix_dimension = len(weight.size())
    if matrix_dimension != 2:
        raise ValueError(
            f"Only support 2D matrix, but your input has {matrix_dimension} dimensions."
        )

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    error_f_norm = S[reduced_rank:].pow(2).sum().sqrt().cpu().item()

    return {
        "L": L,
        "R": R,
        "U": U,
        "S": S,
        "Vh": Vh,
        "reduced_rank": reduced_rank,
        "error_f_norm": error_f_norm,
    }


@torch.no_grad()
def init_lora_loftq_4bit(
    qweight: bnb.nn.Params4bit,
    weight: torch.Tensor,
    num_iters: int,
    reduced_rank: int,
    compute_device,
    init_method: str = None, 
    hess=None,  
):
    """
    Update quantized weight and LoRA weights in a BitsAndBytes 4-bit Lora layer
    """
    compute_device = "cuda" if compute_device is None else compute_device
    assert compute_device != "cpu"

    quant_state = qweight.quant_state
    dequantized_weight = bnb.functional.dequantize_4bit(qweight.data, quant_state)

    residual = weight.clone()
    error_f_norm = []

    for i in range(num_iters):
        qweight = bnb.nn.Params4bit(
            residual.cpu(),
            requires_grad=False,
            compress_statistics=quant_state.nested,
            quant_type=quant_state.quant_type,
            quant_state=None,
            bnb_quantized=False,
            blocksize=quant_state.blocksize,
            quant_storage=qweight.quant_storage,
        ).to(compute_device)
        dequantized_weight = bnb.functional.dequantize_4bit(qweight.data, quant_state)

        weight = weight.to(device=compute_device, dtype=torch.float32)
        dequantized_weight = dequantized_weight.to(
            device=compute_device, dtype=torch.float32
        )
        residual = weight - dequantized_weight

        torch.cuda.empty_cache()
        output = _low_rank_decomposition_loftq(residual, reduced_rank)
        L, R = output["L"], output["R"]
        residual = weight - torch.mm(L, R)

        error_f_norm.append(output["error_f_norm"])

    lora_A, lora_B = R, L

    return lora_A, lora_B, qweight, error_f_norm

def replace_lora_weights_loftq_4bit(
    peft_model,
    model_path=None,
    adapter_name="default",
    num_iters: int = 1,
    compute_device=None,
    init_method: str = None,
    hess_dict=None,
):
    """
    LoftQ (A replicate of Official LoftQ)

    q_weight, weight # (out_dim, in_dim)

    residual = weight - q_weight
    U, S, V_T = svd(residual)
    L = U[:, :rank] @ sqrt(S[:rank]) # L.shape = (out_dim, rank)
    R = sqrt(S[:rank]) @ V_T[:rank, :] # R.shape = (rank, in_dim)

    lora_A = R # lora_A.shape = (rank, in_dim)
    lora_B = L # lora_B.shape = (out_dim, rank)

    forward: x @ (q_weight.T + lora_A.T @ lora_B.T) = x @ (q_weight + lora_B @ lora_A).T = x @ (q_weight + L @ R).T
            = x @ (q_weight + U[:, :rank] @ sqrt(S[:rank]) @ sqrt(S[:rank]) @ V_T[:rank, :]).T
            = x @ (q_weight + U[:, :rank] @ S[:rank] @ V_T[:rank, :]).T = x @ (q_weight + residual).T
            ~ x @ weight.T
    """
    prefix = "base_model.model."
    safetensor_loader = _SafetensorLoader(peft_model, model_path)

    named_modules = {name: module for name, module in peft_model.named_modules()}
    error_dict = {}

    for name, module in tqdm.tqdm(
        named_modules.items(), desc="Replacing LoRA adapters (LoftQ)"
    ):
        if not isinstance(module, Linear4bit):
            continue
        if not name.startswith(prefix):
            raise TypeError(f"Not peft model")

        name = name[len(prefix) :]
        ori_weight = safetensor_loader.get_tensor(name + ".weight")

        reduced_rank = module.r[adapter_name]
        lora_layer_scaling = module.scaling[adapter_name]

        lora_A, lora_B, new_qweight, error_f_norm = init_lora_loftq_4bit(
            qweight=module.weight,
            weight=ori_weight,
            num_iters=num_iters,
            reduced_rank=reduced_rank,
            compute_device=compute_device,
            init_method=init_method,
            hess=None,
        )
        error_dict[name] = error_f_norm

        module.lora_A[adapter_name].weight.data = lora_A
        module.lora_B[adapter_name].weight.data = lora_B / lora_layer_scaling
        module.weight.data = new_qweight

    return error_dict


@torch.no_grad()
def init_lora_loftq_2bit_bnb(
    weight: torch.Tensor,
    bnb_quant_type: str,
    num_iters,
    reduced_rank: int,
    compute_device,
    init_method: str = None,
    hess=None,  
):
    """
    Emulated 2-bit BitsAndBytes
    """
    compute_device = "cuda" if compute_device is None else compute_device
    assert compute_device != "cpu"
    ori_weight_dtype = weight.dtype

    residual = weight.clone().to(device=compute_device, dtype=torch.float32)
    weight = weight.to(device=compute_device, dtype=torch.float32)
    error_f_norm = []

    quantizer = NFQuantizer(
        num_bits=2, device=compute_device, method=bnb_quant_type, block_size=64
    )

    for i in range(num_iters):
        qweight, max_abs, shape = quantizer.quantize_block(residual)
        dequantized_weight = quantizer.dequantize_block(qweight, max_abs, shape)

        dequantized_weight = dequantized_weight.to(
            device=compute_device, dtype=torch.float32
        )
        residual = weight - dequantized_weight

        torch.cuda.empty_cache()
        output = _low_rank_decomposition_loftq(residual, reduced_rank)
        L, R = output["L"], output["R"]
        residual = weight - torch.mm(L, R)
        error_f_norm.append(output["error_f_norm"])

    lora_A, lora_B = R, L
    return lora_A, lora_B, dequantized_weight.to(ori_weight_dtype), error_f_norm


@torch.no_grad()
def init_lora_loftq_kbit_mxint(
    weight: torch.Tensor,
    num_bits: int,
    num_iters: int,
    reduced_rank: int,
    compute_device,
    block_size,
    init_method: str = None,
    hess=None,  
):
    """
    Emulated k-bit MXInt
    """
    compute_device = "cuda" if compute_device is None else compute_device
    assert compute_device != "cpu"
    ori_weight_dtype = weight.dtype

    residual = weight.clone().to(device=compute_device, dtype=torch.float32)
    weight = weight.to(device=compute_device, dtype=torch.float32)
    error_f_norm = []

    for i in range(num_iters):
        quantized_weight = mxint_quantizer(
            residual, width=num_bits, block_size=block_size, block_axis=-1
        )
        residual = weight - quantized_weight

        torch.cuda.empty_cache()
        output = _low_rank_decomposition_loftq(residual, reduced_rank)
        L, R = output["L"], output["R"]
        residual = weight - torch.mm(L, R)
        error_f_norm.append(output["error_f_norm"])

    lora_A, lora_B = R, L
    return lora_A, lora_B, quantized_weight.to(ori_weight_dtype), error_f_norm


def replace_lora_weights_loftq_kbit(
    peft_model,
    quant_type,
    num_bits: int,
    adapter_name="default",
    num_iters: int = 1,
    compute_device=None,
    mxint_block_size=32,
    init_method: str = None,
    hess_dict=None,
):
    """
    Emulated 2-bit loftq
    """
    assert quant_type in ["nf", "fp", "mxint"]
    if quant_type in ["nf", "fp"]:
        assert num_bits == 2
    prefix = "base_model.model."

    named_modules = {name: module for name, module in peft_model.named_modules()}
    error_dict = {}

    for name, module in tqdm.tqdm(
        named_modules.items(),
        desc=f"Replacing LoRA adapters (Emulated {num_bits}-bit LoftQ)",
    ):
        if not isinstance(module, LoraLinear):
            continue
        if not name.startswith(prefix):
            raise TypeError(f"Not peft model")

        reduced_rank = module.r[adapter_name]
        lora_layer_scaling = module.scaling[adapter_name]

        if quant_type in ["normal", "uniform"]:
            lora_A, lora_B, new_weight, error_f_norm = init_lora_loftq_2bit_bnb(
                weight=module.weight,
                bnb_quant_type="normal" if quant_type == "nf" else "uniform",
                num_iters=num_iters,
                reduced_rank=reduced_rank,
                compute_device=compute_device,
                init_method=init_method,
                hess=None,
            )
        else:
            lora_A, lora_B, new_weight, error_f_norm = init_lora_loftq_kbit_mxint(
                weight=module.weight,
                num_bits=num_bits,
                num_iters=num_iters,
                reduced_rank=reduced_rank,
                compute_device=compute_device,
                block_size=mxint_block_size,
                init_method=init_method,
                hess=None,
            )
        error_dict[name] = error_f_norm

        module.lora_A[adapter_name].weight.data = lora_A
        module.lora_B[adapter_name].weight.data = lora_B / lora_layer_scaling
        module.weight.data = new_weight

    return error_dict



#################################qera####################################

@torch.no_grad()
def _low_rank_decomposition_qera(x: torch.Tensor, reduced_rank: int):
    assert x.ndim == 2
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    Ur  = U[:, :reduced_rank]         
    Sr  = S[:reduced_rank]            
    Vhr = Vh[:reduced_rank, :]        
    sqrt_Sr = torch.diag(torch.sqrt(Sr))
    
    L = Ur
    R = torch.diag(Sr) @ Vhr

    L = L.contiguous()
    R = R.contiguous()

    return L, R

@torch.no_grad()
def _low_rank_decomposition_qera_add(x: torch.Tensor, reduced_rank: int):
    assert x.ndim == 2
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    Ur1  = U[:, :reduced_rank]         
    Sr1  = S[:reduced_rank]           
    Vhr1 = Vh[:reduced_rank, :]        

    L1 = Ur1
    R1 = torch.diag(Sr1) @ Vhr1

    L1 = L1.contiguous()
    R1 = R1.contiguous()

    Ur2  = U[:, reduced_rank:reduced_rank*2]         
    Sr2  = S[reduced_rank:reduced_rank*2]            
    Vhr2 = Vh[reduced_rank:reduced_rank*2, :]        

    L2 = Ur2
    R2 = torch.diag(Sr2) @ Vhr2

    L2 = L2.contiguous()
    R2 = R2.contiguous()

    return L1, R1, L2, R2


@torch.no_grad()
def init_lora_qera_4bit(
    weight: torch.Tensor, scale: torch.Tensor, reduced_rank, compute_device,
    init_method: str = None, hess=None,  
):
    assert scale.ndim in [1, 2]
    assert scale.shape[0] == weight.shape[1]
    compute_device = weight.data.device if compute_device is None else compute_device

    if scale.ndim == 1:
        scale = torch.diag(scale)
        
    weight = weight.to(device=compute_device, dtype=torch.float32)
    scale = scale.to(device=compute_device, dtype=torch.float32)
        
    if init_method:
        init_fn = get_lr_initializer(init_method)
        layer_qera_config = {}
        if hess is not None:
            layer_qera_config["hess"] = hess
        L_init, R_init = init_fn(weight, scale, reduced_rank, layer_qera_config or {})
        weight_residual = weight - (L_init @ R_init).T
    else:
        weight_residual = weight


    blocksize = 64
    compress_statistics = True
    quant_type = "nf4"          
    quant_storage = torch.uint8
    
    q_data, q_state = bnb.functional.quantize_4bit(
        weight_residual.contiguous(),
        blocksize=blocksize,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
        quant_storage=quant_storage,
    )
    
    q_state.orig_shape = tuple(weight_residual.shape)
    q_state.orig_dtype = weight_residual.dtype

    dequantized_weight = bnb.functional.dequantize_4bit(q_data, q_state)

    dequantized_weight = dequantized_weight.view(q_state.orig_shape).to(device=compute_device, dtype=weight.dtype)

    residual = weight - dequantized_weight
    residual_scaled = residual @ scale

    torch.cuda.empty_cache()
    L1, R1_, L2, R2_ = _low_rank_decomposition_qera_add(residual_scaled, reduced_rank)
    R1 = torch.linalg.solve(scale, R1_.T).T  
    R2 = torch.linalg.solve(scale, R2_.T).T  

    corrected = dequantized_weight + L1 @ R1

    error_f_norm = torch.linalg.norm(residual - L1 @ R1, ord="fro").cpu().item()
    lora_A, lora_B = R2, L2

    U, S, Vh = torch.linalg.svd(lora_A @ lora_B, full_matrices=False)
    freeze_alpha_vec = []
    freeze_alpha_collector = {}

    return lora_A, lora_B, corrected.to(weight.dtype), error_f_norm



@torch.no_grad()
def replace_lora_weights_qera_4bit(
    peft_model,
    scale_dict: dict[str, torch.Tensor],
    model_path=None,
    adapter_name="default",
    compute_device=None,
    init_method: str = None,
    hess_dict=None,
):
    prefix = "base_model.model."
    safetensor_loader = _SafetensorLoader(peft_model, model_path)

    named_modules = {name: module for name, module in peft_model.named_modules()}

    error_dict = {}

    for name, module in tqdm.tqdm(
        named_modules.items(), desc="Replacing LoRA adapter (QERA+)"
    ):
        if not isinstance(module, LoraLinear):
            continue
        if not name.startswith(prefix):
            raise TypeError(f"Not peft model")

        name = name[len(prefix) :]
        ori_weight = safetensor_loader.get_tensor(name + ".weight")
        scale = scale_dict[name]
        reduced_rank = module.r[adapter_name]
        lora_layer_scaling = module.scaling[adapter_name]
        
        layer_name = name
        hess = hess_dict[layer_name].clone() if hess_dict is not None and layer_name in hess_dict else None
        if hess is not None:
            hess = hess.to(module.weight.device) 

        lora_A, lora_B, new_weight, error_f_norm = init_lora_qera_4bit(
            weight=ori_weight,
            scale=scale,
            reduced_rank=reduced_rank,
            compute_device=(
                module.weight.device if compute_device is None else compute_device
            ),
            init_method=init_method,
            hess=hess,
        )
        error_dict[name] = [error_f_norm]

        module.lora_A[adapter_name].weight.data = lora_A
        module.lora_B[adapter_name].weight.data = lora_B / lora_layer_scaling
        
        module.weight.data = new_weight.data           
        
    freeze_rank_dict = {}
    freeze_alpha_dict = {}
    return error_dict, freeze_rank_dict, freeze_alpha_dict


@torch.no_grad()
def init_lora_qera_2bit_bnb(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bnb_quant_type: str,
    reduced_rank: int,
    compute_device,
    init_method: str = None,
    hess=None,
):
    assert scale.ndim in [1, 2]
    assert scale.shape[0] == weight.shape[1]
    assert bnb_quant_type in ["normal", "uniform"]
    compute_device = "cuda" if compute_device is None else compute_device

    orig_weight_dtype = weight.dtype
    weight = weight.to(device=compute_device, dtype=torch.float32)

    orig_weight = weight.clone().to(device=compute_device, dtype=torch.float32)
    scale = scale.to(device=compute_device, dtype=torch.float32)

    if scale.ndim == 1:
        scale = torch.diag(scale)

    if init_method:
        init_fn = get_lr_initializer(init_method)
        layer_qera_config = {}
        if hess is not None:
            layer_qera_config["hess"] = hess
        L_init, R_init = init_fn(weight, scale, reduced_rank, layer_qera_config or {})
        weight_residual = weight - (L_init @ R_init).T
    else:
        weight_residual = weight

    quantizer = NFQuantizer(
        num_bits=2, device=compute_device, method=bnb_quant_type, block_size=64
    )

    qweight, max_abs, shape = quantizer.quantize_block(weight_residual)
    dequantized_weight = quantizer.dequantize_block(qweight, max_abs, shape)
    dequantized_weight = dequantized_weight.to(device=compute_device, dtype=torch.float32)

    
    residual = orig_weight - dequantized_weight
    residual_scaled = residual @ scale

    torch.cuda.empty_cache()

    L, R_ = _low_rank_decomposition_qera(residual_scaled, reduced_rank)
    R = torch.linalg.solve(scale, R_.T).T

    error_f_norm = torch.linalg.norm(residual - L @ R, ord="fro").cpu().item()

    lora_A, lora_B = R, L
    return lora_A, lora_B, dequantized_weight.to(orig_weight_dtype), error_f_norm


@torch.no_grad()
def init_lora_qera_kbit_mxint(
    weight: torch.Tensor,
    scale: torch.Tensor,
    num_bits: int,
    reduced_rank: int,
    compute_device,
    block_size,
    init_method: str = None,
    hess=None,
    qera_num_iter: int = 1,
    layer_name: str = None,  
):
    assert scale.ndim in [1, 2]
    assert scale.shape[0] == weight.shape[1]
    compute_device = "cuda" if compute_device is None else compute_device

    orig_weight_dtype = weight.dtype
    orig_weight = weight.clone().to(device=compute_device, dtype=torch.float32)
    scale = scale.to(device=compute_device, dtype=torch.float32)

    if scale.ndim == 1:
        scale = torch.diag(scale)

    if init_method == 'lq_lora':
        init_fn = get_lr_initializer('lrqlr')
    else:
        init_fn = get_lr_initializer(init_method)
    
    freeze_alpha_collector = {}
    layer_qera_config = {
        "freeze_alpha_collector": freeze_alpha_collector,
        "layer_name": layer_name if layer_name is not None else "weight",  
    }
    if hess is not None:
        layer_qera_config["hess"] = hess
    
    if layer_name is not None:
        freeze_alpha_collector[layer_name] = {
            "freeze_r": None,  
            "s_vals": [],  
            "s_ref": None,  
            "alpha_vec": None,  
        }
        
    if init_method == 'srr':
        L_init, R_init, freeze_rank = init_fn(orig_weight, scale, int(reduced_rank), layer_qera_config or {}, return_ada=True) 
    elif init_method == 'lrqlr':
        L_init, R_init = init_fn(orig_weight, scale, int(reduced_rank), layer_qera_config or {})
        freeze_rank = int(reduced_rank)
    else: 
        L_init, R_init = init_fn(weight, scale, int(reduced_rank), layer_qera_config or {})
        freeze_rank = 0
    
    L_tem = R_init.T
    R_tem = L_init.T
    weight_residual = weight - (L_tem @ R_tem)

    for iter in range(qera_num_iter):
        quantized_weight = mxint_quantizer(weight_residual, width=num_bits, block_size=block_size, block_axis=-1)
        residual = orig_weight - quantized_weight 
        residual_scaled = residual @ scale

        if iter+1 == qera_num_iter and init_method == 'lq_lora':
            L = L_tem
            R = R_tem
            break

        if iter+1 != qera_num_iter: 
            L, R_ = _low_rank_decomposition_qera(residual_scaled, reduced_rank)   
            R = torch.linalg.solve(scale, R_.T).T    
            L_tem = L
            R_tem = R
            weight_residual = weight - (L_tem @ R_tem)
            
        else:
            L, R_ = _low_rank_decomposition_qera(residual_scaled, reduced_rank)
            R = torch.linalg.solve(scale, R_.T).T  

    error_f_norm = torch.linalg.norm(residual - L@ R, ord="fro").cpu().item()
    lora_A, lora_B = R, L

    lora_approx = lora_B @ lora_A  
    U, S, Vh = torch.linalg.svd(lora_approx, full_matrices=False)
    
    if isinstance(layer_qera_config, dict):
        collector = layer_qera_config.get("freeze_alpha_collector", None)
        if collector is not None and layer_name is not None:
            collector[layer_name]["s_vals"] = S.detach().cpu().tolist() 
            collector[layer_name]["alpha_vec"] = None  
    
    freeze_alpha_collector[layer_name] = {
        "freeze_r": freeze_rank,
        "s_vals": S.detach().cpu().tolist(),
        "s_ref": float(S[freeze_rank].item()) if freeze_rank < len(S) else float(S[-1].item()),
    }

    return lora_A, lora_B, quantized_weight.to(orig_weight_dtype), error_f_norm, freeze_rank, freeze_alpha_collector


def replace_lora_weight_qera_kbit(
    peft_model,
    scale_dict,
    quant_type,
    num_bits: int,
    adapter_name="default",
    compute_device=None,
    mxint_block_size=32,
    init_method=None, 
    hess_dict=None,
    qera_num_iter: int = 1,
):

    assert quant_type in ["nf", "fp", "mxint"]
    if quant_type in ["fp", "nf"]:
        assert num_bits == 2
    prefix = "base_model.model."

    named_modules = {name: module for name, module in peft_model.named_modules()}
    for name, module in named_modules.items():
        if isinstance(module, Linear4bit):
            print(f"[LoRA target] {name} → {module.__class__.__name__}")

    error_dict = {}
    freeze_rank_dict = {}
    freeze_alpha_dict = {}  

    for name, module in tqdm.tqdm(
        named_modules.items(),
        desc=f"Replacing LoRA adapters (Emulated {num_bits}-bit QERA+)",
    ):
        if not isinstance(module, LoraLinear):
            continue
        if not name.startswith(prefix):
            raise TypeError(f"Not peft model")

        reduced_rank = module.r[adapter_name]
        name = name[len(prefix) :]
        scale = scale_dict[name]
        lora_layer_scaling = module.scaling[adapter_name]
        
        layer_name = name
        hess = hess_dict[layer_name].clone() if hess_dict is not None and layer_name in hess_dict else None
        if hess is not None:
            hess = hess.to(module.weight.device) 

            
        if quant_type in ["nf", "fp"]:
            lora_A, lora_B, new_weight, error_f_norm = init_lora_qera_2bit_bnb(
                weight=module.weight,
                scale=scale,
                bnb_quant_type="normal" if quant_type == "nf" else "uniform",
                reduced_rank=reduced_rank,
                compute_device=(
                    module.weight.device if compute_device is None else compute_device
                ),
                init_method=init_method, 
                hess=hess,
            )
        else:
            lora_A, lora_B, new_weight, error_f_norm, freeze_rank, freeze_alpha_single = init_lora_qera_kbit_mxint( 
                weight=module.weight,
                scale=scale,
                num_bits=num_bits,
                reduced_rank=reduced_rank,
                compute_device=(
                    module.weight.device if compute_device is None else compute_device
                ),
                block_size=mxint_block_size,
                init_method=init_method, 
                hess=hess,
                qera_num_iter=qera_num_iter,
                layer_name=layer_name, 
            )
            if freeze_alpha_single:
                for k, v in freeze_alpha_single.items():
                    freeze_alpha_dict[k] = v

        error_dict[name] = [error_f_norm]
        freeze_rank_dict[name] = freeze_rank

        module.lora_A[adapter_name].weight.data = lora_A
        module.lora_B[adapter_name].weight.data = lora_B / lora_layer_scaling
        module.weight.data = new_weight
    
    return error_dict, freeze_rank_dict, freeze_alpha_dict 




######################qlora####################################
@torch.no_grad()
def init_lora_qlora_2bit_bnb(weight: torch.Tensor, bnb_quant_type: str, compute_device, init_method: str = None, hess=None):
    compute_device = "cuda" if compute_device is None else compute_device
    assert compute_device != "cpu"
    ori_weight_dtype = weight.dtype
    ori_weight = weight.clone().to(device=compute_device, dtype=torch.float32)
    weight = weight.to(device=compute_device, dtype=torch.float32)

    quantizer = NFQuantizer(
        num_bits=2, device=compute_device, method=bnb_quant_type, block_size=64
    )

    qweight, max_abs, shape = quantizer.quantize_block(weight)
    dequantized_weight = quantizer.dequantize_block(qweight, max_abs, shape)

    error_f_norm = [
        torch.linalg.norm(ori_weight - dequantized_weight, ord="fro").cpu().item()
    ]

    dequantized_weight = dequantized_weight.to(ori_weight_dtype)
    return dequantized_weight, error_f_norm


@torch.no_grad()
def init_lora_qlora_kbit_mxint(
    weight: torch.Tensor, num_bits: int, compute_device, block_size, init_method: str = None, hess=None
):
    compute_device = "cuda" if compute_device is None else compute_device
    assert compute_device != "cpu"
    ori_weight_dtype = weight.dtype
    ori_weight = weight.clone().to(device=compute_device, dtype=torch.float32)
    weight = weight.to(device=compute_device, dtype=torch.float32)

    quantized_weight = mxint_quantizer(
        weight, width=num_bits, block_size=block_size, block_axis=-1
    )

    error_f_norm = [
        torch.linalg.norm(ori_weight - quantized_weight, ord="fro").cpu().item()
    ]

    quantized_weight = quantized_weight.to(ori_weight_dtype)
    return quantized_weight, error_f_norm


@torch.no_grad()
def replace_lora_weight_qlora_kbit(
    peft_model, quant_type, num_bits: int, compute_device=None, mxint_block_size=32, init_method: str = None, hess_dict=None
):
    assert quant_type in ["nf", "fp", "mxint"]
    if quant_type in ["nf", "fp"]:
        assert num_bits == 2
    prefix = "base_model.model."

    named_modules = {name: module for name, module in peft_model.named_modules()}
    error_dict = {}

    for name, module in tqdm.tqdm(
        named_modules.items(),
        desc=f"Replacing LoRA adapters (Emulated {num_bits}-bit qLoRA)",
    ):
        if not isinstance(module, LoraLinear):
            continue
        if not name.startswith(prefix):
            raise TypeError(f"Not peft model")

        name = name[len(prefix) :]

        if quant_type in ["nf", "fp"]:
            new_qweight, error_f_norm = init_lora_qlora_2bit_bnb(
                weight=module.weight,
                bnb_quant_type="normal" if quant_type == "nf" else "uniform",
                compute_device=(
                    module.weight.device if compute_device is None else compute_device
                ),
                init_method=init_method,
                hess=None,
            )
        else:
            new_qweight, error_f_norm = init_lora_qlora_kbit_mxint(
                weight=module.weight,
                num_bits=num_bits,
                compute_device=(
                    module.weight.device if compute_device is None else compute_device
                ),
                block_size=mxint_block_size,
                init_method=init_method,
                hess=None,
            )
        error_dict[name] = error_f_norm

        module.weight.data = new_qweight

    return error_dict

import hf_patch
from peft import get_peft_model, LoraConfig, AdaLoraConfig, TaskType
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from utils import (
    train_text_to_text_model,
    model_inference,
    initialize_text_to_text_model,
    transform_dataset,
    merge_llama,
)
import json
import math
from datasets import load_dataset
import wandb
from data import *
from typing import List
import torch
from copy import deepcopy
import logging
from tqdm import tqdm, trange
from typing import Tuple, List, Dict
from peft.tuners.lora.layer import Linear as LoraLinear
from split import rebuild
import re
import itertools
import matplotlib.pyplot as plt
try:
    from commonsense_evaluate import common_evaluate
except ImportError:
    common_evaluate = None
try:
    from eval_humaneval import humaneval
except ImportError:
    humaneval = None
# from eval_mtbench import evaluate_mtbench_from_model
log = logging.getLogger(__name__)

s = 0

def kron(A, B):
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

def modified_gram_schmidt(W, eps=1e-12):
    """
    Modified Gram–Schmidt QR
    W: (m, n)
    Returns:
        Q: (m, n)
        R: (n, n)
    """
    m, n = W.shape
    Q = W.clone()
    R = torch.zeros(n, n, device=W.device, dtype=W.dtype)

    for i in range(n):
        R[i, i] = torch.norm(Q[:, i])
        if R[i, i] < eps:
            raise RuntimeError("Linearly dependent columns")

        Q[:, i] = Q[:, i] / R[i, i]

        for j in range(i + 1, n):
            R[i, j] = torch.dot(Q[:, i], Q[:, j])
            Q[:, j] = Q[:, j] - R[i, j] * Q[:, i]

    return Q, R

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def find_all_linear_modules(model) -> List[str]:
    r"""
    Finds all available modules to apply lora.
    """
    linear_cls = torch.nn.Linear

    output_layer_names = ["lm_head", "embed_tokens"]

    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls) and not any(
            [output_layer in name for output_layer in output_layer_names]
        ):
            module_names.add(name.split(".")[-1])
    return list(module_names)


def find_hidden_state_size(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            return min(module.weight.shape)
    return None


def calculate_gain(
    nonlinearity, param
) -> float:
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (
            not isinstance(param, bool)
            and isinstance(param, int)
            or isinstance(param, float)
        ):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return (
            3.0 / 4
        )  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")

def kaimings(weight, a=math.sqrt(5), fan=4096):
    nonlinearity = "leaky_relu"
    generator = None
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return weight.uniform_(-bound, bound, generator=generator)

@torch.no_grad()
def reinit_lora_modules(name, module, init_config, **kwargs):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    lora_r1 = kwargs["lora_r1"]
    lora_r2 = kwargs["lora_r2"]
    lora_r = kwargs["lora_r"]
    # lora_r1 = min(module.lora_A.default.weight.shape)
    # lora_r2 = min(module.lora_B.default.weight.shape)
    a_dim = max(module.lora_A.default.weight.shape)
    b_dim = max(module.lora_B.default.weight.shape)
    if init_config.mode == "simple":
        match init_config.lora_A:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_A.default.weight, mean=0.0, std=init_config.lora_A_std
                )
            case "kaiming":
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                torch.nn.init.kaiming_uniform_(module.lora_A.default.weight, a=math.sqrt(5))
            case "kaimings":
                kaimings(module.lora_A.default.weight, a=math.sqrt(5), fan=module.weight.size(1))
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_A.default.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_A.default.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_A.default.weight)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_A.default.weight, mean=0.0, std=1.0 / (a_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_A.default.weight)
            case _:
                raise ValueError(f"Unknown lora_A initialization: {init_config.lora_A}")
        match init_config.lora_B:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_B.default.weight, mean=0.0, std=init_config.lora_B_std
                )
            case "kaiming":
                torch.nn.init.kaiming_normal_(module.lora_B.default.weight.T, a=math.sqrt(5))
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_B.default.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_B.default.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_B.default.weight)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_B.default.weight, mean=0.0, std=1.0 / (b_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_B.default.weight)
            case _:
                raise ValueError(f"Unknown lora_B initialization: {init_config.lora_B}")
        if init_config.get("scale", "") == "stable":
            gamma = init_config.stable_gamma
            #module.lora_B.default.weight.data *= (m**0.25) / gamma**0.5
            #module.lora_A.default.weight.data *= (n**0.25) / gamma**0.5
            #module.lora_B.default.weight.data *= (m**0.25)
            #module.lora_A.default.weight.data *= (n**0.25)
            module.lora_B.default.weight.data *= 1
            module.lora_A.default.weight.data *= 1


    elif init_config.mode == "svd":
        U, S, V = torch.svd_lowrank(module.weight.float(), q=4 * lora_r, niter=4)
        V = V.T
        m, n = module.weight.shape
        if init_config.scale == "default":
            S = S / module.scaling["default"]
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])).T.contiguous()
            )
        elif init_config.scale == "stable":
            gamma = init_config.stable_gamma
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * (m**0.25) / gamma**0.5).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :] * (n**0.25) / gamma**0.5).contiguous()
            )
        elif init_config.scale == "unit":
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r]).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :]).contiguous()
            )
        elif init_config.scale == "normalized":
            S_sum = S[:lora_r].sum()
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])/torch.sqrt(S_sum)*lora_r**0.5).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])/torch.sqrt(S_sum)*lora_r**0.5).T.contiguous()
            )

    elif init_config.mode == "qr":
        W = module.weight.float()
        k,d = W.shape
        Q, R = torch.linalg.qr(W, mode="reduced")
        diag = torch.sign(torch.diag(R))
        diag[diag == 0] = 1.0

        D = torch.diag(diag)

        Q = Q @ D
        R = D @ R
        print(torch.min(torch.diag(R)))
        lambda_vals = torch.abs(torch.diag(R))
        perm = torch.argsort(lambda_vals, descending=True)

        I1 = perm[:lora_r2] 
        I2 = perm[lora_r2:lora_r1+lora_r2]  
        Q1 = Q[:, I1]          # (m, r_high)
        R1 = R[I1]
        Q2 = Q[:, I2]
        R2 = R[I2]
        B = Q1[:k // lora_r1] @ R1[:, :lora_r2]      
        A = (Q2[:d // lora_r2] @ R2[:, :lora_r1]).T
        module.lora_B.default.weight = torch.nn.Parameter(B.contiguous().to(module.lora_B.default.weight.device))
        module.lora_A.default.weight = torch.nn.Parameter(A.contiguous().to(module.lora_A.default.weight.device))

    elif init_config.mode == "gradient":
        named_grad = kwargs["named_grads"]
        grad_name = ".".join(name.split(".")[2:]) + ".weight"
        grads = named_grad[grad_name]
        # print(grads.shape)
        if lora_r1 == 1 and lora_r2 == 1:
            U, S, V = torch.svd_lowrank(-grads.cuda().float(), q=512, niter=16)
        else:
            U, S, V = torch.svd_lowrank(rebuild(-grads.float(),lora_r1, lora_r2), q=4*lora_r, niter=16)
        V = V.T
        # set direction
        if init_config.direction == "ArBr":
            if lora_r1 == 1 and lora_r2 == 1:
                B = U[:, :lora_r] @ torch.diag(torch.sqrt(S[:lora_r])) / torch.sqrt(S[0]) / 128.0 **0.5
                A = torch.diag(torch.sqrt(S[:lora_r])) @ V[:lora_r, :] / torch.sqrt(S[0]) / 128.0 **0.5
                module.lora_B.default.weight = torch.nn.Parameter(B.contiguous().to(module.lora_B.default.weight.device))
                module.lora_A.default.weight = torch.nn.Parameter(A.contiguous().to(module.lora_A.default.weight.device))
            else:
                for i in range(lora_r):
                    B = (S[i] / S[0] / 1024)**0.5 * V[i, :].reshape([lora_r2, grads.shape[0]//lora_r1]).T
                    A = (S[i] / S[0] / 1024)**0.5 * U[:, i].reshape([grads.shape[1]//lora_r2,lora_r1]).T
                    module.lora_A.default.weight[i::lora_r] = torch.nn.Parameter(A.contiguous().to(module.lora_A.default.weight.device))
                    module.lora_B.default.weight[:,i::lora_r] = torch.nn.Parameter(B.contiguous().to(module.lora_B.default.weight.device))
        elif init_config.direction == "A2rBr":
            B = U[:, :lora_r]
            A = V[lora_r : 2 * lora_r, :]
        elif init_config.direction == "ArB2r":
            B = U[:, lora_r : 2 * lora_r]
            A = V[:lora_r, :]
        scaling_factor = module.scaling["default"]
        if init_config.scale == "gd":
            A = A / scaling_factor
            B = B / scaling_factor
        elif init_config.scale == "unit":
            # Because A,B is orthogonal, do not need to scale
            pass
        elif init_config.scale == "stable":
            m, n = grads.shape # m: feature_out, n: feature_in
            # the scale of output is only related to the feature_out
            gamma = init_config.stable_gamma


        elif init_config.scale == "weightS":
            _, S, _ = torch.svd_lowrank(module.weight.float(), q=4 * lora_r, niter=4)
            S = S / module.scaling["default"]
            avg_s = torch.sqrt(S[:lora_r]).mean().to(A.device)
            B = B * avg_s
            A = A * avg_s
        # module.lora_B.default.weight = torch.nn.Parameter(B.contiguous().to(module.lora_B.default.weight.device))
        # module.lora_A.default.weight = torch.nn.Parameter(A.contiguous().to(module.lora_A.default.weight.device))

    with torch.no_grad():
        # consider dtype not in init_config
        if "dtype" not in init_config:
            pass
        elif init_config.dtype == "bf16":
            module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
                torch.bfloat16
            )
            module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
                torch.bfloat16
            )
        elif init_config.dtype == "fp32":
            module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
                torch.float32
            )
            module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
                torch.float32
            )
        # If lora_A@lora_B is not zero, then we need to subtract lora_A@lora_B from the original weight matrix
        if init_config.mode == "qr":
            offset = (kron(module.lora_B.default.weight.contiguous(),module.lora_A.default.weight.contiguous())).to(
            module.weight.data.device
        )
        else:
            offset = 0
        # offset = (module.lora_B.default.weight @ module.lora_A.default.weight).to(
        #     module.weight.data.device
        # )

        scaling_factor = module.scaling["default"]
        offset *= scaling_factor
        if "norm_clip" in init_config and init_config.norm_clip:
            # for numerical stability, offset's largest value must be less then weight's largest value
            ratio = torch.max(torch.abs(module.weight.data)) / torch.max(
                torch.abs(offset)
            )
            if ratio < 1:
                offset *= ratio
                module.lora_A.default.weight.data *= ratio**0.5
                module.lora_B.default.weight.data *= ratio**0.5
                log.warning(f"Clipping offset by {ratio}")
        try:
            module.weight.data -= offset
        except:
            breakpoint()


def reinit_lora(model, init_config, **kwargs):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    for name, module in tqdm(
        model.named_modules(),
        desc="Reinitializing Lora",
        total=len(list(model.named_modules())),
    ):
        if isinstance(module, LoraLinear):
            reinit_lora_modules(name, module, init_config, **kwargs)

    return model


def get_record_gradient_hook(model, record_dict):
    def record_gradient_hook(grad):
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n not in record_dict:
                    record_dict[n] = p.grad.cpu()
                else:
                    record_dict[n] += p.grad.cpu()
                p.grad = None
        return grad

    return record_gradient_hook


def estimate_gradient(
    model, dataset, batch_size: int = 4
) -> Dict[str, List[torch.Tensor]]:
    r"""
    Estimate the gradient of the model on the given dataset
    """
    log.info("Estimating gradient")
    model.train()
    named_grads = {}
    hooks = []
    for name, param in model.named_parameters():
        hook = param.register_hook(get_record_gradient_hook(model, named_grads))
        hooks.append(hook)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    num = 0
    for batch in tqdm(dataloader, desc="Estimating gradient"):
        num += 1
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        outputs.loss.backward()
        get_record_gradient_hook(model, named_grads)(None)  # get gradient of last layer
        # make sure the gradient is cleared
        for n, p in model.named_parameters():
            if p.grad is not None:
                p.grad = None
    for n, g in named_grads.items():
        named_grads[n] /= num
    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    return named_grads






def extract_num(text):
    # Regex pattern to find the number following '####'
    pattern = r'####\s*(\d+)'
    # Using re.search to find the first match
    match = re.search(pattern, text)
    if match:
        result = match.group(1)
        print(text)
    else:
        print(text)
        result = ""
    try:
        return int(result.replace(",", ""))
    except:
        print(f"'{result}' can't be converted")
        return 0


def eval_gsm8k(model,tokenizer,model_type, test_set):
    all = 0
    correct = 0
    t = tqdm(test_set)
    for example in t:
        # print(example['x'])
        pred_text = model_inference(model, tokenizer, example['x'], model_type, max_target_length=512)
        gt = extract_num(example["y"])
        pred = extract_num(pred_text)
        correct += int(gt == pred)
        all += 1
        t.set_description(f"Accuracy: {correct / all * 100:02f}%")

    print("Acc:", correct / all)
    # append to gsm8k_results.txt (create if not exists)
    if not os.path.exists("gsm8k_results.txt"):
        with open("gsm8k_results.txt", "w") as f:
            f.write("Model Acc\n")
    with open("gsm8k_results.txt", "a") as f:
        f.write(f"{model_name} {correct / all}\n")

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_exp(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    model_name = cfg.model.name
    model_type = cfg.model.type
    dataset_name = cfg.dataset_name
    dataset_func = DATASET_MAP[dataset_name]
    use_peft = cfg.peft.use_peft
    if_use_rslora = cfg.peft.use_rslora
    lora_r = cfg.peft.lora_r
    lora_r1 = cfg.peft.lora_r1
    lora_r2 = cfg.peft.lora_r2
    lora_relative_r = cfg.peft.lora_relative_r
    lora_target_modules = cfg.peft.lora_target_modules
    train_embeddings = cfg.peft.train_embeddings
    if cfg.dry_run:
        return
    if use_peft:
        lora_r = cfg.peft.lora_r
        lora_r1 = cfg.peft.lora_r1
        lora_r2 = cfg.peft.lora_r2
        lora_alpha = cfg.peft.lora_alpha
        lora_relative_r = None
        init = cfg.init.mode
    else:
        lora_r = None
        lora_target_modules = None
        lora_relative_r = None
        train_embeddings = True
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "use_peft": use_peft,
        "lora_r1": lora_r1,
        "lora_r2": lora_r2,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "init": init,
        "lora_target_modules": str(lora_target_modules),
        "lora_relative_r": lora_relative_r,
        "train_embeddings": train_embeddings,
    }
    if cfg.wandb.name:
        name = cfg.wandb.name
    else:
        name = "_".join([f"{k}={v}" for k, v in config.items()])
    cfg.wandb.project += "_" + cfg.dataset_name
    wandb.init(
        project=cfg.wandb.project,
        name=name,
        config=config,
    )
    train_set, val_set, eval_set = dataset_func()
    model, tokenizer = initialize_text_to_text_model(
        model_name, model_type, cfg.model.bf16, cfg.peft.use_peft, flash_attention=True
    )
    additional_kwargs = {}
    if use_peft and cfg.init.mode == "gradient":
        if isinstance(train_set, list):
            temp_set = train_set[: cfg.init.bsz * cfg.init.iters]
        else:
            temp_set = train_set.select(range(cfg.init.bsz * cfg.init.iters))
        transform_dataset(
            model_type=model_type,
            dataset=temp_set,
            tokenizer=tokenizer,
            max_length=cfg.init.max_length,
        )
        # named_grads = estimate_layer_inputs(model, temp_set, cfg.init.bsz)
        named_grads = estimate_gradient(model, temp_set, cfg.init.bsz)
        additional_kwargs["named_grads"] = named_grads
        
    additional_kwargs["lora_r1"] = lora_r1
    additional_kwargs["lora_r"] = lora_r
    additional_kwargs["lora_r2"] = lora_r2

    if lora_target_modules == "all":
        lora_target_modules = find_all_linear_modules(model)
    else:
        lora_target_modules = list(lora_target_modules) if lora_target_modules else []
    if lora_relative_r is not None:
        hidden_size = find_hidden_state_size(model)
        lora_r = int(hidden_size * lora_relative_r)
        log.info(f"lora_r is set to {hidden_size} * {lora_relative_r} = {lora_r}")
    if use_peft and cfg.peft.get("dora", False):
        log.info("Using Dora")
        peft_config = LoraConfig(
            r1=lora_r1,
            r2=lora_r2,
            lora_alpha=cfg.peft.lora_alpha,
            target_modules=lora_target_modules,
            use_rslora=if_use_rslora,
            use_dora=True,
        )
        orig_model_params = sum(p.numel() for p in model.parameters())
        model = get_peft_model(model, peft_config)
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
    elif use_peft and cfg.peft.get("adalora", False):
        log.info("Using AdaLora")
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_r=lora_r,
            lora_alpha=cfg.peft.lora_alpha,
            target_modules=lora_target_modules,
            total_step=int(len(train_set)/cfg.model.real_batch_size)*cfg.model.epochs,
        )
        orig_model_params = sum(p.numel() for p in model.parameters())
        model = get_peft_model(model, peft_config)
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
    elif use_peft:
        peft_config = LoraConfig(
            r1=lora_r1,
            r2=lora_r2,
            r= lora_r,
            lora_alpha=cfg.peft.lora_alpha,
            target_modules=lora_target_modules,
            use_rslora=if_use_rslora,
        )
        orig_model_params = sum(p.numel() for p in model.parameters())
        model = get_peft_model(model, peft_config)
        reinit_lora(model, cfg.init, **additional_kwargs)
        if train_embeddings:
            model.lm_head.weight.requires_grad = True
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
        save_dir = os.path.join(
            "results", f"{cfg.wandb.project}/{name}/{cfg.seed}", "orig_checkpoint"
        )
        model.save_pretrained(save_dir)
        adapter_config = json.load(open(os.path.join(save_dir, "adapter_config.json")))
        adapter_config["lora_alpha"] = -adapter_config["lora_alpha"]
        json.dump(
            adapter_config, open(os.path.join(save_dir, "adapter_config.json"), "w")
        )
    else:
        # full finetune
        all_param = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rate = {
            "trainable_params": trainable_params,
            "orig_params": all_param,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": 1,
        }
    log.info(rate)
    wandb.summary.update(rate)
    training_loop = train_text_to_text_model
    global s
    print(s)
    
    model, eval_results = training_loop(
        f"{cfg.wandb.project}/{name}",
        train_set,
        val_set,
        model,
        tokenizer,
        model_type,
        num_train_epochs=cfg.model.epochs,
        per_device_batch_size=cfg.model.per_device_batch_size,
        real_batch_size=cfg.model.real_batch_size,
        bf16=cfg.model.bf16,
        eval_epochs=cfg.model.eval_epochs,
        early_stopping_patience=cfg.model.early_stopping_patience,
        max_length=cfg.model.max_length,
        logging_steps=cfg.model.logging_steps,
        use_loraplus=cfg.peft.use_loraplus,
        loraplus_lr_ratio=cfg.peft.loraplus_lr_ratio,
        learning_rate=cfg.model.learning_rate,
        # deepspeed=(
        #     "z3_offload_all_bf16.json" if cfg.peft == False else None
        # ),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        seed=cfg.seed,
    )
    wandb.summary.update(eval_results)



    save_dir = os.path.join(
        "results", f"{cfg.wandb.project}/{name}/{cfg.seed}"
    )
    if not use_peft:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    else:
        # merge_llama(os.path.join("results", f"{cfg.wandb.project}/{name}/{cfg.seed}"))
        pass
    log.info(f"Saving model to {save_dir}")
    if dataset_name == 'meta_math':
        train_set, val_set, eval_set = load_gsm8k()
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        eval_gsm8k(model,tokenizer,model_type,eval_set)
    if dataset_name == 'codefeedback':
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        humaneval(model,tokenizer,save_dir, model_type)
    wandb.finish()


if __name__ == "__main__":
    run_exp()

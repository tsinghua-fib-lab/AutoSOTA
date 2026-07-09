from argparse import ArgumentParser
from pathlib import Path
import logging
import yaml
from pprint import pformat
import shutil
import time

import torch
from torch.utils.data import DataLoader
import transformers
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from peft import TaskType
from accelerate import dispatch_model
import gzip

import bitsandbytes as bnb

from .models import (
    find_layers_to_approximate,
    quantize_model,
    find_layers_to_register_scale_hook,
)
from .statistic_profiler import register_scale_hooks, share_scales

from qera.datasets import get_data_module_for_peft
from qera.models import find_layers_to_register_scale_hook
from qera.statistic_profiler import register_scale_hooks, share_scales
from qera.evaluate import evaluate_perplexity
from qera.fine_tuning import (
    replace_lora_weights_loftq_4bit,
    replace_lora_weights_qera_4bit,
    replace_lora_weights_loftq_kbit,
    replace_lora_weight_qlora_kbit,
    replace_lora_weight_qera_kbit,
)
from qera.utils import create_device_map
from qera.approximate_with_init import get_lr_initializer
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _remove_hess(qera_config, hess_dict):
    '''
    remove hessian from working files.
    '''
    pass


def hess_checker(config):
    '''
    This function checks if Hessian (2x.Tx) is used in quantization process.
    if yes, it will return True, else False.
    '''
    init_method = config['init_method']
    if init_method in ['odlri', 'resq_x', 'resq_wx']:
        return True
    else:
        return False

def hess_loader(config): 
    '''
    This function loads the Hessian (2x.Tx) from the model.
    If hess is not used, it will return None.
    '''
    
    model_name = config['model_name'].replace('/', '_')
    cal_set = config['qera_calibration_set']
    cal_num = config['qera_num_calibration_samples']

    glob_dir = config['global_dir']
    hess_path = f'qpeft_hessian_{model_name}___{cal_set}_{cal_num}.pt'
    total_path = Path(os.path.join(glob_dir, hess_path))
    hess_dict = {}
    if total_path.is_file():
        inter_dict = torch.load(total_path)
        for k, v in inter_dict.items():
            N, diag_val, upper_val = v['N'], v['diag_val'], v['upper_val']
            recon = torch.zeros((N, N), device='cpu')
            idcs = torch.arange(N, device='cpu')
            r_idx, c_idx = torch.triu_indices(N, N, offset=1)
            recon[idcs, idcs] = diag_val
            recon[r_idx, c_idx] = upper_val
            recon[c_idx, r_idx] = upper_val
            hess_dict[k] = recon
    else:
        config['hess_path'] = total_path
    
    return hess_dict

def generate_save_load_hessian(model, tokenizer, data_collator, **kwargs):
    
    hess_path = kwargs.get('hess_path')
    is_clm = kwargs.get('mode') == 'clm'
    
    
    layers_to_register_and_share = find_layers_to_register_scale_hook(model)
    profiler_factory = register_scale_hooks(
            model,
            layers_to_register_and_share,
            mode="hess",
            torch_dtype=torch.float32,
        )
    
    calibration_datamodule = get_data_module_for_peft(
                kwargs.get('qera_calibration_set'),
                tokenizer=tokenizer,
                model_config=None if is_clm else model.config, 
                pad_to_max_length=None if is_clm else kwargs.get("PAD_TO_MAX_LENGTH"),
                max_length=kwargs.get('qera_max_seq_length'),
                num_workers=kwargs.get('num_workers'),
                overwrite_cache=kwargs.get('overwrite_dataset_cache'),
                )

    calibration_dataloader = DataLoader(
            calibration_datamodule["train"],
            batch_size=kwargs.get('qera_calibration_batch_size'),
            shuffle=False,
            num_workers=kwargs.get('num_workers'),
            collate_fn=data_collator,
        )
    
    if is_clm:
         _ = evaluate_perplexity(
            model,
            eval_dataloader=calibration_dataloader,
            num_samples=kwargs.get("qera_num_calibration_samples"),
            progress_bar=True,
            description="Generating Hessian",
        )
    else:
        num_samples = 0
        input_device = next(model.parameters()).device
        for i, batch in enumerate(calibration_dataloader):
            with torch.no_grad():
                batch = {
                    k: v.to(input_device) for k, v in batch.items() if k != "labels"
                }
                _ = model(**batch, output_hidden_states=True)
            num_samples += kwargs.get('qera_calibration_batch_size')
            if num_samples >= kwargs.get("qera_num_calibration_samples"):
                break

    profiler_factory.remove_all_hooks()
    hess_dict = profiler_factory.get_scale_dict(progress_bar=True)
    
    f32_hess_dict = {}
    inter_dict = {}
    for k, v in tqdm(hess_dict.items(), desc="Saving Hessians"):
        N = v.shape[0]
        diag_val = v.diag().float()
        r_idx, c_idx = torch.triu_indices(N, N, offset=1)
        upper_val = v[r_idx, c_idx].float()
        
        inter_dict[k] = {
            'N': N,
            'diag_val': diag_val,
            'upper_val': upper_val
        }
        f32_hess_dict[k] = v.float()

    torch.save(inter_dict, hess_path)
    logger.info(f"Hessian successfully saved in {hess_path}")

    return f32_hess_dict



def save_scale_dict(scale_dict, config):
    save_path = config['scale_path']
    scale_mode = config['qera_scaling_mode']
    
    if scale_mode in ['rxx']:
        inter_dict = {}
        for k, v in scale_dict.items():
            N = v.shape[0]
            diag_val = v.diag() 
            r_idx, c_idx = torch.triu_indices(N, N, offset=1)
            upper_val = v[r_idx, c_idx]
            
            inter_dict[k] = {
                'N': N,
                'diag_val': diag_val,
                'upper_val': upper_val
            }

        torch.save(inter_dict, save_path)
    
    else: 
        for k, v in scale_dict.items():
            scale_dict[k] = v.cpu()
        torch.save(scale_dict, save_path)
        
    logger.info(f"✅ Precomputed scale dict saved to {save_path}")



def get_precomputed_scale_dict(config, logger):
    '''
    function that returns the precomputed scale dict.
    if not found, it will return empty dict
    '''
    scale_dict = {}
    scale_mode = config['qera_scaling_mode'] if config['adapter_init'] == 'qera' else ""
    match scale_mode:
        case 'diag' | 'rxx':
            glob_dir = config['global_dir']
            model_name, cal_set, num_cal_set = config['model_name'].replace('/', '_'), \
                                                config['qera_calibration_set'], config['qera_num_calibration_samples']
            f_name = f'qpeft_{scale_mode}_{model_name}___{cal_set}_{num_cal_set}.pt'
            f_pth = Path(os.path.join(glob_dir, f_name))
            if f_pth.is_file():
                logger.info(f"✅ Precomputed scale dict found. Skipping data calibration...")
                if scale_mode == 'diag':
                    scale_dict = torch.load(f_pth)
                else:
                    inter_dict = torch.load(f_pth)
                    for k, v in inter_dict.items():
                        N, diag_val, upper_val = v['N'], v['diag_val'], v['upper_val']
                        recon = torch.zeros((N, N), device='cpu')
                        idcs = torch.arange(N, device='cpu')
                        r_idx, c_idx = torch.triu_indices(N, N, offset=1)
                        
                        diag_val = diag_val.to(recon.device)
                        upper_val = upper_val.to(recon.device)
                        
                        recon[idcs, idcs] = diag_val
                        recon[r_idx, c_idx] = upper_val
                        recon[c_idx, r_idx] = upper_val
                        scale_dict[k] = recon
            else:
                config['scale_path'] = f_pth
                logger.info(f"❌ No precomputed scale dict found for {scale_mode}. Running calibration")
        
        case _:
            logger.info(f"❌ No precomputed scale dict needed. Running calibration")
    
    return scale_dict

def dequantize_qera_params(model: torch.nn.Module):
    for module in model.modules():
        w = getattr(module, "weight", None)
        if w is None or not hasattr(w, "quant_state"):
            continue

        q_data  = w.data
        q_state = w.quant_state

        orig_shape = getattr(q_state, "orig_shape", None)
        if orig_shape is None:
            raise RuntimeError(...)

        orig_dtype = getattr(q_state, "orig_dtype", torch.float32)
        flat = bnb.functional.dequantize_4bit(q_data, q_state)
        dequantized_weight = (
            flat.view(orig_shape)
                .to(device=w.device)
                .to(orig_dtype))

        module.weight.data = dequantized_weight
        delattr(module.weight, "quant_state")
        

def make_weights_contiguous(model):
    for name, param in model.named_parameters():
        if isinstance(param.data, torch.Tensor) and not param.data.is_contiguous():
            param.data = param.data.contiguous()

def adapt_and_save_clm_model(
    model_name_or_path: str,
    adapter_init: str,
    output_dir: str,
    qera_calibration_set: str,
    qera_num_calibration_samples: int,
    qera_calibration_batch_size: int,
    qera_max_seq_length: int,
    loftq_num_iters: int,
    quant_type: str,
    quant_bits: int,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_target_modules: list[str] | None,
    device_map: str,
    num_workers: int,
    overwrite_output_dir: bool,
    overwrite_dataset_cache: bool,
    qera_scaling_mode: str,
    peek_post_init_metrics: bool,
    lora_modules_to_save: list[str] | None,  # lm_head will not be quantized
    mxint_block_size: int,
    init_method: str,
    config: dict,
    qera_num_iter: int,
):
    """
    Apply Lora or qLoRA to a causal language model and save the base model & adapted model to disk.
    """
    assert adapter_init in ["loftq", "qera", "qlora", "lora"]
    assert qera_scaling_mode in ["diag", "rxx"]
    assert quant_type in ["nf", "fp", "mxint"]
    if quant_type in ["nf", "fp"]:
        assert quant_bits in [2, 4]


    if lora_target_modules is None:
        lora_target_modules = "all-linear"
        logger.warning(
            f" ⚠️ Defaulting lora_target_modules to {lora_target_modules}, which automatically selects all linear layers except for lm_head"
        )
    if lora_modules_to_save is None:
        logger.warning(
            f" ⚠️ Defaulting lora_modules_to_save to 'None'. LM head will not be quantized."
        )

    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite_output_dir:
            raise FileExistsError(f"Output directory {output_dir} already exists")
        else:
            logger.warning(
                f"⚠️ Output directory {output_dir} already exists and will be overwritten"
            )
            shutil.rmtree(output_dir, ignore_errors=True)

    scale_dict = None
    calibration_dataloader = None
    calibration_time = 0.0
    
    hess_dict = None
    need_hess = hess_checker(config)

    if need_hess:
        hess_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        if hess_tokenizer.pad_token_id is None:
            hess_tokenizer.pad_token = hess_tokenizer.eos_token
        
        hess_data_collator = transformers.default_data_collator
        hess_model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, _attn_implementation="eager")
        
        hess_model.eval()
        if "cuda" in device_map:
            hess_model.to(device_map)
        else:
            if hasattr(hess_model, "tie_weights"):
                hess_model.tie_weights()
            map_dict = create_device_map(hess_model, device_map)
            hess_model = dispatch_model(hess_model, map_dict)
            
        layers_to_register_and_share = find_layers_to_register_scale_hook(hess_model)
        hess_dict = hess_loader(config)
        
        if hess_dict: 
            logger.info(f"✅ Hessian loaded successfully") 
        else:
            logger.info(f"🔊 Hessian required, but not found. Generating...")
            hess_dict = generate_save_load_hessian(hess_model, hess_tokenizer, hess_data_collator,
                                                   qera_calibration_set=qera_calibration_set,
                                                   qera_max_seq_length=qera_max_seq_length,
                                                   num_workers=num_workers,
                                                   overwrite_dataset_cache=overwrite_dataset_cache,
                                                   qera_calibration_batch_size=qera_calibration_batch_size,
                                                   qera_num_calibration_samples=qera_num_calibration_samples,
                                                   hess_path=config['hess_path'],
                                                   mode='clm'
                                                   )
        del hess_tokenizer
        del hess_model
        del hess_data_collator
        
        share_scales(hess_dict, layers_to_register_and_share)
    else:
        logger.info(f"❌ Hessian not used in whole script")  
    
    
    if adapter_init == "qera" or (peek_post_init_metrics and adapter_init != "lora"):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # load the model for calibration
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path, _attn_implementation="eager"
        )
        model.eval()
        if "cuda" in device_map:
            model.to(device_map)
        else:
            if hasattr(model, "tie_weights"):
                model.tie_weights()
            map_dict = create_device_map(model, device_map)
            model = dispatch_model(model, map_dict)
                
        data_collator = transformers.default_data_collator

        scale_dict = get_precomputed_scale_dict(config, logger)
        if not scale_dict:
            if adapter_init == "qera":
                layers_to_register_and_share = find_layers_to_register_scale_hook(model)
                profiler_factory = register_scale_hooks(
                    model,
                    layers_to_register_and_share,
                    mode=qera_scaling_mode,
                    torch_dtype=torch.float32,
                )
            calibration_datamodule = get_data_module_for_peft(
                qera_calibration_set,
                tokenizer=tokenizer,
                model_config=None,
                pad_to_max_length=None,
                max_length=qera_max_seq_length,
                num_workers=num_workers,
                overwrite_cache=overwrite_dataset_cache,
            )
            calibration_dataloader = DataLoader(
                calibration_datamodule["train"],
                batch_size=qera_calibration_batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=data_collator,
            )
            
            start = time.time()
            profile_outputs = evaluate_perplexity(
                model,
                eval_dataloader=calibration_dataloader,
                num_samples=qera_num_calibration_samples,
                progress_bar=True,
                description="Pretrained model profiling",
            )
            logger.info(f"FP32 outputs:\n{pformat(profile_outputs, sort_dicts=False)}")

            if adapter_init == "qera":
                profiler_factory.remove_all_hooks()
                scale_dict = profiler_factory.get_scale_dict(progress_bar=True)
                share_scales(scale_dict, layers_to_register_and_share)
                
                if qera_scaling_mode in ['diag', 'rxx']:
                    save_scale_dict(scale_dict, config)
                
            calibration_time = time.time() - start
        else: 
            layers_to_register_and_share = find_layers_to_register_scale_hook(model)
            share_scales(scale_dict, layers_to_register_and_share)
            

    bnb_config = None
    use_4bit_bnb = (
        adapter_init in ["qlora", "loftq"]
        and quant_type in ["nf", "fp"]
        and quant_bits == 4
    )
    
    if use_4bit_bnb:
        logger.info("4-bit BnB detected -> skipping device_map logic (multi-GPU) to avoid .to() error.")
        bnb_4bit_use_double_quant = (quant_bits == 4)
        bnb_quant_type_4bit = "nf4" if quant_type == "nf" else "fp4"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb_quant_type_4bit,
            bnb_4bit_quant_storage=torch.uint8,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config
        )
        model.eval()
        logger.info("Loaded 4-bit model on a single GPU (no dispatch_model).")

    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path
        )
        model.eval()
        
        if "cuda" in device_map:
            model.to(device_map)
        else:
            if hasattr(model, "tie_weights"):
                model.tie_weights()
            map_dict = create_device_map(model, device_map)
            model = dispatch_model(model, map_dict)
        
    lora_target_modules_ = lora_target_modules
    
    if isinstance(lora_target_modules, (list, tuple)) and len(lora_target_modules) == 1 and lora_target_modules[0] == "all-linear":
        lora_target_modules_ = "all-linear"

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules_,
        init_lora_weights=True,
        modules_to_save=lora_modules_to_save,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.eval()

    error_dict = None
    elapsed = None
    freeze_rank_dict = None 
    freeze_alpha_dict = None  
    
    if adapter_init == "loftq":
        if quant_bits == 4 and quant_type in ["nf", "fp"]:
            start = time.time()
            error_dict = replace_lora_weights_loftq_4bit(
                peft_model, num_iters=loftq_num_iters, init_method=init_method, hess_dict=hess_dict,   
            
            )
            elapsed = time.time() - start
        else:
            start = time.time()
            error_dict = replace_lora_weights_loftq_kbit(
                peft_model,
                quant_type=quant_type,
                num_bits=quant_bits,
                num_iters=loftq_num_iters,
                mxint_block_size=mxint_block_size,
                init_method=init_method,
                hess_dict=hess_dict,
            )
            elapsed = time.time() - start
    elif adapter_init == "qera":
        if quant_bits == 4 and quant_type in ["nf", "fp"]:
            start = time.time()
            error_dict = replace_lora_weights_qera_4bit(
                peft_model, scale_dict=scale_dict, init_method=init_method, hess_dict=hess_dict, qera_num_iter=qera_num_iter,
            )
            elapsed = time.time() - start + calibration_time
        else:
            start = time.time()
            error_dict, freeze_rank_dict, freeze_alpha_dict = replace_lora_weight_qera_kbit(
                peft_model,
                scale_dict=scale_dict,
                quant_type=quant_type,
                num_bits=quant_bits,
                mxint_block_size=mxint_block_size,
                init_method=init_method, 
                hess_dict=hess_dict,
                qera_num_iter=qera_num_iter,
            )
            elapsed = time.time() - start + calibration_time
    elif adapter_init == "qlora":
        if quant_bits == 4 and quant_type in ["nf", "fp"]:
            pass
        else:
            start = time.time()
            error_dict = replace_lora_weight_qlora_kbit(
                peft_model,
                quant_type=quant_type,
                num_bits=quant_bits,
                mxint_block_size=mxint_block_size,
                init_method=init_method, 
                hess_dict=hess_dict,   
            )
            elapsed = time.time() - start
    elif adapter_init == "lora":
        pass
    else:
        raise ValueError(f"Invalid adapter init: {adapter_init}")

    post_init_ppl = None
    if peek_post_init_metrics and adapter_init != "lora":
        peft_model.eval()
        post_init_ppl = evaluate_perplexity(
            peft_model,
            eval_dataloader=calibration_dataloader,
            num_samples=qera_num_calibration_samples,
            progress_bar=True,
            description=f"Evaluating post initialization ({adapter_init})",
        )
        logger.info(
            f"Post initialization perplexity ({adapter_init}):\n{pformat(post_init_ppl, sort_dicts=False)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(output_dir / "adapter")

    if freeze_rank_dict is not None:
        import json
        freeze_rank_path = output_dir / "adapter" / "freeze_rank_dict.json"  
        with open(freeze_rank_path, "w") as f: 
            json.dump(freeze_rank_dict, f)

    if freeze_alpha_dict is not None:
        import json
        freeze_alpha_path = output_dir / "adapter" / "freeze_alpha_dict.json" 
        with open(freeze_alpha_path, "w") as f:
            json.dump(freeze_alpha_dict, f)

    logger.info(f"Adapter saved to {output_dir / 'adapter'}")

    base_model = peft_model.unload()
    
    for name, param in base_model.named_parameters():
        if isinstance(param.data, torch.Tensor) and not param.data.is_contiguous():
            param.data = param.data.contiguous()
    base_model.save_pretrained(output_dir / "base_model")
    logger.info(f"Base model saved to {output_dir / 'base_model'}")

    if elapsed is not None or error_dict is not None or post_init_ppl is not None:
        results = {
            "initialization_time": elapsed,
            "error_dict": error_dict,
            "post_init_ppl": post_init_ppl,
        }
        with open(output_dir / "adapt_and_save_results.yaml", "w") as f:
            yaml.safe_dump(results, f)
        results.pop("error_dict")
        logger.info(f"Adapter initialization ({adapter_init}) completed:\n{results}")
    else:
        logger.info(f"Adapter initialization ({adapter_init}) completed")



def adapt_and_save_cls_model(
    model_name_or_path: str,
    adapter_init: str,
    output_dir: str,
    qera_calibration_set: str,
    qera_calibration_set_type: str,
    qera_num_calibration_samples: int,
    qera_calibration_batch_size: int,
    qera_max_seq_length: int,
    loftq_num_iters: int,
    quant_type: str,
    quant_bits: int,
    lora_rank: int,
    lora_alpha: float,
    lora_target_modules: list[str] | None,
    device_map: str,
    num_workers: int,
    overwrite_output_dir: bool,
    overwrite_dataset_cache: bool,
    qera_scaling_mode: str,
    peek_post_init_metrics: bool,
    lora_modules_to_save: list[str] | None,
    mxint_block_size: int,
    num_labels: int,
    init_method: str,
    config: dict,
    qera_num_iter: int
):
    assert adapter_init in ["loftq", "qera", "qlora", "lora"]
    assert qera_scaling_mode in ["diag", "rxx"]
    assert quant_type in ["nf", "fp", "mxint"]
    if quant_type in ["nf", "fp"]:
        assert quant_bits in [2, 4]
    assert qera_calibration_set_type in ["downstream", "pretrain"]
    PAD_TO_MAX_LENGTH = True
    MLM_PROBABILITY = 0.15
    

    if lora_target_modules is None:
        if "deberta" in model_name_or_path.lower():
            lora_target_modules = ["key_proj", "query_proj", "value_proj", "dense"]
        elif "roberta" in model_name_or_path.lower():
            lora_target_modules = r"roberta\.encoder\.layer\.\d+\.(attention\.self\.(query|key|value)|(attention\.output\.dense)|(intermediate\.dense)|(output\.dense))"
        else:
            raise ValueError(
                f"Cannot determine default modules to save for {model_name_or_path}"
            )

        logger.info(f"🔍 Using default lora_target_modules: {lora_target_modules}")

    if lora_modules_to_save is None:
        if "deberta" in model_name_or_path.lower():
            lora_modules_to_save = ["pooler.dense", "classifier"]
        elif "roberta" in model_name_or_path.lower():
            lora_modules_to_save = ["classifier"]
        else:
            raise ValueError(
                f"Cannot determine default modules to save for {model_name_or_path}"
            )

    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite_output_dir:
            raise FileExistsError(f"Output directory {output_dir} already exists")
        else:
            logger.warning(
                f"⚠️ Output directory {output_dir} already exists and will be overwritten"
            )
            shutil.rmtree(output_dir, ignore_errors=True)

    scale_dict = None
    calibration_dataloader = None
    calibration_time = 0.0

    
    hess_dict = None
    need_hess = hess_checker(config)

    if need_hess:
        hess_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        
        if qera_calibration_set_type == "downstream":
            hess_data_collator = transformers.default_data_collator
        else:
            hess_data_collator = transformers.DataCollatorForLanguageModeling(
                tokenizer=hess_tokenizer, mlm=True, mlm_probability=MLM_PROBABILITY
            )
            
        hess_model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, _attn_implementation="eager"
        )
        
        hess_model.eval()
        if "cuda" in device_map:
            hess_model.to(device_map)
        else:
            if hasattr(hess_model, "tie_weights"):
                hess_model.tie_weights()
            map_dict = create_device_map(hess_model, device_map)
            hess_model = dispatch_model(hess_model, map_dict)
            
        layers_to_register_and_share = find_layers_to_register_scale_hook(hess_model)
        hess_dict = hess_loader(config)
        
        if hess_dict: 
            logger.info(f"✅ Hessian loaded successfully") 
        else:
            logger.info(f"🔊 Hessian required, but not found. Generating...")
            hess_dict = generate_save_load_hessian(hess_model, hess_tokenizer, hess_data_collator,
                                                   qera_calibration_set=qera_calibration_set,
                                                   qera_max_seq_length=qera_max_seq_length,
                                                   num_workers=num_workers,
                                                   overwrite_dataset_cache=overwrite_dataset_cache,
                                                   qera_calibration_batch_size=qera_calibration_batch_size,
                                                   qera_num_calibration_samples=qera_num_calibration_samples,
                                                   hess_path=config['hess_path'],
                                                   mode='cls'
                                                   )
        del hess_tokenizer
        del hess_model
        del hess_data_collator
        
        share_scales(hess_dict, layers_to_register_and_share)
    else:
        logger.info(f"❌ Hessian not used in whole script")  
        
    output_ref = []
    
    if adapter_init == "qera" or (peek_post_init_metrics and adapter_init != "lora"):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=False
        )
        
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, _attn_implementation="eager"
        )
        model.eval()
        
        if "cuda" in device_map:
            model.to(device_map)
        else:
            if hasattr(model, "tie_weights"):
                model.tie_weights()
            device_map = create_device_map(model, device_map)
            model = dispatch_model(model, device_map)

        if qera_calibration_set_type == "downstream":
            data_collator = transformers.default_data_collator
        else:
            data_collator = transformers.DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROBABILITY
            )
        
        scale_dict = get_precomputed_scale_dict(config, logger)
        if not scale_dict:
            if adapter_init == "qera":
                layers_to_register_and_share = find_layers_to_register_scale_hook(model)
                profiler_factory = register_scale_hooks(
                    model,
                    layers_to_register_and_share,
                    mode=qera_scaling_mode,
                    torch_dtype=torch.float32,
                )
            calibration_datamodule = get_data_module_for_peft(
                qera_calibration_set,
                tokenizer=tokenizer,
                model_config=model.config,
                pad_to_max_length=PAD_TO_MAX_LENGTH,
                max_length=qera_max_seq_length,
                num_workers=num_workers,
                overwrite_cache=overwrite_dataset_cache,
            )
            calibration_dataloader = DataLoader(
                calibration_datamodule["train"],
                batch_size=qera_calibration_batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=data_collator,
            )
            start = time.time()
            num_samples = 0
            input_device = next(model.parameters()).device
            for i, batch in enumerate(calibration_dataloader):
                with torch.no_grad():
                    batch = {
                        k: v.to(input_device) for k, v in batch.items() if k != "labels"
                    }
                    outputs = model(**batch, output_hidden_states=True)
                last_hidden_states = outputs.hidden_states[-1]
                output_ref.append(last_hidden_states.cpu())
                num_samples += qera_calibration_batch_size
                if num_samples >= qera_num_calibration_samples:
                    break
            if adapter_init == "qera":
                profiler_factory.remove_all_hooks()
                scale_dict = profiler_factory.get_scale_dict(progress_bar=True)
                share_scales(scale_dict, layers_to_register_and_share)
                
                if qera_scaling_mode in ['diag', 'rxx']:
                    save_scale_dict(scale_dict, config)
                    
            calibration_time = time.time() - start
        else:
            layers_to_register_and_share = find_layers_to_register_scale_hook(model)
            share_scales(scale_dict, layers_to_register_and_share)
            
    use_4bit_bnb = (
        adapter_init in ["qlora", "loftq"]
        and quant_type in ["nf", "fp"]
        and quant_bits == 4
    )
    bnb_config = None
    if use_4bit_bnb:
        bnb_4bit_use_double_quant = True
        bnb_4bit_use_double_quant = quant_bits == 4
        bnb_quant_type_4bit = "nf4" if quant_type == "nf" else "fp4"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb_quant_type_4bit,
            bnb_4bit_quant_storage=torch.uint8,
            llm_int8_skip_modules=lora_modules_to_save,
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, quantization_config=bnb_config, num_labels=num_labels, 
        )
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=num_labels
        )
        
        if "cuda" in device_map:
            model = model.to(device_map)
        else:
            device_map = create_device_map(model, device_map)
            model = dispatch_model(model, device_map)

    model.eval()
    lora_target_modules_ = lora_target_modules
    if isinstance(lora_target_modules, (list, tuple)) and len(lora_target_modules) == 1 and lora_target_modules[0] == "all-linear":
        lora_target_modules_ = "all-linear"

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=True,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=lora_target_modules_,
        init_lora_weights=True,
        modules_to_save=lora_modules_to_save,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.eval()

    error_dict = None
    elapsed = None
    freeze_rank_dict = None 
    freeze_alpha_dict = None  
    
    if adapter_init == "loftq":
        if quant_bits == 4 and quant_type in ["nf", "fp"]:
            start = time.time()
            error_dict = replace_lora_weights_loftq_4bit(
                peft_model, num_iters=loftq_num_iters, init_method=init_method, hess_dict=hess_dict,
            )
            elapsed = time.time() - start
        else:
            start = time.time()
            error_dict = replace_lora_weights_loftq_kbit(
                peft_model,
                quant_type=quant_type,
                num_bits=quant_bits,
                num_iters=loftq_num_iters,
                mxint_block_size=mxint_block_size,
                init_method=init_method,
                hess_dict=hess_dict,
            )
            elapsed = time.time() - start
    elif adapter_init == "qera":
        if quant_bits == 4 and quant_type in ["nf", "fp"]:
            start = time.time()
            error_dict = replace_lora_weights_qera_4bit(
                peft_model, scale_dict=scale_dict, init_method=init_method, hess_dict=None, qera_num_iter=qera_num_iter,
            )
            elapsed = time.time() - start + calibration_time
        else:
            start = time.time()
            error_dict, freeze_rank_dict, freeze_alpha_dict = replace_lora_weight_qera_kbit(
                peft_model,
                scale_dict=scale_dict,
                quant_type=quant_type,
                num_bits=quant_bits,
                mxint_block_size=mxint_block_size,
                init_method=init_method,
                hess_dict=hess_dict,
                qera_num_iter=qera_num_iter,
            )
            elapsed = time.time() - start + calibration_time
    elif adapter_init == "qlora":
        if quant_bits == 4 and quant_type in ["nf", "fp"]:
            pass
        else:
            start = time.time()
            error_dict = replace_lora_weight_qlora_kbit(
                peft_model,
                quant_type=quant_type,
                num_bits=quant_bits,
                mxint_block_size=mxint_block_size,
                init_method=init_method, 
                hess_dict=hess_dict,   
            )
            elapsed = time.time() - start
    elif adapter_init == "lora":
        pass
    else:
        raise ValueError(f"Invalid adapter init: {adapter_init}")

    post_init_error = None
    if peek_post_init_metrics and adapter_init != "lora":
        num_samples = 0
        peft_model.eval()
        post_init_outputs = []
        input_device = next(peft_model.parameters()).device
        for i, batch in enumerate(calibration_dataloader):
            batch = {k: v.to(input_device) for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = peft_model(**batch, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            post_init_outputs.append(last_hidden_states.cpu())
            num_samples += qera_calibration_batch_size
            if num_samples >= qera_num_calibration_samples:
                break
        errors = []
        for ref, post in zip(output_ref, post_init_outputs):
            errors.append((ref.cuda() - post.cuda()).abs().mean().cpu().item())
        post_init_error = sum(errors) / len(errors)

    output_dir.mkdir(parents=True, exist_ok=True)
    make_weights_contiguous(peft_model)

    peft_model.save_pretrained(output_dir / "adapter")

    if freeze_rank_dict is not None:
        import json
        freeze_rank_path = output_dir / "adapter" / "freeze_rank_dict.json" 
        with open(freeze_rank_path, "w") as f: 
            json.dump(freeze_rank_dict, f)

    if freeze_alpha_dict is not None:
        import json
        freeze_alpha_path = output_dir / "adapter" / "freeze_alpha_dict.json" 
        with open(freeze_alpha_path, "w") as f:
            json.dump(freeze_alpha_dict, f)
         
    logger.info(f"Adapter saved to {output_dir / 'adapter'}")

    base_model = peft_model.unload()
    for name, param in base_model.named_parameters():
        if isinstance(param.data, torch.Tensor) and not param.data.is_contiguous():
            param.data = param.data.contiguous()
    base_model.save_pretrained(output_dir / "base_model")
    logger.info(f"Base model saved to {output_dir / 'base_model'}")


    if elapsed is not None or error_dict is not None or post_init_error is not None:
        results = {
            "initialization_time": elapsed,
            "error_dict": error_dict,
            "post_init_error": post_init_error,
        }
        with open(output_dir / "adapt_and_save_results.yaml", "w") as f:
            yaml.safe_dump(results, f)
        results.pop("error_dict")
        logger.info(f"Adapter initialization ({adapter_init}) completed:\n{results}")
    else:
        logger.info(f"Adapter initialization ({adapter_init}) completed")


def adapt_and_save_pipeline():
    parser = ArgumentParser()
    parser.add_argument(
        "model_type", type=str, choices=["clm", "cls"], help="Model type: clm or cls"
    )
    parser.add_argument("model_name_or_path", type=str)
    parser.add_argument(
        "adapter_init", type=str, choices=["loftq", "qera", "qlora", "lora"]
    )
    parser.add_argument("output_dir", type=str)
    parser.add_argument(
        "--qera-calibration-set",
        type=str,
        default=None,
        help="Default: wikitext2_peft for clm, required for cls",
    )
    parser.add_argument(
        "--qera-calibration-set-type",
        type=str,
        default="downstream",
        help="Default: downstream, required for cls",
        choices=["downstream", "pretrain"],
    )
    parser.add_argument("--qera-num-calibration-samples", type=int, default=128)
    parser.add_argument("--qera-calibration-batch-size", type=int, default=2)
    parser.add_argument("--qera-max-seq-length", type=int, default=2048)
    parser.add_argument(
        "--qera-scaling-mode",
        type=str,
        default="diag",
        help="Default: diag",
        choices=["diag", "rxx"],
    )
    parser.add_argument("--loftq-num-iters", type=int, default=1, help="Default: 1")
    parser.add_argument(
        "--quant-type",
        type=str,
        default="fp",
        choices=["nf", "fp", "mxint"],
        help="quantization type for the frozen weights. 'nf' means NormalFloat and 'fp' means FloatingPoint",
    )
    parser.add_argument(
        "--quant-bits", type=int, default=4, help="Default: 4", choices=[2, 3, 4]
    )
    parser.add_argument("--lora-rank", type=int, default=64, help="Default: 64")
    parser.add_argument(
        "--lora-alpha", type=float, default=128.0, help="Default: 128.0"
    )
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="Default: 0.1")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="+",
        default=None,
        help="Default: all linear layers except the output layer",
    )
    parser.add_argument(
        "--lora-modules-to-save",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument("--device-map", type=str, default="cuda", help="Default: cuda")
    parser.add_argument("--num-workers", type=int, default=8, help="Default: 8")
    parser.add_argument(
        "--overwrite-output-dir",
        "-ow",
        dest="overwrite_output_dir",
        action="store_true",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-dataset-cache", action="store_true")
    parser.add_argument("--peek-post-init-metrics", action="store_true", default=False)
    parser.add_argument("--mxint-block-size", type=int, default=32)
    parser.add_argument("--num-labels", type=int, default=2)
    
    parser.add_argument(
        "--init-method",
        dest="init_method",
        type=str,
        default=None,
        help="Method for LR initialization. e.g. 'odlri'"
    )

    parser.add_argument(
        "--qera-num-iter",
        dest="qera_num_iter",
        type=int,
        default=1,
        help="Number of iterations for QERA initialization. Default: 1"
    )
    
    parser.add_argument(
        "--global_dir",
        dest="global_dir",
        type=str,
        default=None,
        help="storage directory for resources, such as Hessian."
    )

    
    args = parser.parse_args()
    logger.info(f"Arguments\n{pformat(vars(args), sort_dicts=True)}")
    transformers.set_seed(args.seed)

    if args.global_dir is None: 
        default_dir = os.getcwd() + '/storage'
        os.makedirs(default_dir, exist_ok=True)
        logger.info(f"Storage for Hessian, RXX is not explicitly outlined. {default_dir} for default") 
        args.global_dir = default_dir

    else: 
        logger.info(f"Storage dir: {args.global_dir}")
        os.makedirs(args.global_dir, exist_ok=True)


    mini_cfg = {
        "model_name": args.model_name_or_path,
        "adapter_init": args.adapter_init,
        "init_method": args.init_method,
        "output_dir": args.output_dir,
        "qera_calibration_set": args.qera_calibration_set,
        "qera_num_calibration_samples": args.qera_num_calibration_samples,
        "qera_scaling_mode": args.qera_scaling_mode,
        "global_dir": args.global_dir
    }

    if args.model_type == "clm":
        adapt_and_save_clm_model(
            args.model_name_or_path,
            adapter_init=args.adapter_init,
            output_dir=args.output_dir,
            qera_calibration_set=args.qera_calibration_set,
            qera_num_calibration_samples=args.qera_num_calibration_samples,
            qera_calibration_batch_size=args.qera_calibration_batch_size,
            qera_max_seq_length=args.qera_max_seq_length,
            loftq_num_iters=args.loftq_num_iters,
            quant_type=args.quant_type,
            quant_bits=args.quant_bits,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
            lora_modules_to_save=args.lora_modules_to_save,
            device_map=args.device_map,
            num_workers=args.num_workers,
            overwrite_output_dir=args.overwrite_output_dir,
            overwrite_dataset_cache=args.overwrite_dataset_cache,
            qera_scaling_mode=args.qera_scaling_mode,
            peek_post_init_metrics=args.peek_post_init_metrics,
            mxint_block_size=args.mxint_block_size,
            init_method=args.init_method, 
            config=mini_cfg,
            qera_num_iter=args.qera_num_iter,
        )
    elif args.model_type == "cls":
        adapt_and_save_cls_model(
            args.model_name_or_path,
            adapter_init=args.adapter_init,
            output_dir=args.output_dir,
            qera_calibration_set=args.qera_calibration_set,
            qera_calibration_set_type=args.qera_calibration_set_type,
            qera_num_calibration_samples=args.qera_num_calibration_samples,
            qera_calibration_batch_size=args.qera_calibration_batch_size,
            qera_max_seq_length=args.qera_max_seq_length,
            loftq_num_iters=args.loftq_num_iters,
            quant_type=args.quant_type,
            quant_bits=args.quant_bits,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_target_modules=args.lora_target_modules,
            lora_modules_to_save=args.lora_modules_to_save,
            device_map=args.device_map,
            num_workers=args.num_workers,
            overwrite_output_dir=args.overwrite_output_dir,
            overwrite_dataset_cache=args.overwrite_dataset_cache,
            qera_scaling_mode=args.qera_scaling_mode,
            peek_post_init_metrics=args.peek_post_init_metrics,
            mxint_block_size=args.mxint_block_size,
            num_labels=args.num_labels,
            init_method=args.init_method,
            config=mini_cfg,
            qera_num_iter=args.qera_num_iter,
        )

    args_dict = vars(args)
    with open(Path(args.output_dir) / "adapt_and_save_args.yaml", "w") as f:
        yaml.safe_dump(args_dict, f)

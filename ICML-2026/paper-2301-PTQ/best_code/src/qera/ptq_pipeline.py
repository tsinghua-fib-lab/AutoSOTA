import logging
import re
import yaml
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat
import math
import datetime

from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
import transformers
from accelerate import dispatch_model, init_empty_weights
import pandas as pd

from .statistic_profiler import register_scale_hooks, share_scales
from .datasets import get_data_module
from .evaluate import evaluate_perplexity, evaluate_harness_downstream
from .models import (
    find_layers_to_approximate,
    quantize_model,
    find_layers_to_register_scale_hook,
)
from .approximate import compute_AB_and_approximation_error, attach_LR, attach_quantized_weight
from .utils import create_device_map, get_all_device_mem_info
import sys
import os

#Perplexity computation through lm-eval
from lm_eval import evaluator, tasks    
from lm_eval.models.huggingface import HFLM

logger = logging.getLogger(__name__)


def _mse_threshold_emoji(mse: float) -> str:
    warning_threshold = 1e-4
    error_threshold = 0.1

    if mse < warning_threshold:
        return "✅"
    elif mse < error_threshold:
        return "⚠️"
    else:
        return "❌"
    
    
def _act_mse_threshold_emoji(act_mse: float) -> str:
    warning_threshold = 1e-4
    error_threshold = 0.1

    #TODO - add a threshold for act_mse (later)

    if act_mse < warning_threshold:
        return "✅"
    elif act_mse < error_threshold:
        return "⚠️"
    else:
        return "❌"
    

def make_output_dir(config, srr_seed=42):
    model_name = config["model_name"].replace("/", "_")
    init_name = config['quant_config']['default-1']['lr_initializer']['name']
    itr = config['quant_config']['default-1']['iter']
    lr_rank = config['quant_config']['default-1']['rank']
    scale_mode = config['lr_processor']['scaling_mode']
    
    q_cfg = config['quant_config']['default-1']['w_quantizer']
    q_name = q_cfg['name']
    q_bit = q_cfg['width'] if q_name in ['mxint'] else q_cfg['num_bits']
    
    cal_set = config["calibration_set"]
    cal_num = config["num_calibration_samples"]
    
    output_dir = f'./checkpoints/ptq/{init_name}/{scale_mode}/{model_name}/{cal_set}_{cal_num}/{q_name}_{q_bit}/{lr_rank}_{itr}/seed_{srr_seed}'
    os.makedirs(output_dir, exist_ok=True)
    
    return Path(output_dir)
    

def hess_checker(config):
    '''
    This function checks if Hessian (2x.Tx) is used in quantization process.
    if yes, it will return True, else False.
    '''
    hess_require_quantizer = ['gptq']
    hess_require_lr = ['hess', 'cholesky']
    hess_require_lr_init = ['odlri', 'resq_x', 'resq_wx', 'resq_wx_ada']

    
    need_hess = False
    need_hess = config['lr_processor']['scaling_mode'] in hess_require_lr
    
    cfg = config['quant_config']
    cfg_lst = list(cfg.keys())
    for default in cfg_lst:
        if default.startswith('default'):
            is_dft_hess = cfg[default].get('hess', None)
            is_hess_quant =  cfg[default].get('w_quantizer')['name'] in hess_require_quantizer
            is_hess_lr_init =  cfg[default].get('lr_initializer')['name'] in hess_require_lr_init
            need_hess = need_hess or (is_dft_hess or is_hess_quant or is_hess_lr_init)
            
            cfg[default]['hess_require_quantizer'] = hess_require_quantizer
            cfg[default]['hess_require_lr_init'] = hess_require_lr_init

    return need_hess

def hess_loader(config):
    '''
    This function loads the Hessian (2x.Tx) from the model.
    If hess is not used, it will return None. 
    '''
    model_name = config['model_name'].replace('/', '_')
    cal_set = config['calibration_set']
    num_cal_set = config['num_calibration_samples']
    hess_path = f'ptq_hess_{model_name}___{cal_set}___{num_cal_set}.pt'
    hess_dir = config['global_dir']
    
    os.makedirs(hess_dir, exist_ok=True)
    hess_path = os.path.join(hess_dir, hess_path)
    
    # hess_path example: '/opt/data/quantization/hessian_llama2_7b.pt'
    hess_dict = {}
    if os.path.exists(hess_path):
        inter_dict = torch.load(hess_path)
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
        config['hess_path'] = hess_path
            
    return hess_dict



def _ledoit_wolf_shrinkage(S, logger=None):
    """
    Ledoit-Wolf optimal shrinkage towards identity.
    
    Computes the analytically optimal shrinkage intensity that minimizes
    the Frobenius MSE between the shrunk estimator and the true covariance.
    Parameter-free; especially beneficial when n_samples < n_features
    (256 calibration samples vs 2048-dim activations).
    
    Reference: Ledoit & Wolf (2004)
    """
    import torch as _torch
    d = S.shape[0]
    if d <= 1:
        return S, 0.0
    
    device = S.device
    dtype = S.dtype
    
    # Pooled variance: mean of diagonal
    mu = _torch.trace(S) / d
    
    # Variance of diagonal elements around mu
    diag = S.diag()
    d2 = _torch.sum((diag - mu) ** 2) / d
    
    # Mean squared off-diagonal elements
    total_sq = _torch.sum(S ** 2)
    diag_sq = _torch.sum(diag ** 2)
    off_diag_sq = total_sq - diag_sq
    b2_bar = off_diag_sq / (d * (d - 1))
    
    # Ledoit-Wolf shrinkage intensity
    b2 = _torch.min(b2_bar, d2)
    shrinkage = (b2 / (d2 + _torch.finfo(dtype).eps)).clamp(0.0, 1.0)
    
    # Apply shrinkage: S_shrunk = (1-lambda)*S + lambda*mu*I
    target = mu * _torch.eye(d, device=device, dtype=dtype)
    S_shrunk = (1.0 - shrinkage) * S + shrinkage * target
    
    if logger is not None:
        logger.info(f'Ledoit-Wolf shrinkage lambda={shrinkage.item():.4f} | '
                    f'cond(S)={_torch.linalg.cond(S).item():.2e} -> '
                    f'cond(S_shrunk)={_torch.linalg.cond(S_shrunk).item():.2e}')
    
    return S_shrunk, shrinkage.item()


def get_precomputed_scale_dict(config, hess_dict, logger):
    '''
    function that returns the precomputed scale dict.
    if not found, it will return empty dict
    '''
    scale_dict = {}
    
    scale_mode = config['lr_processor']['scaling_mode']
    match scale_mode:
        case 'diag' | 'rxx' | 'lqer':
            glob_dir = config['global_dir']
            model_name, cal_set, num_cal_set = config['model_name'].replace('/', '_'), \
                                                config['calibration_set'], config['num_calibration_samples']
            f_name = f'{scale_mode}_{model_name}___{cal_set}_{num_cal_set}.pt'
            f_pth = Path(os.path.join(glob_dir, f_name))
            if f_pth.is_file():
                logger.info(f"✅ Precomputed scale dict found. Skipping data calibration...")
                if scale_mode in ['diag', 'lqer']:
                    scale_dict = torch.load(f_pth, map_location=torch.device('cpu'))
                else:
                    inter_dict = torch.load(f_pth, map_location=torch.device('cpu'))
                    for k, v in tqdm(inter_dict.items(), desc="Loading Scales"):
                        N, diag_val, upper_val = v['N'], v['diag_val'], v['upper_val']
                        recon = torch.zeros((N, N), device='cpu')
                        idcs = torch.arange(N, device='cpu')
                        r_idx, c_idx = torch.triu_indices(N, N, offset=1)
                        recon[idcs, idcs] = diag_val
                        recon[r_idx, c_idx] = upper_val
                        recon[c_idx, r_idx] = upper_val
                        scale_dict[k] = recon
            else:
                logger.info(f"❌ No precomputed scale dict found for {scale_mode}. Running calibration")
                    
        case 'cholesky': 
            logger.info("✅ Using Cholesky scale. Skipping data calibration...")
            for k, h in tqdm(hess_dict.items(), desc="Loading Scales"):
                H = h/2  #assume hess is 2x.Tx
                H = H.to(torch.float64).to('cuda')
                try:
                    H_shrunk, lw_lambda = _ledoit_wolf_shrinkage(H, logger)
                    L = torch.linalg.cholesky(H_shrunk).to(torch.float32).cpu()
                except Exception:
                    logger.info(f"Cholesky failed for {k}. Adding noise and retrying...")
                    H += torch.eye(H.shape[0], device='cuda', dtype=torch.float64) * 1e-6
                    try:
                        H_shrunk, lw_lambda = _ledoit_wolf_shrinkage(H, logger)
                        L = torch.linalg.cholesky(H_shrunk).to(torch.float32).cpu()
                    except Exception:
                        logger.info(f"❌ Cholesky failed again for {k}. Fallback to Eigendecomposition...")
                        # Fallback to Eigendecomposition: H = Q * Lambda * Q^T => L = Q * sqrt(Lambda)
                        # Ensure Lambda is non-negative
                        eig_val, eig_vec = torch.linalg.eigh(H)
                        eig_val = torch.clamp(eig_val, min=0.0)
                        Lambda_sqrt = torch.diag(torch.sqrt(eig_val))
                        L = (eig_vec @ Lambda_sqrt).to(torch.float32).cpu()
                scale_dict[k] = L.T
        
        case _:
            logger.info(f"❌ No precomputed scale dict for {scale_mode}. Running calibration")
    
    return scale_dict


def save_scale_dict(scale_dict, config):
    glob_dir = config['global_dir']
    scale_mode = config['lr_processor']['scaling_mode']
    model_name, cal_set, num_cal_set = config['model_name'].replace('/', '_'), \
                                    config['calibration_set'], config['num_calibration_samples']
    f_name = f'{scale_mode}_{model_name}___{cal_set}_{num_cal_set}.pt'
    f_pth = Path(os.path.join(glob_dir, f_name))

    if scale_mode in ['rxx']:
        # with gzip.open(f_pth, "wb") as f:
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

        torch.save(inter_dict, f_pth)
    
    else: 
        # with gzip.open(f_pth, "wb") as f:
        for k, v in scale_dict.items():
            scale_dict[k] = v.cpu()
        torch.save(scale_dict, f_pth)
            
    logger.info(f"✅ Precomputed scale dict saved to {f_pth}")

def generate_save_load_hessian(model, tokenizer, data_collator, config):
    '''
    This function generates Hessian 2x.Tx, saves it.
    ''' 
    layers_to_register_and_share = find_layers_to_register_scale_hook(model)
    profiler_factory = register_scale_hooks(
        model,
        layers_to_register_and_share=layers_to_register_and_share,
        mode='hess',
        torch_dtype=config['quant_dtype'],
        mode_map=config["lr_processor"]["scaling_mode_map"])
    
    calibration_datamodule = get_data_module(
        name=config["calibration_set"],
        tokenizer=tokenizer,
        padding="max_length",
        max_length=config["perplexity_max_seq_length"],
        num_raw_samples=20 * config["num_calibration_samples"],
        num_workers=config["num_workers"],
    )
    calibration_dataloader = DataLoader(
        calibration_datamodule["train"],
        batch_size=config["perplexity_eval_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=data_collator,
    )
    
    mem_info = get_all_device_mem_info()
    logger.info(f"Device memory before profiling starts: \n{pformat(mem_info)}")
    _ = evaluate_perplexity(
        model=model,
        eval_dataloader=calibration_dataloader,
        num_samples=config["num_calibration_samples"],
        progress_bar=True,
        input_device=None,
        description="Generating Hessian",
    )
    profiler_factory.remove_all_hooks()
    hess_dict = profiler_factory.get_scale_dict(progress_bar=True)
    hess_path = config['hess_path']
    
    f32_hess_dict = {}
    # with gzip.open(hess_path, "wb") as f:
    inter_dict = {}
    for k, v in hess_dict.items():
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

    return f32_hess_dict


def pipeline_qera():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--model-name", dest="model_name", type=str, help="Model name", default=None
    )
    parser.add_argument(
        "--quant-dtype",
        dest="quant_dtype",
        type=str,
        help="data type used in quantization",
        default=None,
    )
    parser.add_argument(
        "--eval-dtype",
        dest="eval_dtype",
        type=str,
        help="Evaluation data type",
        default=None,
    )
    parser.add_argument(
        "--device-map", dest="device_map", type=str, help="Device map", default=None
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        help="Number of workers",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        help="Output directory",
        default=None,
    )
    parser.add_argument(
        "--LR-dict", dest="LR_dict", type=str, help="LR dict", default=None
    )
    parser.add_argument(
        "--calibration-set",
        dest="calibration_set",
        type=str,
        help="Calibration set",
        default=None,
    )
    parser.add_argument(
        "--num-calibration-samples",
        dest="num_calibration_samples",
        type=int,
        help="Number of calibration samples",
        default=None,
    )
    parser.add_argument(
        "--perplexity-eval-batch-size",
        dest="perplexity_eval_batch_size",
        type=int,
        help="Perplexity evaluation batch size",
        default=None,
    )
    parser.add_argument(
        "--perplexity-eval-set",
        dest="perplexity_eval_set",
        type=str,
        help="Perplexity evaluation set",
        default=None,
    )
    parser.add_argument(
        "--perplexity-max-seq-length",
        dest="perplexity_max_seq_length",
        type=int,
        help="Perplexity max sequence length",
        default=None,
    )
    parser.add_argument(
        "--lm-eval-tasks",
        dest="lm_eval_tasks",
        type=str,
        nargs="+",
        help="LM eval tasks",
        default=None,
    )
    parser.add_argument(
        "--lm-eval-num-fewshot",
        dest="lm_eval_num_fewshot",
        type=int,
        help="LM eval num fewshot",
        default=None,
    )
    parser.add_argument(
        "--lm-eval-batch-size",
        dest="lm_eval_batch_size",
        type=str,
        help="LM eval batch size",
        default=None,
    )
    parser.add_argument(
        "--disable-lr",
        dest="disable_lr",
        action="store_true",
        help="Disable LR",
        default=None,
    )
    parser.add_argument(
        "--lr-scaling-mode",
        dest="lr_scaling_mode",
        type=str,
        help="LR scaling mode, one of ['diagonal', 'diag', 'rxx', 'identity', 'lqer', 'hess', 'w-only', 'null'].",
        default=None,
        choices=[
            "diagonal",
            "diag",
            "rxx",
            "identity",
            "lqer",
            'hess',
            "cholesky",
            'w-only',
            'null'
        ],  # "diag" is alias of "diagonal"
    )
    parser.add_argument(
        "--rxx-sqrtm-implementation",
        dest="rxx_sqrtm_implementation",
        type=str,
        help="RXX sqrtm implementation, one of ['blocked', 'iterative'].",
        default=None,
        choices=["blocked", "iterative"],
    )
    parser.add_argument(
        "--rxx-sqrtm-num-iters",
        dest="rxx_sqrtm_num_iters",
        type=int,
        help="Number of iterations for iterative sqrtm",
        default=None,
    )
    parser.add_argument(
        "--disable-perplexity-eval",
        dest="disable_perplexity_eval",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--disable-lm-eval", dest="disable_lm_eval", action="store_true", default=None
    )
    parser.add_argument(
        "--overwrite-output-dir",
        "-ow",
        dest="overwrite_output_dir",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--max-position-embeddings",
        dest="max_position_embeddings",
        type=int,
        default=None,
        help="Llama-3-8.1 max position embeddings is too large for perplexity eval in harness",
    )
    parser.add_argument(
        "--srr-seed",
        type=int,
        default=42,
        help="Random seed for SRR's quantization error matrix"
    )
    parser.add_argument(
        "--apply-rand-svd",
        dest="apply_rand_svd",
        action="store_true",
        default=None,
        help="Use randomized SVD (svd_lowrank) instead of full SVD in SRR initialization",
    )

    args = parser.parse_args()
    args = vars(args)

    with open(args["config"], "r") as f:
        config = yaml.safe_load(f)

    override_args = {}
    args.pop("config")
    for entry, value in args.items():
        if value is not None:
            config[entry] = value
            override_args[entry] = value
            
    #overwrite scaling mode
    config['lr_processor']['scaling_mode'] = args['lr_scaling_mode'] if args['lr_scaling_mode'] \
                                                        else config['lr_processor']['scaling_mode']

    # --apply-rand-svd: validate and inject into quant_config lr_initializer dicts
    if config.get("apply_rand_svd", False):
        quant_cfg = config["quant_config"]
        has_srr = False
        for key in list(quant_cfg.keys()):
            if isinstance(quant_cfg[key], dict):
                lr_init = quant_cfg[key].get("lr_initializer", {})
                if lr_init.get("name") == "srr":
                    lr_init["apply_rand_svd"] = True
                    has_srr = True
        if not has_srr:
            raise ValueError(
                "--apply-rand-svd requires at least one lr_initializer with "
                "name 'srr' in the quant_config, but none was found."
            )
        logger.info("Randomized SVD (svd_lowrank) enabled for SRR initialization")

    logger.info(f"Configuration: \n{pformat(config, indent=4)}")
    logger.info(f"Override arguments: \n{pformat(override_args, indent=4)}")

    model_name = config["model_name"]
    quant_dtype = getattr(torch, config["quant_dtype"])
    eval_dtype = getattr(torch, config["eval_dtype"])
    device_map = config["device_map"]
    num_workers = config["num_workers"]
    output_dir = (
        Path(config["output_dir"]) if config["output_dir"] is not None else None
    )
    LR_dict = config["LR_dict"]
    Wq_dict = config['Wq_dict']
    calibration_set = config["calibration_set"]
    num_calibration_samples = config["num_calibration_samples"]
    perplexity_evaluation_set = config["perplexity_eval_set"]
    perplexity_eval_batch_size = config["perplexity_eval_batch_size"]
    perplexity_max_seq_length = config["perplexity_max_seq_length"]
    lm_eval_tasks = config["lm_eval_tasks"]
    lm_eval_num_fewshot = config["lm_eval_num_fewshot"]
    lm_eval_batch_size = config["lm_eval_batch_size"]
    if isinstance(lm_eval_batch_size, str) and not "auto" in lm_eval_batch_size:
        lm_eval_batch_size = int(lm_eval_batch_size)
    max_position_embeddings = config["max_position_embeddings"]

    disable_lr = config["disable_lr"]
    lr_scaling_mode = config["lr_processor"]["scaling_mode"]
    rxx_sqrtm_implementation = config["lr_processor"]["rxx_sqrtm_implementation"]
    rxx_sqrtm_num_iters = config["lr_processor"]["rxx_sqrtm_num_iters"]
    quant_config = config["quant_config"]
    disable_perplexity_eval = config["disable_perplexity_eval"]
    disable_lm_eval = config["disable_lm_eval"]
    lr_scaling_mode_map = config["lr_processor"]["scaling_mode_map"]
    overwrite_output_dir = config["overwrite_output_dir"]
    save_params = config['save_params']

    if config['lr_processor']['scaling_mode'] == 'w-only':
        disable_lr = True
    
    output_dir = make_output_dir(config, args['srr_seed']) if output_dir is None else output_dir
    if output_dir and len(list(output_dir.iterdir())) > 0:
        if not overwrite_output_dir:
            logger.warning(
                f"⚠️ Output directory {output_dir} already exists. Use --overwrite-output-dir to overwrite it."
            )
            raise ValueError(f"Output directory {output_dir} is not empty")
        else:
            logger.warning(
                f"⚠️ Output directory {output_dir} already exists. Overwriting it."
            )
            param_dir = output_dir / "quant_params.pt.gz"
            if param_dir.is_file():
                logger.warning(
                    f"⚠️ Precomputed parameters found in {param_dir}. No PTQ will be performed."
                )

    #When 'null' => evaluates full precision model, thus no quantization is performed
    if lr_scaling_mode == 'null': 
        other_model_kwargs = {}
        if max_position_embeddings is not None:
            other_model_kwargs["max_position_embeddings"] = max_position_embeddings
        pure_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=eval_dtype,
        _attn_implementation="eager",
        **other_model_kwargs,
        ).to('cuda')
        
        device_map_ = create_device_map(pure_model, device_map=device_map)
        pure_model = dispatch_model(pure_model, device_map_)
        pure_model.eval() 
        
        if not disable_perplexity_eval:
            logger.info("🚀 Evaluating perplexity...")    
            Pure_HFLN_model = HFLM(pure_model)
            results = evaluator.simple_evaluate(
                model=Pure_HFLN_model,  
                tasks=['wikitext'],
                device=Pure_HFLN_model.device,  
                batch_size=16,  
            )
            del Pure_HFLN_model

            ppl_results = {
                "perplexity": results["results"]["wikitext"]['word_perplexity,none'],
            }
        
            logger.info(
                f"Pure model Perplexity: {ppl_results['perplexity']:.4f}"
            )
            with open(output_dir / "perplexity_results.yaml", "w") as f:
                yaml.dump(ppl_results, f)
            
        if not disable_lm_eval:
            logger.info("🚀 Evaluating lm-eval downstream tasks...")
            lm_eval_results = evaluate_harness_downstream(
                pure_model,
                tasks=lm_eval_tasks,
                num_fewshot=lm_eval_num_fewshot,
                use_cache=None,
                batch_size=lm_eval_batch_size,
            )
            logger.info(f"Downstream task results: \n{lm_eval_results['table_view']}")
            with open(output_dir / "lm_eval_results.yaml", "w") as f:
                yaml.dump(lm_eval_results, f)
        
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        sys.exit()
        
    # sqrtm_implementation (use blocked as default)
    if lr_scaling_mode in ["rxx", "mixed"]:
        if rxx_sqrtm_implementation == "blocked":
            logger.info("🔊 Using blocked sqrtm i mplementation. Only CPU + Scipy is supported")
        elif rxx_sqrtm_implementation == "iterative":
            logger.info(f"🔊 Using iterative sqrtm implementation (number of iterations={rxx_sqrtm_num_iters})")
        else:
            raise ValueError(f"Unknown sqrtm_implementation: {rxx_sqrtm_implementation}")


    # Load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    other_model_kwargs = {}
    if max_position_embeddings is not None:
        other_model_kwargs["max_position_embeddings"] = max_position_embeddings
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=quant_dtype,
        _attn_implementation="eager",
        **other_model_kwargs,
    )
    model.eval()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    device_map_ = create_device_map(model, device_map=device_map)
    logger.info(f"Device map: {device_map_}")
    model = dispatch_model(model, device_map_)
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    layers_to_register_and_share = find_layers_to_register_scale_hook(model)
    
    hess_dict = None
    need_hess = hess_checker(config)
    if need_hess:
        hess_dict = hess_loader(config)
        if hess_dict: 
            logger.info(f"✅ Hessian loaded successfully") 
        else:
            logger.info(f"🔊 Hessian required, but not found. Generating...")
            hess_dict = generate_save_load_hessian(model, tokenizer, data_collator, config)
            
        share_scales(hess_dict, layers_to_register_and_share)
    else: 
        logger.info(f"❌ Hessian not used in whole script")


    scale_dict = None
    if not disable_lr and LR_dict is None:

        scale_dict = get_precomputed_scale_dict(config, hess_dict, logger)

        if not scale_dict:
            if lr_scaling_mode == "identity":
                logger.info("🔊 Using identity scale (torch.eye)")
                
            logger.info("🚀 Running data calibration...")
            profiler_factory = register_scale_hooks(
                model,
                layers_to_register_and_share=layers_to_register_and_share,
                mode=lr_scaling_mode,
                torch_dtype=quant_dtype,
                mode_map=lr_scaling_mode_map,
            )

            calibration_datamodule = get_data_module(
                name=calibration_set,
                tokenizer=tokenizer,
                padding="max_length",
                max_length=perplexity_max_seq_length,
                num_raw_samples=20 * num_calibration_samples,
                num_workers=num_workers,
            )

            calibration_dataloader = DataLoader(
                calibration_datamodule["train"],
                batch_size=perplexity_eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=data_collator,
            )

            mem_info = get_all_device_mem_info()
            logger.info(f"Device memory before profiling starts: \n{pformat(mem_info)}")
            
            profile_outputs = evaluate_perplexity(
                model=model,
                eval_dataloader=calibration_dataloader,
                num_samples=(
                    num_calibration_samples
                    if lr_scaling_mode != "identity"
                    else perplexity_eval_batch_size
                ),
                progress_bar=True,
                input_device=None,
                description="Calibrating",
            )
            
            profiler_factory.remove_all_hooks()
            if lr_scaling_mode == "rxx":
                scale_dict = profiler_factory.get_scale_dict(
                    progress_bar=True,
                    sqrtm_implementation=rxx_sqrtm_implementation,
                    sqrtm_num_iters=rxx_sqrtm_num_iters,
                )
            else:
                scale_dict = profiler_factory.get_scale_dict(progress_bar=True)

            del profiler_factory
            # logger.info(f"Perplexity after profiling: {profile_outputs['perplexity']:.4f}")
                    
        share_scales(scale_dict, layers_to_register_and_share)            


    #NOTE - This is the main quantization process.
    if LR_dict is None and Wq_dict is None:
        logger.info("🚀 Quantizing. Computing L & R...")
        layers_to_approximate = find_layers_to_approximate(model)
        LR_dict, W_q_dict, mse_df = compute_AB_and_approximation_error( 
                model, layers_to_approximate, quant_config, scale_dict, hess_dict, False, args['srr_seed'])
        
        mse_df_emoji = mse_df.copy()
        mse_df_emoji.loc[:, "mse?"] = mse_df["mse"].apply(_mse_threshold_emoji)
        mse_df_emoji.loc[:, "act_mse?"] = mse_df["act_mse"].apply(_act_mse_threshold_emoji)
        
        logger.info(
                f"Approximation error (mean squared error): \n{mse_df_emoji.to_markdown()}"
        )
        
        del scale_dict
    else: 
        raise NotImplementedError("Assume LR_dict is None and Wq_dict is None")
    
 
    del model
    torch.cuda.empty_cache()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=eval_dtype,
        _attn_implementation="eager",
        **other_model_kwargs,
    )
    logger.info("🚀 Loading new model...")
    quantize_model(model, quant_config) #Original weights are replaced to quantized weights
    
    
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    model.eval()
    
    logger.info("🚀 Replacing weights to Q, LR...")
    attach_LR(model, layers_to_approximate, LR_dict)
    attach_quantized_weight(model, layers_to_approximate, W_q_dict)
    
    device_map_ = create_device_map(model, device_map=device_map)
    model = dispatch_model(model, device_map_)

    if not disable_perplexity_eval:
        logger.info("🚀 Evaluating perplexity...")

        model.eval() 
        HFLN_model = HFLM(model)
        results = evaluator.simple_evaluate(
            model=HFLN_model,  
            tasks=['wikitext'],
            device=model.device,  
            batch_size=16,  
        )

        del HFLN_model
        torch.cuda.empty_cache()

        ppl_results = {
            "perplexity": results["results"]["wikitext"]['word_perplexity,none'],
        }

        if disable_lr:
            logger.info(
                f"Perplexity after quantization (no LR): {ppl_results['perplexity']:.4f}"
            )
        else:
            logger.info(
                f"Perplexity after approximation: {ppl_results['perplexity']:.4f}"
            )
            

    if not disable_lm_eval:
        logger.info("🚀 Evaluating lm-eval downstream tasks...")
        lm_eval_results = evaluate_harness_downstream(
            model,
            tasks=lm_eval_tasks,
            num_fewshot=lm_eval_num_fewshot,
            use_cache=None,
            batch_size=lm_eval_batch_size,
        )
        logger.info(f"Downstream task results: \n{lm_eval_results['table_view']}")

    if output_dir is not None:
        logger.info(f"🚀 Saving results to {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        if not disable_lr and mse_df is not None:
            # save approximation results
            mse_df.to_csv(output_dir / "approximation_error.csv", index=False)
            
            if save_params:
                logger.info(f"🚀 Saving quantization parameters to {output_dir}")
                param_dir = output_dir / "quant_params.pt.gz"
                config['quant_params'] = param_dir.resolve().as_posix()

        # save perplexity results
        if not disable_perplexity_eval:
            with open(output_dir / "perplexity_results.yaml", "w") as f:
                yaml.dump(ppl_results, f)

        # save lm-eval results
        if not disable_lm_eval:
            with open(output_dir / "lm_eval_results.yaml", "w") as f:
                yaml.dump(lm_eval_results, f)

        # save config
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)


def pipeline_fp16_bf16_fp32():
    parser = ArgumentParser()
    parser.add_argument("model_name", type=str, help="Model name")
    parser.add_argument(
        "--dtype",
        dest="dtype",
        type=str,
        help="Evaluation data type",
        default="bfloat16",
    )
    parser.add_argument(
        "--device-map",
        dest="device_map",
        type=str,
        help="Device map",
        default="auto-balanced",
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        help="Number of workers",
        default=8,
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        help="Output directory",
        default=None,
    )
    parser.add_argument(
        "--perplexity-eval-set",
        dest="perplexity_eval_set",
        type=str,
        help="Perplexity evaluation set",
        default="wikitext2",
    )
    parser.add_argument(
        "--perplexity-eval-batch-size",
        dest="perplexity_eval_batch_size",
        type=int,
        help="Perplexity evaluation batch size",
        default=4,
    )
    parser.add_argument(
        "--perplexity-max-seq-length",
        dest="perplexity_max_seq_length",
        type=int,
        help="Perplexity max sequence length",
        default=2048,
    )
    parser.add_argument(
        "--lm-eval-tasks",
        dest="lm_eval_tasks",
        type=str,
        nargs="+",
        help="LM eval tasks",
        default=["qera_benchmark_classic", "qera_benchmark_hard"],
    )
    parser.add_argument(
        "--lm-eval-num-fewshot",
        dest="lm_eval_num_fewshot",
        type=int,
        help="LM eval num fewshot",
        default=None,
    )
    parser.add_argument(
        "--lm-eval-batch-size",
        dest="lm_eval_batch_size",
        type=str,
        help="LM eval batch size",
        default="auto",
    )
    parser.add_argument(
        "--disable-perplexity-eval", dest="disable_perplexity_eval", action="store_true"
    )
    parser.add_argument(
        "--disable-lm-eval", dest="disable_lm_eval", action="store_true"
    )
    parser.add_argument(
        "--max-position-embeddings",
        dest="max_position_embeddings",
        type=int,
        default=None,
        help="Llama-3-8.1 max position embeddings is too large for perplexity eval in harness",
    )
    parser.add_argument(
        "--attn-implementation",
        dest="attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "flash_attention_2", "sdpa"],
    )

    args = parser.parse_args()

    logger.info(f"Arguments: \n{pformat(vars(args), indent=4)}")

    model_name = args.model_name
    dtype = getattr(torch, args.dtype)
    device_map = args.device_map
    num_workers = args.num_workers
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    perplexity_evaluation_set = args.perplexity_eval_set
    perplexity_eval_batch_size = args.perplexity_eval_batch_size
    perplexity_max_seq_length = args.perplexity_max_seq_length
    lm_eval_tasks = args.lm_eval_tasks
    lm_eval_num_fewshot = args.lm_eval_num_fewshot
    lm_eval_batch_size = args.lm_eval_batch_size
    if not "auto" in lm_eval_batch_size:
        lm_eval_batch_size = int(lm_eval_batch_size)
    disable_perplexity_eval = args.disable_perplexity_eval
    disable_lm_eval = args.disable_lm_eval
    attn_implementation = args.attn_implementation

    # check output directory
    if (
        output_dir is not None
        and output_dir.is_dir()
        and len(list(output_dir.iterdir())) > 0
    ):
        raise ValueError(f"Output directory {output_dir} is not empty")

    # Load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    other_model_kwargs = {}
    if args.max_position_embeddings is not None:
        other_model_kwargs["max_position_embeddings"] = args.max_position_embeddings
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        _attn_implementation=attn_implementation,
        **other_model_kwargs,
    )
    model.eval()
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    model = dispatch_model(
        model, device_map=create_device_map(model, device_map=device_map)
    )

    if not disable_perplexity_eval:
        logger.info("🚀 Evaluating perplexity...")
        perplexity_datamodule = get_data_module(
            name=perplexity_evaluation_set,
            tokenizer=tokenizer,
            padding="max_length",
            max_length=perplexity_max_seq_length,
            num_raw_samples=None,
            num_workers=num_workers,
        )
        perplexity_dataloader = DataLoader(
            perplexity_datamodule["test"],
            batch_size=perplexity_eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
        )

        perplexity_results = evaluate_perplexity(
            model=model,
            eval_dataloader=perplexity_dataloader,
            num_samples=None,
            progress_bar=True,
            input_device=None,
            description="Evaluating perplexity",
        )

        logger.info(f"Perplexity: {perplexity_results['perplexity']:.4f}")

    if not disable_lm_eval:
        logger.info("🚀 Evaluating lm-eval downstream tasks...")
        lm_eval_results = evaluate_harness_downstream(
            model,
            tasks=lm_eval_tasks,
            num_fewshot=lm_eval_num_fewshot,
            use_cache=None,
            batch_size=lm_eval_batch_size,
        )
        logger.info(f"Downstream task results: \n{lm_eval_results['table_view']}")

    if output_dir is not None:
        logger.info(f"🚀 Saving results to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # save perplexity results
        if not disable_perplexity_eval:
            with open(output_dir / "perplexity_results.yaml", "w") as f:
                yaml.dump(perplexity_results, f)

        # save lm-eval results
        if not disable_lm_eval:
            with open(output_dir / "lm_eval_results.yaml", "w") as f:
                yaml.dump(lm_eval_results, f)

        # save args
        with open(output_dir / "args.yaml", "w") as f:
            yaml.dump(vars(args), f)


def pipeline_q_baseline():
    from transformers import BitsAndBytesConfig, AwqConfig, GPTQConfig, HqqConfig

    gptq_available = False
    try:
        from auto_gptq import exllama_set_max_input_length

        gptq_available = True
    except ImportError:
        pass

    parser = ArgumentParser()
    parser.add_argument("model_name", type=str, help="Model name")
    parser.add_argument(
        "q_method",
        type=str,
        help="Quantization method",
        choices=[
            "bnb-4bit",
            "gptq",
            "awq",
            "bnb-8bit",
            "hqq-4bit",
            "hqq-3bit",
            "hqq-2bit",
        ],
    )
    parser.add_argument(
        "--dtype",
        dest="dtype",
        type=str,
        help="Evaluation data type",
        default="bfloat16",
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        help="Number of workers",
        default=8,
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        help="Output directory",
        default=None,
    )
    parser.add_argument(
        "--perplexity-eval-set",
        dest="perplexity_eval_set",
        type=str,
        help="Perplexity evaluation set",
        default="wikitext2",
    )
    parser.add_argument(
        "--perplexity-eval-batch-size",
        dest="perplexity_eval_batch_size",
        type=int,
        help="Perplexity evaluation batch size",
        default=4,
    )
    parser.add_argument(
        "--perplexity-max-seq-length",
        dest="perplexity_max_seq_length",
        type=int,
        help="Perplexity max sequence length",
        default=2048,
    )
    parser.add_argument(
        "--lm-eval-tasks",
        dest="lm_eval_tasks",
        type=str,
        nargs="+",
        help="LM eval tasks",
        default=["qera_benchmark_classic", "qera_benchmark_hard"],
    )
    parser.add_argument(
        "--lm-eval-num-fewshot",
        dest="lm_eval_num_fewshot",
        type=int,
        help="LM eval num fewshot",
        default=None,
    )
    parser.add_argument(
        "--lm-eval-batch-size",
        dest="lm_eval_batch_size",
        type=str,
        help="LM eval batch size",
        default="auto",
    )
    parser.add_argument(
        "--max-position-embeddings",
        dest="max_position_embeddings",
        type=int,
        default=None,
        help="Llama-3-8.1 max position embeddings is too large for perplexity eval in harness",
    )
    parser.add_argument(
        "--hqq-group-size",
        dest="hqq_group_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--disable-perplexity-eval", dest="disable_perplexity_eval", action="store_true"
    )
    parser.add_argument(
        "--disable-lm-eval", dest="disable_lm_eval", action="store_true"
    )

    args = parser.parse_args()

    logger.info(f"Arguments: \n{pformat(vars(args), indent=4)}")

    model_name = args.model_name
    q_method = args.q_method
    dtype = getattr(torch, args.dtype)
    num_workers = args.num_workers
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    perplexity_evaluation_set = args.perplexity_eval_set
    perplexity_eval_batch_size = args.perplexity_eval_batch_size
    perplexity_max_seq_length = args.perplexity_max_seq_length
    lm_eval_tasks = args.lm_eval_tasks
    lm_eval_num_fewshot = args.lm_eval_num_fewshot
    lm_eval_batch_size = args.lm_eval_batch_size
    if not "auto" in lm_eval_batch_size:
        lm_eval_batch_size = int(lm_eval_batch_size)
    disable_perplexity_eval = args.disable_perplexity_eval
    disable_lm_eval = args.disable_lm_eval

    hqq_group_size = args.hqq_group_size

    other_model_kwargs = {}
    if args.max_position_embeddings is not None:
        other_model_kwargs["max_position_embeddings"] = args.max_position_embeddings

    # check output directory
    if (
        output_dir is not None
        and output_dir.is_dir()
        and len(list(output_dir.iterdir())) > 0
    ):
        raise ValueError(f"Output directory {output_dir} is not empty")

    # Load model and tokenizer
    if q_method in ["bnb-4bit", "bnb-8bit"]:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=q_method == "bnb-4bit",
            load_in_8bit=q_method == "bnb-8bit",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            **other_model_kwargs,
        )
    elif q_method == "hqq-4bit":
        hqq_config = HqqConfig(
            nbits=4,
            group_size=hqq_group_size,
            quant_zero=False,
            quant_scale=False,
            axis=0,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            quantization_config=hqq_config,
            **other_model_kwargs,
        )
    elif q_method == "hqq-3bit":
        hqq_config = HqqConfig(
            nbits=3,
            group_size=hqq_group_size,
            quant_zero=False,
            quant_scale=False,
            axis=0,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            quantization_config=hqq_config,
            **other_model_kwargs,
        )
    elif q_method == "hqq-2bit":
        hqq_config = HqqConfig(
            nbits=2,
            group_size=hqq_group_size,
            quant_zero=False,
            quant_scale=False,
            axis=0,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            quantization_config=hqq_config,
            **other_model_kwargs,
        )
    elif q_method == "awq":
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="cuda", **other_model_kwargs
        )
    elif q_method == "gptq":
        assert gptq_available, "auto-gptq is not installed"
        model = exllama_set_max_input_length(model, 8192)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="cuda", **other_model_kwargs
        )
    model.eval()

    logger.info(f"🔊 Quantization method: {q_method}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    if not disable_perplexity_eval:
        logger.info("🚀 Evaluating perplexity...")
        perplexity_datamodule = get_data_module(
            name=perplexity_evaluation_set,
            tokenizer=tokenizer,
            padding="max_length",
            max_length=perplexity_max_seq_length,
            num_raw_samples=None,
        )
        perplexity_dataloader = DataLoader(
            perplexity_datamodule["test"],
            batch_size=perplexity_eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
        )

        perplexity_results = evaluate_perplexity(
            model=model,
            eval_dataloader=perplexity_dataloader,
            num_samples=None,
            progress_bar=True,
            input_device=None,
            description="Evaluating perplexity",
        )

        logger.info(f"Perplexity: {perplexity_results['perplexity']:.4f}")

    if not disable_lm_eval:
        logger.info("🚀 Evaluating lm-eval downstream tasks...")
        lm_eval_results = evaluate_harness_downstream(
            model,
            tasks=lm_eval_tasks,
            num_fewshot=lm_eval_num_fewshot,
            use_cache=None,
            batch_size=lm_eval_batch_size,
        )
        logger.info(f"Downstream task results: \n{lm_eval_results['table_view']}")

    if output_dir is not None:
        logger.info(f"🚀 Saving results to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # save perplexity results
        if not disable_perplexity_eval:
            with open(output_dir / "perplexity_results.yaml", "w") as f:
                yaml.dump(perplexity_results, f)

        # save lm-eval results
        if not disable_lm_eval:
            with open(output_dir / "lm_eval_results.yaml", "w") as f:
                yaml.dump(lm_eval_results, f)

        # save args
        with open(output_dir / "args.yaml", "w") as f:
            yaml.dump(vars(args), f)


def _check_chunk_id(model_name, layers_per_chunk, chunk_id=None):
    """
    Check if the chunk_id is valid for the given model and layers_per_chunk.
    """
    with init_empty_weights():
        config = transformers.AutoConfig.from_pretrained(
            model_name, _attn_implementation="eager"
        )
        model = transformers.AutoModelForCausalLM.from_config(config)
        model_cls = model.__class__
        model = model_cls(config)
    layers_to_register_and_share = find_layers_to_register_scale_hook(model)

    num_chunks = math.ceil(len(layers_to_register_and_share) / layers_per_chunk)

    if chunk_id is not None:
        if chunk_id > num_chunks:
            logger.error(
                f"❌ chunk_id (={chunk_id}) must be smaller than the number of chunks ({num_chunks})"
            )
            raise RuntimeError(
                f"chunk_id (={chunk_id}) must be smaller than the number of chunks ({num_chunks})"
            )
    else:
        logger.info(f"Model name: {model_name}")
        logger.info(f"Layers per chunk: {layers_per_chunk}")
        logger.info(f"Allowed chunk IDs: [0, {num_chunks - 1}]")

    return num_chunks


def _verify_AB_dict_chunks(
    AB_dict_dir: Path, num_chunks: int, current_chunk_tag=None
) -> set[str]:
    chunks_to_check = [f"{i}-of-{num_chunks-1}.pt" for i in range(num_chunks)]
    if current_chunk_tag is not None:
        chunks_to_check.remove(current_chunk_tag + ".pt")

    if AB_dict_dir.is_dir():
        existing_chunks = [f.name for f in AB_dict_dir.iterdir() if f.is_file()]
    else:
        existing_chunks = []
    missing_chunks = set(chunks_to_check) - set(existing_chunks)
    return missing_chunks


def pipeline_qera_chunked():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--model-name", dest="model_name", type=str, help="Model name", default=None
    )
    parser.add_argument(
        "--qera-dtype",
        dest="qera_dtype",
        type=str,
        help="QERA data type",
        default=None,
    )
    parser.add_argument(
        "--eval-dtype",
        dest="eval_dtype",
        type=str,
        help="Evaluation data type",
        default=None,
    )
    parser.add_argument(
        "--device-map", dest="device_map", type=str, help="Device map", default=None
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        help="Number of workers",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        help="Output directory",
        default=None,
    )
    parser.add_argument(
        "--calibration-set",
        dest="calibration_set",
        type=str,
        help="Calibration set",
        default=None,
    )
    parser.add_argument(
        "--num-calibration-samples",
        dest="num_calibration_samples",
        type=int,
        help="Number of calibration samples",
        default=None,
    )
    parser.add_argument(
        "--perplexity-eval-batch-size",
        dest="perplexity_eval_batch_size",
        type=int,
        help="Perplexity evaluation batch size",
        default=None,
    )
    parser.add_argument(
        "--perplexity-eval-set",
        dest="perplexity_eval_set",
        type=str,
        help="Perplexity evaluation set",
        default=None,
    )
    parser.add_argument(
        "--perplexity-max-seq-length",
        dest="perplexity_max_seq_length",
        type=int,
        help="Perplexity max sequence length",
        default=None,
    )
    parser.add_argument(
        "--lm-eval-tasks",
        dest="lm_eval_tasks",
        type=str,
        nargs="+",
        help="LM eval tasks",
        default=None,
    )
    parser.add_argument(
        "--lm-eval-num-fewshot",
        dest="lm_eval_num_fewshot",
        type=int,
        help="LM eval num fewshot",
        default=None,
    )
    parser.add_argument(
        "--lm-eval-batch-size",
        dest="lm_eval_batch_size",
        type=str,
        help="LM eval batch size",
        default=None,
    )
    parser.add_argument(
        "--disable-qera",
        dest="disable_qera",
        action="store_true",
        help="Disable QERA",
        default=None,
    )
    parser.add_argument(
        "--qera-scaling-mode",
        dest="qera_scaling_mode",
        type=str,
        help="QERA scaling mode, one of ['diagonal', 'diag', 'rxx', 'identity', 'lqer'].",
        default=None,
        choices=[
            "diagonal",
            "diag",
            "rxx",
            "identity",
            "lqer",
        ],  # "diag" is alias of "diagonal"
    )
    parser.add_argument(
        "--qera-sqrtm-implementation",
        dest="qera_sqrtm_implementation",
        type=str,
        help="QERA sqrtm implementation, one of ['blocked', 'iterative'].",
        default=None,
        choices=["blocked", "iterative"],
    )
    parser.add_argument(
        "--qera-sqrtm-num-iters",
        dest="qera_sqrtm_num_iters",
        type=int,
        help="Number of iterations for iterative sqrtm",
        default=None,
    )
    parser.add_argument(
        "--disable-perplexity-eval",
        dest="disable_perplexity_eval",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--disable-lm-eval", dest="disable_lm_eval", action="store_true", default=None
    )
    parser.add_argument(
        "--max-position-embeddings",
        dest="max_position_embeddings",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--layers-per-chunk",
        dest="layers_per_chunk",
        type=int,
        help="Layers per chunk",
        default=None,
    )
    parser.add_argument(
        "--chunk-id", dest="chunk_id", type=int, help="Chunk ID", default=None
    )

    args = parser.parse_args()
    args = vars(args)

    with open(args["config"], "r") as f:
        config = yaml.safe_load(f)

    override_args = {}
    args.pop("config")
    for entry, value in args.items():
        if value is not None:
            config[entry] = value
            override_args[entry] = value

    logger.info(f"Configuration: \n{pformat(config, indent=4)}")
    logger.info(f"Override arguments: \n{pformat(override_args, indent=4)}")

    model_name = config["model_name"]
    qera_dtype = getattr(torch, config["qera_dtype"])
    eval_dtype = getattr(torch, config["eval_dtype"])
    device_map = config["device_map"]
    num_workers = config["num_workers"]
    output_dir = (
        Path(config["output_dir"]) if config["output_dir"] is not None else None
    )
    # AB_dict = config["AB_dict"]
    calibration_set = config["calibration_set"]
    num_calibration_samples = config["num_calibration_samples"]
    perplexity_evaluation_set = config["perplexity_eval_set"]
    perplexity_eval_batch_size = config["perplexity_eval_batch_size"]
    perplexity_max_seq_length = config["perplexity_max_seq_length"]
    lm_eval_tasks = config["lm_eval_tasks"]
    lm_eval_num_fewshot = config["lm_eval_num_fewshot"]
    lm_eval_batch_size = config["lm_eval_batch_size"]
    if isinstance(lm_eval_batch_size, str) and not "auto" in lm_eval_batch_size:
        lm_eval_batch_size = int(lm_eval_batch_size)

    disable_qera = config["disable_qera"]
    qera_scaling_mode = config["qera_scaling_mode"]
    qera_sqrtm_implementation = config["qera_sqrtm_implementation"]
    qera_sqrtm_num_iters = config["qera_sqrtm_num_iters"]
    qera_config = config["qera_config"]
    disable_perplexity_eval = config["disable_perplexity_eval"]
    disable_lm_eval = config["disable_lm_eval"]
    qera_scaling_mode_map = config["qera_scaling_mode_map"]

    layers_per_chunk = config["layers_per_chunk"]
    chunk_id = config["chunk_id"]

    # assert chunk_id is not None
    assert output_dir is not None

    num_chunks = _check_chunk_id(model_name, layers_per_chunk, chunk_id)
    chunk_tag = f"{chunk_id}-of-{num_chunks-1}"

    # check output directory
    AB_dict_dir = output_dir.joinpath("AB_dict")
    missing_chunks = _verify_AB_dict_chunks(
        AB_dict_dir=AB_dict_dir, num_chunks=num_chunks, current_chunk_tag=None
    )
    assert not (
        len(missing_chunks) > 0 and chunk_id is None
    ), f"Missing chunks: {missing_chunks}"
    other_model_kwargs = {}
    if config["max_position_embeddings"] is not None:
        other_model_kwargs["max_position_embeddings"] = config[
            "max_position_embeddings"
        ]

    if len(missing_chunks) > 0:
        # only allows disable_qera=False and qera_scaling_mode in ["diag", "diagonal", "rxx", "mixed", "identity"]
        if disable_qera:
            raise ValueError("disable_qera=True is not supported for chunked pipeline.")
        else:
            if qera_scaling_mode not in [
                "diag",
                "diagonal",
                "rxx",
                "mixed",
                "identity",
                "lqer",
            ]:
                raise ValueError(
                    "qera_scaling_mode should be one of ['diagonal', 'diag', 'rxx', 'mixed', 'identity', 'lqer']"
                )

        # sqrtm_implementation
        if qera_scaling_mode in ["rxx", "mixed"]:
            if qera_sqrtm_implementation == "blocked":
                # refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html
                logger.info(
                    "🔊 Using blocked sqrtm implementation. Only CPU + Scipy is supported"
                )
            elif qera_sqrtm_implementation == "iterative":
                # refer to https://link.springer.com/article/10.1023/A:1019150005407
                logger.info(
                    f"🔊 Using iterative sqrtm implementation (number of iterations={qera_sqrtm_num_iters})"
                )
            else:
                raise ValueError(
                    f"Unknown sqrtm_implementation: {qera_sqrtm_implementation}"
                )

        # Load model and tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=qera_dtype,
            _attn_implementation="eager",
            **other_model_kwargs,
        )
        model.eval()
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        device_map = create_device_map(model, device_map=device_map)
        logger.info(f"Device map: {device_map}")
        model = dispatch_model(model, device_map)
        data_collator = transformers.DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )

        # solve chunk_id
        layers_to_register_and_share = find_layers_to_register_scale_hook(model)
        layers_to_register_and_share = layers_to_register_and_share[
            chunk_id::num_chunks
        ]
        logger.info(
            f"🔊 Chunk id = {chunk_id}, total number of chunks = {num_chunks}, layers included in this chunk:\n{pformat(list(map(lambda x: x['target_layer'], layers_to_register_and_share)))}"
        )

        profiler_factory = register_scale_hooks(
            model,
            layers_to_register_and_share=layers_to_register_and_share,
            mode=qera_scaling_mode,
            torch_dtype=qera_dtype,
            mode_map=qera_scaling_mode_map,
        )

        calibration_datamodule = get_data_module(
            name=calibration_set,
            tokenizer=tokenizer,
            padding="max_length",
            max_length=perplexity_max_seq_length,
            num_raw_samples=20 * num_calibration_samples,
            num_workers=num_workers,
        )

        calibration_dataloader = DataLoader(
            calibration_datamodule["train"],
            batch_size=perplexity_eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
        )

        mem_info = get_all_device_mem_info()
        logger.info(f"Device memory before profiling starts: \n{pformat(mem_info)}")
        profile_outputs = evaluate_perplexity(
            model=model,
            eval_dataloader=calibration_dataloader,
            num_samples=(
                num_calibration_samples
                if qera_scaling_mode != "identity"
                else perplexity_eval_batch_size
            ),
            progress_bar=True,
            input_device=None,
            description="Calibrating",
        )

        profiler_factory.remove_all_hooks()
        if qera_scaling_mode in ["rxx", "mixed"]:
            scale_dict = profiler_factory.get_scale_dict(
                progress_bar=True,
                sqrtm_implementation=qera_sqrtm_implementation,
                sqrtm_num_iters=qera_sqrtm_num_iters,
            )
        else:
            scale_dict = profiler_factory.get_scale_dict(progress_bar=True)

        share_scales(scale_dict, layers_to_register_and_share)
        logger.info(f"Perplexity after profiling: {profile_outputs['perplexity']:.4f}")
        logger.info("🚀 QERA is enabled. Computing A & B...")
        layers_to_approximate = find_layers_to_approximate(model)
        layers_to_approximate = list(
            filter(lambda x: x in scale_dict, layers_to_approximate)
        )
        AB_dict, mse_df = compute_AB_and_approximation_error(
            model, layers_to_approximate, scale_dict, qera_config, move_model_back=False
        )
        del scale_dict
        mse_df_emoji = mse_df.copy()
        mse_df_emoji.loc[:, "mse?"] = mse_df["mse"].apply(_mse_threshold_emoji)
        logger.info(
            f"Approximation error (mean squared error): \n{mse_df_emoji.to_markdown()}"
        )

        missing_chunks = _verify_AB_dict_chunks(
            AB_dict_dir=AB_dict_dir, num_chunks=num_chunks, current_chunk_tag=chunk_tag
        )

        # save this chunk
        mse_df_dir = output_dir.joinpath("approximation_error")
        config_dir = output_dir.joinpath("config")
        AB_dict_path = AB_dict_dir.joinpath(f"{chunk_tag}.pt")
        logger.info(f"Current missing chunks: {missing_chunks}")
        AB_dict_dir.mkdir(parents=True, exist_ok=True)
        mse_df_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)

        mse_df.to_csv(mse_df_dir.joinpath(f"{chunk_tag}.csv"), index=False)
        AB_dict = {k: v.cpu() for k, v in AB_dict.items()}
        torch.save(AB_dict, AB_dict_path)
        with open(config_dir.joinpath(f"{chunk_tag}.yaml"), "w") as f:
            yaml.dump(config, f)
    else:
        logger.info(
            f"🔊 All chunks of AB_dict are ready. Quantize model, attach AB_dict and run evaluation."
        )
        del model
        torch.cuda.empty_cache()
        # Load model and tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=eval_dtype,
            _attn_implementation="eager",
            **other_model_kwargs,
        )
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        device_map = create_device_map(model, device_map=device_map)
        logger.info(f"Device map: {device_map}")
        data_collator = transformers.DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )
        
        quantize_model(model, qera_config) #NOTE - Here
        model.eval()

    if len(missing_chunks) == 0:
        # merge all chunks
        AB_dict = {}
        AB_dict_chunks = list(
            filter(
                lambda x: x.is_file() and x.name.endswith(".pt"), AB_dict_dir.iterdir()
            )
        )
        for chunk in tqdm(AB_dict_chunks, desc="Loading chunks"):
            AB_dict.update(torch.load(chunk))

        # attach A & B
        layers_to_approximate = find_layers_to_approximate(model)
        attach_LR(model, layers_to_approximate, AB_dict)

        # evaluate
        if not disable_perplexity_eval:
            logger.info("🚀 Evaluating perplexity...")
            eval_datamodule = get_data_module(
                name=perplexity_evaluation_set,
                tokenizer=tokenizer,
                padding="max_length",
                max_length=perplexity_max_seq_length,
                num_raw_samples=None,
                num_workers=num_workers,
            )
            eval_dataloader = DataLoader(
                eval_datamodule["test"],
                batch_size=perplexity_eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=data_collator,
            )
            model = dispatch_model(model, device_map)
            mem_info = get_all_device_mem_info()
            logger.info(
                f"Device memory before perplexity evaluation starts: \n{pformat(mem_info)}"
            )
            ppl_results = evaluate_perplexity(
                model=model,
                eval_dataloader=eval_dataloader,
                num_samples=None,
                progress_bar=True,
                input_device=None,
                description="Evaluating",
            )

            if disable_qera:
                logger.info(
                    f"Perplexity after quantization (no QERA): {ppl_results['perplexity']:.4f}"
                )
            else:
                logger.info(
                    f"Perplexity after approximation: {ppl_results['perplexity']:.4f}"
                )

        if not disable_lm_eval:
            logger.info("🚀 Evaluating lm-eval downstream tasks...")
            model = dispatch_model(model, device_map)
            lm_eval_results = evaluate_harness_downstream(
                model,
                tasks=lm_eval_tasks,
                num_fewshot=lm_eval_num_fewshot,
                use_cache=None,
                batch_size=lm_eval_batch_size,
            )
            logger.info(f"Downstream task results: \n{lm_eval_results['table_view']}")

        # save perplexity results
        if not disable_perplexity_eval:
            with open(output_dir / "perplexity_results.yaml", "w") as f:
                yaml.dump(ppl_results, f)

        # save lm-eval results
        if not disable_lm_eval:
            with open(output_dir / "lm_eval_results.yaml", "w") as f:
                yaml.dump(lm_eval_results, f)
    else:
        logger.info(
            f"Chunk {chunk_tag} is saved. Please run the pipeline for the rest chunks."
        )
        logger.info(f"Missing chunks: \n{pformat(missing_chunks)}")


def chunk_checker():
    parser = ArgumentParser()
    parser.add_argument("model_name", type=str, help="Model name")
    parser.add_argument("layers_per_chunk", type=int, help="Layers per chunk")
    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        help="Output directory",
        default=None,
    )
    args = parser.parse_args()

    model_name = args.model_name
    layers_per_chunk = args.layers_per_chunk
    output_dir = Path(args.output_dir) if args.output_dir is not None else None

    num_chunks = _check_chunk_id(model_name, layers_per_chunk, None)

    if output_dir is not None:
        AB_dict_dir = output_dir.joinpath("AB_dict")
        if not AB_dict_dir.is_dir():
            logger.warning(f"Output directory {output_dir} does not exist.")
            return
        else:
            missing_chunks = _verify_AB_dict_chunks(
                AB_dict_dir=AB_dict_dir, num_chunks=num_chunks, current_chunk_tag=None
            )
            if len(missing_chunks) == 0:
                logger.info("All chunks are ready.")
            else:
                logger.info(
                    f"Missing chunks: \n{pformat(missing_chunks, sort_dicts=False)}"
                )


def _merge_chunked_approximation_error(approx_error_dir: Path):
    if isinstance(approx_error_dir, str):
        approx_error_dir = Path(approx_error_dir)
    df = None
    for file in approx_error_dir.iterdir():
        if not file.is_file():
            continue

        if not re.match(r"\d+-of-\d+.csv", file.name):
            continue

        chunk_df = pd.read_csv(file)
        if df is None:
            df = chunk_df
        else:
            df = pd.concat([df, chunk_df], ignore_index=True)

    return df


def merge_chunked_results():
    parser = ArgumentParser()
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--quick-save",
        "-s",
        dest="quick_save",
        action="store_true",
        help="Save merged results to $output_dir/approximation_error/quick-save-$timestamp.csv",
        default=False,
    )
    parser.add_argument(
        "--output-file",
        "-o",
        dest="output_file",
        type=str,
        help="Output file",
        default=None,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    approx_error_dir = output_dir.joinpath("approximation_error")
    assert approx_error_dir.is_dir(), f"Directory {approx_error_dir} does not exist."

    df = _merge_chunked_approximation_error(approx_error_dir)

    logger.info(f"Merged approximation error: \n{df.to_markdown()}")

    if args.quick_save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        df.to_csv(approx_error_dir.joinpath(f"quick-save-{timestamp}.csv"), index=False)
        logger.info(
            f"Quick save to {approx_error_dir.joinpath(f'quick-save-{timestamp}.csv')}"
        )

    if args.output_file is not None:
        df.to_csv(args.output_file, index=False)
        logger.info(f"Saved to {args.output_file}")
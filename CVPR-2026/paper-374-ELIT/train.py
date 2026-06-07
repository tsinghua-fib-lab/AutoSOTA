import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json
import yaml

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torch.optim.lr_scheduler import ConstantLR, LinearLR, SequentialLR

# Import all model registries
from models.sit import SiT_models as _sit_models
from models.sit_dfm import SiT_models as _dfm_models
from models.sit_elit import SiT_models as _elit_models

from loss import SILoss, DFMSILoss
from utils import load_encoders

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

logger = get_logger(__name__)

# Combined model registry
ALL_MODELS = {}
ALL_MODELS.update(_sit_models)
ALL_MODELS.update(_dfm_models)
ALL_MODELS.update(_elit_models)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def get_model_type(model_name):
    """Determine model type from model name prefix."""
    if model_name.startswith('DFM-'):
        return 'dfm'
    elif model_name.startswith('ELIT-'):
        return 'elit'
    return 'sit'


def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag



def sample_sit(model, xT, ys, args, vae, latents_scale, latents_bias, accelerator):
    """Sample from a standard SiT or ELIT model."""
    from samplers import euler_sampler
    with torch.no_grad():
        samples = euler_sampler(
            model, 
            xT, 
            ys,
            num_steps=50, 
            cfg_scale=4.0,
            guidance_low=0.,
            guidance_high=1.,
            path_type=args.path_type,
            heun=False,
        ).to(torch.float32)
        samples = vae.decode((samples - latents_bias) / latents_scale).sample
        samples = (samples + 1) / 2.
    return {"samples": samples}


def sample_dfm(model, multiscale_xT, ys, args, vae, latents_scale, latents_bias, 
               accelerator, laplacian_decomposer, multiscale_gt_xs):
    """Sample from a DFM model with multiscale sampling."""
    from dfm_utils.samplers_dfm import dfm_euler_sampler
    with torch.no_grad():
        multiscale_samples = dfm_euler_sampler(
            model, 
            multiscale_xT, 
            ys,
            cfg_scale=4.0,
            guidance_low=0.,
            guidance_high=1.,
            path_type=args.path_type,
            heun=False,
            num_steps_per_scale=args.num_steps_per_scale,
            stage_thresholds=args.stage_sampling_thresholds,
        )
        
        log_dict = {}
        
        # Log per-stage outputs
        for stage_idx in range(args.stages_count):
            stage_sample = multiscale_samples[stage_idx].to(torch.float32)
            stage_decoded = vae.decode((stage_sample - latents_bias) / latents_scale).sample
            stage_decoded = (stage_decoded + 1) / 2.
            stage_decoded_gathered = accelerator.gather(stage_decoded)
            log_dict[f"generated_stage_{stage_idx}"] = wandb.Image(array2grid(stage_decoded_gathered))
        
        # Log ground truth stages
        for stage_idx in range(args.stages_count):
            stage_gt = multiscale_gt_xs[stage_idx].to(torch.float32)
            stage_gt_decoded = vae.decode((stage_gt - latents_bias) / latents_scale).sample
            stage_gt_decoded = (stage_gt_decoded + 1) / 2.
            stage_gt_decoded_gathered = accelerator.gather(stage_gt_decoded)
            log_dict[f"gt_stage_{stage_idx}"] = wandb.Image(array2grid(stage_gt_decoded_gathered))
        
        # Recompose
        samples = laplacian_decomposer.recompose(multiscale_samples).to(torch.float32)
        samples = vae.decode((samples - latents_bias) / latents_scale).sample
        samples = (samples + 1) / 2.
    
    return {"samples": samples, "extra_log": log_dict}



def main(args):
    model_type = get_model_type(args.model)
    
    # Set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
        logger.info(f"Model type: {model_type}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    # Determine whether REPA is enabled
    enable_repa = (args.proj_coeff > 0 and args.enc_type.lower() not in ('none', ''))

    # Load encoders only when REPA is enabled
    encoders, encoder_types, architectures = [], [], []
    z_dims = None
    if enable_repa:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
        )
        z_dims = [encoder.embed_dim for encoder in encoders]
        if accelerator.is_main_process:
            logger.info(f"REPA enabled: enc_type={args.enc_type}, proj_coeff={args.proj_coeff}, z_dims={z_dims}")
    else:
        if accelerator.is_main_process:
            logger.info("REPA disabled (no encoder loaded, no projectors created)")

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    
    # Build model with type-specific kwargs
    model_kwargs = dict(
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_prob > 0),
        enable_repa=enable_repa,
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    )
    
    if model_type == 'elit':
        model_kwargs.update(dict(
            enable_elit=True,
            elit_max_mask_prob=args.elit_max_mask_prob,
            elit_min_mask_prob=args.elit_min_mask_prob,
            group_size=args.elit_group_size,
            elit_read_depth=args.elit_read_depth,
            elit_write_depth=args.elit_write_depth,
        ))
    
    model = ALL_MODELS[args.model](**model_kwargs)

    model = model.to(device)
    ema = deepcopy(model).to(device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)
    
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # DFM-specific: Laplacian decomposer
    laplacian_decomposer = None
    if model_type == 'dfm':
        from dfm_utils.laplacian_decomposer import LaplacianDecomposer2D
        laplacian_decomposer = LaplacianDecomposer2D(
            stages_count=args.stages_count
        )

    # Create loss function
    if model_type == 'dfm':
        loss_fn = DFMSILoss(
            prediction=args.prediction,
            path_type=args.path_type, 
            accelerator=accelerator,
            latents_scale=latents_scale,
            latents_bias=latents_bias,
            weighting=args.weighting,
            num_stages=args.stages_count,
            stage_weights=args.stage_weights,
        )
    else:
        loss_fn = SILoss(
            prediction=args.prediction,
            path_type=args.path_type, 
            accelerator=accelerator,
            latents_scale=latents_scale,
            latents_bias=latents_bias,
            weighting=args.weighting
        )
    
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer:
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Setup LR scheduler:
    if args.lr_scheduler_type == "constant_with_linear_warmup" and args.warmup_steps > 0:
        warmup_iters = args.warmup_steps
        total_iters = args.max_train_steps
        start_factor = 1.0 / max(warmup_iters, 1)
        scheduler1 = LinearLR(
            optimizer,
            start_factor=start_factor,
            total_iters=warmup_iters,
        )
        scheduler2 = ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=total_iters - warmup_iters,
        )
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[warmup_iters],
        )
        if accelerator.is_main_process:
            print(f"Using constant_with_linear_warmup scheduler: warmup_steps={warmup_iters}, lr={args.learning_rate}")
    else:
        lr_scheduler = None
        if accelerator.is_main_process:
            print(f"Using constant LR: {args.learning_rate}")

    # Setup data:
    train_dataset = CustomDataset(args.data_dir, s3_cache_dir=args.s3_cache_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()
    
    # Resume:
    global_step = 0
    resume_step = args.resume_step
    
    # Auto-resume: find latest checkpoint if resume_step is -1
    if resume_step == -1:
        checkpoint_dir_path = f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints'
        if os.path.exists(checkpoint_dir_path):
            checkpoints = [f for f in os.listdir(checkpoint_dir_path) if f.endswith('.pt')]
            if checkpoints:
                steps = [int(f.replace('.pt', '')) for f in checkpoints]
                resume_step = max(steps)
                if accelerator.is_main_process:
                    logger.info(f"Auto-resume: Found latest checkpoint at step {resume_step}")
            else:
                resume_step = 0
        else:
            resume_step = 0
    
    if resume_step > 0:
        ckpt_name = str(resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            weights_only=False,
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']
        if lr_scheduler is not None and 'lr_scheduler' in ckpt:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        elif lr_scheduler is not None:
            # Checkpoint has no scheduler state — fast-forward the scheduler
            for _ in range(global_step):
                lr_scheduler.step()
        if accelerator.is_main_process:
            logger.info(f"Resumed training from checkpoint: {ckpt_name} (step {global_step})")

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REPA", 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Setup sampling data
    sample_batch_size = 64 // accelerator.num_processes
    gt_raw_images, gt_xs, _ = next(iter(train_dataloader))
    assert gt_raw_images.shape[-1] == args.resolution
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
        )
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    n = ys.size(0)
    
    # Create sampling noise (model-type specific)
    if model_type == 'dfm':
        multiscale_gt_xs = laplacian_decomposer.decompose(gt_xs)
        multiscale_xT = {}
        for stage in range(args.stages_count):
            stage_latent_size = latent_size // (2 ** (args.stages_count - 1 - stage))
            multiscale_xT[stage] = torch.randn((n, 4, stage_latent_size, stage_latent_size), device=device)
    else:
        xT = torch.randn((n, 4, latent_size, latent_size), device=device)
        
    for epoch in range(args.epochs):
        model.train()
        for raw_image, x, y in train_dataloader:
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            if args.legacy:
                drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, y)
            else:
                labels = y
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                # Compute encoder features only when REPA is enabled
                zs = None
                if enable_repa:
                    zs = []
                    with accelerator.autocast():
                        for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                            raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                            z = encoder.forward_features(raw_image_)
                            if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                            if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                            zs.append(z)

            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                loss, proj_loss = loss_fn(model, x, model_kwargs, zs=zs)
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean() if isinstance(proj_loss, torch.Tensor) else proj_loss
                loss = loss_mean + proj_loss_mean * args.proj_coeff
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model)
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    if lr_scheduler is not None:
                        checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if args.sampling_steps != 0 and (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                with torch.no_grad():
                    if model_type == 'dfm':
                        sample_result = sample_dfm(
                            model, multiscale_xT, ys, args, vae, 
                            latents_scale, latents_bias, accelerator,
                            laplacian_decomposer, multiscale_gt_xs
                        )
                        out_samples = accelerator.gather(sample_result["samples"].to(torch.float32))
                        gt_samples_decoded = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                        gt_samples_decoded = (gt_samples_decoded + 1) / 2.
                        gt_samples = accelerator.gather(gt_samples_decoded.to(torch.float32))
                        
                        log_dict = sample_result.get("extra_log", {})
                        log_dict["samples_recomposed"] = wandb.Image(array2grid(out_samples))
                        log_dict["gt_samples_recomposed"] = wandb.Image(array2grid(gt_samples))
                        accelerator.log(log_dict)
                    else:
                        sample_result = sample_sit(
                            model, xT, ys, args, vae,
                            latents_scale, latents_bias, accelerator
                        )
                        out_samples = accelerator.gather(sample_result["samples"].to(torch.float32))
                        gt_samples_decoded = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                        gt_samples_decoded = (gt_samples_decoded + 1) / 2.
                        gt_samples = accelerator.gather(gt_samples_decoded.to(torch.float32))
                        
                        accelerator.log({
                            "samples": wandb.Image(array2grid(out_samples)),
                            "gt_samples": wandb.Image(array2grid(gt_samples))
                        })
                logging.info("Generating EMA samples done.")

            current_lr = optimizer.param_groups[0]["lr"]
            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(), 
                "proj_loss": (accelerator.gather(proj_loss_mean).mean().detach().item()
                              if isinstance(proj_loss_mean, torch.Tensor) else proj_loss_mean),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
                "lr": current_lr,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


DEFAULTS = dict(
    # logging
    output_dir="exps",
    exp_name=None,
    logging_dir="logs",
    report_to="wandb",
    sampling_steps=10000,
    resume_step=0,
    # model
    model=None,
    num_classes=1000,
    encoder_depth=8,
    fused_attn=True,
    qk_norm=False,
    # dataset
    data_dir="../data/imagenet256",
    s3_cache_dir=None,
    resolution=256,
    batch_size=256,
    # precision
    allow_tf32=False,
    mixed_precision="fp16",
    # optimization
    epochs=1400,
    max_train_steps=400000,
    checkpointing_steps=10000,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_weight_decay=0.0,
    adam_epsilon=1e-08,
    max_grad_norm=1.0,
    # LR scheduler
    lr_scheduler_type="constant_with_linear_warmup",
    warmup_steps=0,
    # seed
    seed=0,
    # cpu
    num_workers=4,
    # loss
    path_type="linear",
    prediction="v",
    cfg_prob=0.1,
    enc_type="none",
    proj_coeff=0.0,
    weighting="uniform",
    legacy=False,
    # DFM
    stages_count=2,
    stage_weights=[0.9, 0.1],
    stage_sampling_thresholds=[0.1],
    num_steps_per_scale=[40, 10],
    # ELIT
    elit_max_mask_prob=0.0,
    elit_min_mask_prob=None,
    elit_group_size=4,
    elit_read_depth=1,
    elit_write_depth=1,
)


def build_parser():
    """Build argument parser with all defaults from DEFAULTS dict."""
    d = DEFAULTS
    parser = argparse.ArgumentParser(
        description="Unified Training for SiT, DFM-SiT, and ELIT-SiT.\n"
                    "Supports YAML config files: --config path/to/config.yaml\n"
                    "CLI args always override YAML values.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file. CLI args override YAML values.")

    # logging
    parser.add_argument("--output-dir", type=str, default=d["output_dir"])
    parser.add_argument("--exp-name", type=str, default=d["exp_name"])
    parser.add_argument("--logging-dir", type=str, default=d["logging_dir"])
    parser.add_argument("--report-to", type=str, default=d["report_to"])
    parser.add_argument("--sampling-steps", type=int, default=d["sampling_steps"])
    parser.add_argument("--resume-step", type=int, default=d["resume_step"])

    # model
    parser.add_argument("--model", type=str, default=d["model"],
                        choices=list(ALL_MODELS.keys()),
                        help="Model name. Prefix determines type: DFM-SiT-*/ELIT-SiT-*/SiT-*")
    parser.add_argument("--num-classes", type=int, default=d["num_classes"])
    parser.add_argument("--encoder-depth", type=int, default=d["encoder_depth"])
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=d["fused_attn"])
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=d["qk_norm"])

    # dataset
    parser.add_argument("--data-dir", type=str, default=d["data_dir"])
    parser.add_argument("--s3-cache-dir", type=str, default=d["s3_cache_dir"],
                        help="Local cache directory for S3 data. Only used when data-dir is an s3:// path.")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=d["resolution"])
    parser.add_argument("--batch-size", type=int, default=d["batch_size"])

    # precision
    parser.add_argument("--allow-tf32", action="store_true", default=d["allow_tf32"])
    parser.add_argument("--mixed-precision", type=str, default=d["mixed_precision"],
                        choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=d["epochs"])
    parser.add_argument("--max-train-steps", type=int, default=d["max_train_steps"])
    parser.add_argument("--checkpointing-steps", type=int, default=d["checkpointing_steps"])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=d["gradient_accumulation_steps"])
    parser.add_argument("--learning-rate", type=float, default=d["learning_rate"])
    parser.add_argument("--adam-beta1", type=float, default=d["adam_beta1"])
    parser.add_argument("--adam-beta2", type=float, default=d["adam_beta2"])
    parser.add_argument("--adam-weight-decay", type=float, default=d["adam_weight_decay"])
    parser.add_argument("--adam-epsilon", type=float, default=d["adam_epsilon"])
    parser.add_argument("--max-grad-norm", type=float, default=d["max_grad_norm"])

    # LR scheduler
    parser.add_argument("--lr-scheduler-type", type=str, default=d["lr_scheduler_type"],
                        choices=["constant", "constant_with_linear_warmup"],
                        help="LR scheduler type. 'constant' = no scheduling, "
                             "'constant_with_linear_warmup' = linear warmup then constant.")
    parser.add_argument("--warmup-steps", type=int, default=d["warmup_steps"],
                        help="Number of warmup steps for LR scheduler (0 = no warmup)")

    # seed
    parser.add_argument("--seed", type=int, default=d["seed"])

    # cpu
    parser.add_argument("--num-workers", type=int, default=d["num_workers"])

    # loss
    parser.add_argument("--path-type", type=str, default=d["path_type"], choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default=d["prediction"], choices=["v"])
    parser.add_argument("--cfg-prob", type=float, default=d["cfg_prob"])
    parser.add_argument("--enc-type", type=str, default=d["enc_type"])
    parser.add_argument("--proj-coeff", type=float, default=d["proj_coeff"])
    parser.add_argument("--weighting", type=str, default=d["weighting"])
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=d["legacy"])

    # DFM-specific
    parser.add_argument("--stages-count", type=int, default=d["stages_count"],
                        help="[DFM] Number of scales for multiscale decomposition")
    parser.add_argument("--stage-weights", type=float, nargs='+', default=d["stage_weights"],
                        help="[DFM] Weights for different stages to be sampled during training")
    parser.add_argument("--stage-sampling-thresholds", type=float, nargs='+',
                        default=d["stage_sampling_thresholds"],
                        help="[DFM] Thresholds for stage switching during inference sampling")
    parser.add_argument("--num-steps-per-scale", type=int, nargs='+',
                        default=d["num_steps_per_scale"],
                        help="[DFM] Number of sampling steps per scale")

    # ELIT-specific
    parser.add_argument("--elit-max-mask-prob", type=float, default=d["elit_max_mask_prob"],
                        help="[ELIT] Maximum masking probability for ELIT during training. "
                             "Use None to sample all valid budgets up to (1 - 1/window_tokens).")
    parser.add_argument("--elit-min-mask-prob", type=float, default=d["elit_min_mask_prob"],
                        help="[ELIT] Minimum masking probability. Defaults to elit_max_mask_prob (single budget). "
                             "If different from max, mask prob is uniformly sampled from valid levels in [min, max].")
    parser.add_argument("--elit-group-size", type=int, default=d["elit_group_size"],
                        help="[ELIT] Group size for ELIT token masking")
    parser.add_argument("--elit-read-depth", type=int, default=d["elit_read_depth"],
                        help="[ELIT] Depth of the read (cross-attention) layer")
    parser.add_argument("--elit-write-depth", type=int, default=d["elit_write_depth"],
                        help="[ELIT] Depth of the write (cross-attention) layer")

    return parser


def _normalize_key(key):
    """Convert between YAML underscore keys and argparse hyphen keys."""
    return key.replace('-', '_')


def parse_args(input_args=None):
    """
    Parse arguments with YAML + CLI support.
    Priority: CLI args  >  YAML config  >  defaults.
    """
    parser = build_parser()

    # First, do a preliminary parse just to get --config
    preliminary, _ = parser.parse_known_args(input_args)

    if preliminary.config is not None:
        with open(preliminary.config, 'r') as f:
            yaml_config = yaml.safe_load(f) or {}

        yaml_config = {_normalize_key(k): v for k, v in yaml_config.items()}

        merged = dict(DEFAULTS)
        merged.update(yaml_config)
        parser.set_defaults(**merged)

    args = parser.parse_args(input_args)

    if args.exp_name is None:
        parser.error("--exp-name is required (via CLI or YAML config)")
    if args.model is None:
        parser.error("--model is required (via CLI or YAML config)")

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

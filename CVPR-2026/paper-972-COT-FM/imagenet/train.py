
import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from sit import SiT_models
from loss import SILoss

from dataset import LMDBLatentsDataset
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
import math

logger = get_logger(__name__)


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

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # Set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
        
        # Log all args for reference
        logger.info("Training arguments:")
        for arg, value in sorted(args_dict.items()):
            logger.info(f"  {arg}: {value}")
            
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8
    
    # Define block_kwargs from args
    block_kwargs = {
        "fused_attn": False,
        "qk_norm": False,
    }

    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    # Create loss function with all RF parameters
    loss_fn = SILoss(
        path_type=args.path_type, 
        # Add RF specific parameters
        time_sampler=args.time_sampler,
        time_mu=args.time_mu,
        time_sigma=args.time_sigma,
        ratio_r_not_equal_t=args.ratio_r_not_equal_t,
        adaptive_p=args.adaptive_p,
        weighting=args.weighting,
        label_dropout_prob=args.cfg_prob,
        cfg_omega=args.cfg_omega,
        cfg_kappa=args.cfg_kappa,
        cfg_min_t=args.cfg_min_t,
        cfg_max_t=args.cfg_max_t,
        device=device,
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data:
    train_dataset = LMDBLatentsDataset(args.data_dir, flip_prob=0.5)

    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    steps_per_epoch = len(train_dataloader) // accelerator.gradient_accumulation_steps
    args.max_train_steps = args.epochs * steps_per_epoch // accelerator.num_processes
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            )
        try:
            model.load_state_dict(ckpt['model'])
            ema.load_state_dict(ckpt['ema'])
            optimizer.load_state_dict(ckpt['opt'])
            global_step = ckpt['steps']
        except:
            model.load_state_dict(ckpt)
            ema.load_state_dict(ckpt)
            global_step = args.resume_step

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    # here is a trick from IMM. https://github.com/lumalabs/imm/blob/main/training/encoders.py
    latents_scale = torch.tensor(
        [0.18125, 0.18125, 0.18125, 0.18125]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)
    for epoch in range(args.epochs):
        model.train()
        for moments, labels in train_dataloader:
            moments = moments.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                posterior = DiagonalGaussianDistribution(moments)
                x = posterior.sample()
                x = x * latents_scale + latents_bias
            
            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                loss, loss_ref = loss_fn(model, x, noises=None, model_kwargs=model_kwargs)
                loss_mean = loss.mean()
                loss_mean_ref = loss_ref.mean()
                loss = loss_mean                
                    
                ## optimization
                accelerator.backward(loss)
                grad_norm = 0.0
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model)
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0 or global_step >= args.max_train_steps:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(), 
                "loss_ref": accelerator.gather(loss_mean_ref).mean().detach().item(), 
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            progress_bar.set_postfix(**logs)
            
            # Log to file periodically
            if accelerator.is_main_process and global_step % 100 == 0:
                logger.info(f"Step {global_step}: loss = {logs['loss']:.4f}, grad_norm = {logs['grad_norm']:.4f}")

            if global_step >= args.max_train_steps:
                break
        
        # Log epoch completion
        if accelerator.is_main_process:
            logger.info(f"Completed epoch {epoch+1}/{args.epochs}")
            
        if global_step >= args.max_train_steps:
            break
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Training completed!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="RF Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str, default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)

    # dataset
    parser.add_argument("--data-dir", type=str, default="/path/to/train_sdvae_latents_lmdb")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=0)

    # basic loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--weighting", default="adaptive", type=str, choices=["uniform", "adaptive"], help="Loss weighting type")
    
    # RF specific parameters
    parser.add_argument("--time-sampler", type=str, default="logit_normal", choices=["uniform", "logit_normal"], 
                       help="Time sampling strategy")
    parser.add_argument("--time-mu", type=float, default=-0.4, help="Mean parameter for logit_normal distribution")
    parser.add_argument("--time-sigma", type=float, default=1.0, help="Std parameter for logit_normal distribution")
    parser.add_argument("--ratio-r-not-equal-t", type=float, default=0.75, help="Ratio of samples where r≠t")
    parser.add_argument("--adaptive-p", type=float, default=1.0, help="Power param for adaptive weighting")
    parser.add_argument("--cfg-omega", type=float, default=1.0, help="CFG omega param, default 1.0 means no CFG")
    parser.add_argument("--cfg-kappa", type=float, default=0.0, help="CFG kappa param for mixing")
    parser.add_argument("--cfg-min-t", type=float, default=0.0, help="Minum time for cfg trigger")
    parser.add_argument("--cfg-max-t", type=float, default=1.0, help="Maxium time for cfg trigger")
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

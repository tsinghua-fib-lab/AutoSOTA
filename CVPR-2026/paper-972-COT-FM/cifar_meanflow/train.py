# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import logging
import os
import sys
import time
from pathlib import Path

from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from models.model_configs import instantiate_model
from train_arg_parser import get_args_parser
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from training import distributed_mode
from training.data_transform import get_transform_cifar, get_transform_mnist
from training.eval_loop import eval_model
from training.load_and_save import load_model, save_model
from training.train_loop import train_one_epoch, train_step
from models.policy import Policy
from torchmetrics.aggregation import MeanMetric
import models.rng as rng
from tqdm import tqdm
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from models.cifar_tsne import cluster_dataloader
from torch.optim.lr_scheduler import CosineAnnealingLR
import training.ppo_utils as ppo_utils
torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)


def print_model(model):
    logger.info("=" * 91)
    num_params = 0
    for name, param in model.named_parameters():
        param_std = param.std().item()
        if param.requires_grad:
            num_params += param.numel()
            logger.info(f"{name:48} | {str(list(param.shape)):24} | std: {param_std:.6f}")
    logger.info("=" * 91)
    logger.info(f"Total params: {num_params}")


def get_data_loader(args, is_for_fid):
    if args.dataset == "cifar10":
        transforms = get_transform_cifar(is_for_fid)
        dataset = datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transforms,
        )
    elif args.dataset == "mnist":  # 3x32x32 MNIST for fast development
        transforms = get_transform_mnist()
        dataset = datasets.MNIST(
            root=args.data_path,
            train=True,
            download=True,
            transform=transforms,
        )
    else:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")

    logger.info(dataset)

    logger.info("Intializing DataLoader")
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    if is_for_fid:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            worker_init_fn=partial(rng.worker_init_fn, rank=global_rank),
            batch_size=100,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=not is_for_fid,  # for FID evaluation, we want to keep all samples
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            worker_init_fn=partial(rng.worker_init_fn, rank=global_rank),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=not is_for_fid,
        )
    logger.info(str(sampler))
    return data_loader

def _setup_ppo_config(config):
    """Initialize PPO configuration."""
    ppo_config = ppo_utils.PPOConfig()
    ppo_config.batch_size = config.batch_size
    ppo_config.device = config.device
    ppo_config.single_class_id = config.single_class_id if config.single_class_id >= 0 else None
    return ppo_config

def _initialize_noise_sampler(ppo_config, config):
    """Create and optionally load noise sampler."""
    noise_sampler = ppo_utils.NoiseSampler(ppo_config).to(config.device)
    checkpoint_path = ppo_config.noise_sampler_path
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
        noise_sampler.load_state_dict(checkpoint['policy_state_dict'])
        noise_sampler.eval()
        logging.info(f"Loaded pre-trained noise sampler from {checkpoint_path}")
    return noise_sampler

from training.optimal_transport import OTPlanSampler
from models.augment import AugmentPipe
import torchvision.transforms.functional as TF
otplansampler = OTPlanSampler(method='exact')
augment_pipe = AugmentPipe(p=0.12, xflip=1e8, yflip=0, scale=1, rotate_frac=0, aniso=1, translate_frac=1)  # turn off yflip and rotate
def create_training_dataset(ppo_config, args, ppo_resources, gaussians=None):
    """Create training dataset with OT + noise_sampler."""
    datasets = []
    logging.info("Creating training dataset with OT + noise_sampler")
    for class_id in tqdm(range(ppo_config.num_classes)):
        target = ppo_resources['ppo_data']['images'][ppo_resources['class_to_indices'][class_id]].to(args.device)
        target = torch.stack([
            TF.hflip(img) if torch.rand(1).item() < 0.5 else img
            for img in target
        ])
        target, aug_cond = rng.augment_with_rng_control(augment_pipe, target, args.seed, torch.rand) if args.use_edm_aug else (target, None)
        if gaussians is not None:
            gaussians["covs"][class_id] += torch.eye(gaussians["covs"][class_id].shape[0]) * 1e-5  # for numerical stability 
            dist = torch.distributions.MultivariateNormal(gaussians['means'][class_id].to(args.device), gaussians['covs'][class_id].to(args.device))
            noise = dist.sample((len(target), )).view_as(target)
        else:
            noise = torch.randn_like(target).to(args.device)
        noise, target, aug_cond = otplansampler.sample_plan(noise, target, aug_cond=aug_cond, sample_size=500)
        dataset = torch.utils.data.TensorDataset(noise.cpu(), target.cpu(), aug_cond.cpu() if aug_cond is not None else None)
        datasets.append(dataset)
    datasets = torch.utils.data.ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return dataloader




def main(args):
    distributed_mode.init_distributed_mode(args)

    print(f"Rank: {distributed_mode.get_rank()}")
    print(f"World Size: {distributed_mode.get_world_size()}")

    if distributed_mode.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logger.addHandler(logging.NullHandler())

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    if distributed_mode.is_main_process():
        # create tensorboard
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
        logger.info(f"Tensorboard writer created at {args.output_dir}")
    else:
        log_writer = None
        logger.info('Writer not created.')

    device = torch.device(args.device)

    # set the seeds
    seed = args.seed + distributed_mode.get_rank()  # legacy. TODO: rng.fold_in 
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    logger.info(f"Initializing Dataset: {args.dataset}")
    data_loader_fid = get_data_loader(args, is_for_fid=True)
    ppo_resources = ppo_utils.load_ppo_resources()
    ppo_config = _setup_ppo_config(args)
    rates = [len(ppo_resources['ppo_data']['images'][ppo_resources['class_to_indices'][class_id]]) for class_id in range(ppo_config.num_classes)]
    rates = torch.tensor(rates, dtype=torch.float)
    rates = rates / rates.sum()
    
    if os.path.exists(ppo_config.gaussian_path):
        gaussians = torch.load(ppo_config.gaussian_path)
    else:
        gaussians = None
    
    # define the model
    logger.info("Initializing Model")
    model = instantiate_model(args)

    model.to(device)
    # chkpt = "model.pth"
    chkpt = "./meanflow/tmp/checkpoint-last_2.6.pth"
    checkpoint = torch.load(chkpt, map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.net_ema.load_state_dict(model.net_ema1.state_dict())
    model.net.load_state_dict(model.net_ema.state_dict())

    model_without_ddp = model


    eff_batch_size = args.batch_size * distributed_mode.get_world_size()

    logger.info(f"Learning rate: {args.lr:.2e}")

    logger.info(f"Effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=False,
            broadcast_buffers=False,
            static_graph=True,
            gradient_as_bucket_view=True
        )
        model_without_ddp = model.module

    optimizer = torch.optim.Adam(  # Note: Adam, not AdamW
        model_without_ddp.net.parameters(),  # only the "net" parameters
        lr=args.lr,
        betas=args.optimizer_betas,
        weight_decay=0.0
    )
    warmup_iters = args.warmup_epochs * 50000 // eff_batch_size
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8 / args.lr, end_factor=1.0, total_iters=warmup_iters,)
    main_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, total_iters=args.epochs * 50000 // eff_batch_size, factor=1.0)
    lr_schedule = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_iters])

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    compiled_train_step = torch.compile(
        train_step,
        disable=not args.compile,
    )

    batch_loss = MeanMetric().to(device, non_blocking=True)
    batch_time = MeanMetric().to(device, non_blocking=True)
    batch_loss.reset()
    batch_time.reset()

    meters = {'batch_loss': batch_loss, 'batch_time': batch_time,}
    # if args.distributed:
    #     dist.destroy_process_group()
    #     exit()
    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        if not args.eval_only:
            data_loader_train = create_training_dataset(ppo_config, args, ppo_resources, gaussians=gaussians)
            if args.train_flow:
                train_one_epoch(
                    model=model,
                    compiled_train_step=compiled_train_step,
                    data_loader=data_loader_train,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    device=device,
                    epoch=epoch,
                    log_writer=log_writer,
                    args=args,
                    meters=meters,
                    noise_sampler=None
                )
            elif args.generate:
                ppo_utils.generate_samples(
                    model=model,
                    args=args,
                    rates=rates,
                )
                exit()
        # torch.save(noise_sampler.state_dict(), os.path.join(args.output_dir, f'output/policy_{epoch}.pth'))

        if args.output_dir and (
            (args.eval_frequency > 0 and (epoch+1) % args.eval_frequency == 0)
            or args.eval_only
            or args.test_run
        ):
            if not args.eval_only:
                save_model(
                    args=args,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    epoch=epoch,
                )
                logging.info(f"Saved checkpoint to {args.output_dir}")
            # Eval ema model:
            net_eval = model_without_ddp.net_ema
            ema_decay = net_eval.ema_decay
            eval_stats = eval_model(model, net_eval, data_loader_fid, device, epoch=epoch, args=args, suffix=f'_ema{ema_decay}', rates=rates)
            if log_writer is not None and "fid" in eval_stats:
                logging.info(f"Eval {epoch + 1} epochs finished: FID_ema{ema_decay}: {eval_stats['fid']}")
                # logging.info(f"Eval {epoch + 1} epochs finished: KID_ema{ema_decay}: {eval_stats['kid_mean']} ± {eval_stats['kid_std']}")
                # logging.info(f"Eval {epoch + 1} epochs finished: InceptionScore_ema{ema_decay}: {eval_stats['inception_score_mean']} ± {eval_stats['inception_score_std']}")
                log_writer.add_scalar(f"FID_ema{ema_decay}", eval_stats["fid"], epoch + 1)
        if args.test_run or args.eval_only:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

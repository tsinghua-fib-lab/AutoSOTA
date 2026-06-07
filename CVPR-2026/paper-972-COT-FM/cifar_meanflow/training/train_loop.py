# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import logging
import math
from typing import Iterable, Any, Callable
import time

import torch
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
import torch.distributed as dist
from models.augment import AugmentPipe
import models.rng as rng
from tqdm import tqdm


logger = logging.getLogger(__name__)


def synchronize_gradients(model: torch.nn.Module):
    """
    In a distributed setting, to enable jvp, we need to call model.module instead of model directly.
    If so, we synchronize gradients across all processes.
    """
    if not isinstance(model, DistributedDataParallel):
        return

    torch.cuda.synchronize()
    for param in model.module.parameters():
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= dist.get_world_size()


def gradient_sanity_check(model):
    if not isinstance(model, DistributedDataParallel):
        return
    torch.cuda.synchronize()
    # logging.info(f"Gradient sanity check ...")
    for name, p in model.module.named_parameters():
        if p.requires_grad and len(p.shape) > 3:
            monitor = p.grad.norm()

            monitor_list = [torch.zeros_like(monitor) for _ in range(dist.get_world_size())]
            dist.all_gather(monitor_list, monitor)
            monitor_tensor = torch.stack(monitor_list)
            # logging.info(f"All_gathered grad norm, param {name}: ")
            # for i, m in enumerate(monitor_tensor):
            #     logging.info(f"Rank {i}: {m:.16f}")
            # break

            # Assert all gradient norms are close to rank 0's
            ref = monitor_tensor[0]
            for i, m in enumerate(monitor_tensor):
                assert torch.isclose(m, ref), \
                    f"Gradient norm mismatch at rank {i}: {m} vs rank 0: {ref}"


def get_compiled_counts():
    metrics = torch._dynamo.utils.get_compilation_metrics()
    return len(metrics)


augment_pipe = AugmentPipe(p=0.12, xflip=1e8, yflip=0, scale=1, rotate_frac=0, aniso=1, translate_frac=1)  # turn off yflip and rotate

def train_step(model_without_ddp, *args, **kwargs):
    loss = model_without_ddp.forward_with_loss(*args, **kwargs)
    # loss.backward(create_graph=False)
    return loss


def train_one_epoch(
    model: torch.nn.Module,
    compiled_train_step: Callable,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    log_writer: Any,
    args: argparse.Namespace,
    meters: dict[str, MeanMetric],
    noise_sampler = None
):
    gc.collect()
    model.train(True)

    batch_loss = meters['batch_loss'] 
    batch_time = meters['batch_time']

    # declare the unwrapped model
    model_without_ddp = model if not isinstance(model, DistributedDataParallel) else model.module
    accum_steps = getattr(args, "accum_steps", 1)
    tic = time.time()
    data_iter_step = 0
    for (noises, samples, aug_cond) in tqdm(data_loader):
        steps = data_iter_step + len(data_loader) * epoch  # global step

        optimizer.zero_grad()            
        if data_iter_step > 0 and args.test_run:
            break
        if noise_sampler is not None:
            samples = samples.permute(0, 3, 1, 2)
        samples = samples.to(device, non_blocking=True)
        aug_cond = aug_cond.to(device, non_blocking=True) if aug_cond is not None else None
        noises = noises.to(device, non_blocking=True) if noises is not None else None
        # samples = samples * 2.0 - 1.0

        # samples, aug_cond = rng.augment_with_rng_control(augment_pipe, samples, args.seed, steps) if args.use_edm_aug else (samples, None)

        if args.compile and epoch == args.start_epoch and data_iter_step == 0:
            logging.info(f"Compiling the first train step, this may take a while...")
        
        loss = rng.train_step_with_rng_control(compiled_train_step, model_without_ddp, steps, args.seed, samples, aug_cond, noise=noises, noise_sampler=noise_sampler)
        if args.compile:
            assert get_compiled_counts() > 0, "Compilation not triggered."


        if isinstance(model, DistributedDataParallel) and (data_iter_step + 1) % accum_steps != 0:
            context = model.no_sync()
        else:
            import contextlib
            context = contextlib.nullcontext()

        with context:
            loss = rng.train_step_with_rng_control(
                compiled_train_step, model_without_ddp, steps, args.seed,
                samples, aug_cond, noise=noises, noise_sampler=noise_sampler
            )
            loss = loss / accum_steps
            loss.backward(create_graph=False)
        # sanity check
        # synchronize_gradients(model)  # To support compiling, we need to call model.module and then sync gradients.
        # if (epoch - args.start_epoch) % 100 == 0 and data_iter_step < 2:  # sanity check after the first steps
        #     gradient_sanity_check(model)

        # loss_value = loss.item()
        # batch_loss.update(loss_value)
        loss_value = loss.item()
        batch_loss.update(loss_value)

        if (data_iter_step + 1) % accum_steps == 0:
            synchronize_gradients(model)
            optimizer.step()
            optimizer.zero_grad()
            model_without_ddp.update_ema()

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        # update the parameters
        # optimizer.step()
        # model_without_ddp.update_ema()  # moved to begin of train_step

        # logging
        toc = time.time()
        batch_time.update(toc - tic)
        tic = toc

        lr = optimizer.param_groups[0]["lr"]
        lr_schedule.step()  # per-iteration lr
        data_iter_step += 1
        if (steps + 1) % args.log_per_step == 0:
            loss_ave = batch_loss.compute().detach().cpu().numpy() # logging only
            sec_per_iter = batch_time.compute()
            batch_time.reset()
            batch_loss.reset()
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: loss = {loss_ave:.6f}, lr = {lr:.6f}, steps = {steps}, sec_per_iter = {sec_per_iter:.4f}"
            )
            epoch_1000x = int(steps / len(data_loader) * 1000)
            metrics = {
                "loss": loss_ave,
                "lr": lr,
                "epoch": steps / len(data_loader),
                "steps": steps,
                "sec_per_iter": sec_per_iter,
            }
            if log_writer is not None:
                for k, v in metrics.items():
                    log_writer.add_scalar(f"ep_{k}", v, epoch_1000x)  # we use epoch * 1000 to plot, for calibrating different batch sizes

    return

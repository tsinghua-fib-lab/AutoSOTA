import os
from omegaconf import OmegaConf
import copy
import pickle
import hydra
from hydra.utils import instantiate

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator

from utils.helper import create_logger, count_parameters, update_ema, unwrap_model


@hydra.main(version_base="1.3", config_path="configs/pretrain", config_name="navier-stokes")
def main(config):
    if config.train.tf32:
        torch.set_float32_matmul_precision("high")
    wandb_log = "wandb" if config.log.wandb else None
    accelerator = Accelerator(log_with=wandb_log)
    if config.log.wandb:
        wandb_init_kwargs = {"project": config.log.project, "group": config.log.group}
        accelerator.init_trackers(
            config.log.project,
            config=OmegaConf.to_container(config),
            init_kwargs=wandb_init_kwargs,
        )

    exp_dir = os.path.join(config.log.exp_dir, config.log.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_dir = os.path.join(exp_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = create_logger(exp_dir, main_process=accelerator.is_main_process)
    logger.info(f"Experiment dir created at {exp_dir}")

    # dataset

    dataset = instantiate(config.data)

    batch_size = config.train.batch_size // accelerator.num_processes
    assert (
        batch_size * accelerator.num_processes == config.train.batch_size
    ), "Batch size must be divisible by num processes"

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    logger.info(f"Dataset loaded with {len(dataset)} samples")
    # construct loss function
    loss_fn = instantiate(config.loss)

    # build model
    net = instantiate(config.model)

    logger.info(f"Number of parameters: {count_parameters(net)}")

    ema_net = copy.deepcopy(net).eval().requires_grad_(False).to(accelerator.device)

    # optimizer
    warmup_steps = config.train.warmup_steps
    optimizer = torch.optim.Adam(net.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(step, warmup_steps) / warmup_steps
    )

    # load checkpoints
    if config.train.resume != 'None':
        checkpoint = torch.load(config.train.resume)
        net.load_state_dict(checkpoint["net"])
        ema_net.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info(f"Resuming from checkpoint {config.train.resume}")
        start_steps = int(os.path.basename(config.train.resume).split(".")[0].split("_")[-1])
    else:
        start_steps = 0
    logger.info(f"Starting from step {start_steps}")
    num_steps = config.train.num_steps - start_steps
    num_epochs = config.train.batch_size * num_steps // len(dataloader) + 1
    ema_rampup_ratio = config.train.ema_rampup_ratio
    # prepare accelerator
    net, dataloader, optimizer, scheduler = accelerator.prepare(
        net, dataloader, optimizer, scheduler
    )
    # net = torch.compile(net)

    training_steps = start_steps
    # training loop
    for e in range(num_epochs):
        for imgs in dataloader:
            if training_steps >= config.train.num_steps:
                break
            optimizer.zero_grad()
            loss = loss_fn(net, imgs)
            loss = torch.mean(loss)
            accelerator.backward(loss)
            if accelerator.sync_gradients and config.train.grad_clip > 0.0:
                accelerator.clip_grad_norm_(
                    net.parameters(), max_norm=config.train.grad_clip
                )
            optimizer.step()
            scheduler.step()

            ema_halflife_nimg = config.train.ema_halflife_nimg
            curr_nimg = training_steps * config.train.batch_size
            ema_halflife_nimg = min(ema_halflife_nimg, curr_nimg * ema_rampup_ratio)
            ema_decay = 0.5 ** (config.train.batch_size / max(ema_halflife_nimg, 1))

            # update ema model
            raw_net = unwrap_model(net)
            update_ema(ema_net, raw_net, decay=ema_decay)

            accelerator.log({"loss": loss})
            if training_steps % config.log.log_every == 0:
                logger.info(f"Step {training_steps}: loss {loss.item():.4f}")
            if accelerator.is_main_process and training_steps % config.log.save_every == 0 and training_steps > 0:
                save_dict = {
                    "net": raw_net.state_dict(),
                    "ema": ema_net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                with open(os.path.join(ckpt_dir, f"ema_{training_steps}.pkl"), "wb") as f:
                    pickle.dump({"ema": ema_net}, f)
                torch.save(
                    save_dict, os.path.join(ckpt_dir, f"ckpt_{training_steps}.pt")
                )
                logger.info(f"Checkpoint saved at {ckpt_dir}/ckpt_{training_steps}.pt")
            training_steps += 1

    accelerator.end_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()

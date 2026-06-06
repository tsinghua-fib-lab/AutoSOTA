import argparse
import os
from omegaconf import OmegaConf
import wandb
import torch
from trainer import GANTrainer, ScoreDistillationVaceTrainer, DiffusionTrainer, ODETrainer
from utils.distributed import launch_distributed_job, setup_for_distributed, init_distributed_mode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="wandb_logs", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    # get the filename of config_path
    config_name = args.logdir.split("/")[-1]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb

    # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    launch_distributed_job()
    rank = int(os.environ["RANK"])
    setup_for_distributed(rank == 0)
    # init_distributed_mode()

    print(config)
    # quit()
    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "gan":
        trainer = GANTrainer(config)
    elif config.trainer == "ode":
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation_vace":
        trainer = ScoreDistillationVaceTrainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()


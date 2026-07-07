import logging
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric import seed_everything

from graphgym.cmd_args import parse_args
from graphgym.config import cfg, dump_cfg, load_cfg, set_run_dir, set_out_dir
from graphgym.loader import create_dataset, create_loader
from graphgym.logger import create_logger, setup_printing
from graphgym.model_builder import create_model
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.register import train_dict
from graphgym.train_parallel import train
from graphgym.utils.agg_runs import agg_runs
from graphgym.utils.comp_budget import params_count
from graphgym.utils.device import auto_select_device

def setup(rank, world_size, gpu_ids):
    # Initialize the process group for distributed training
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set CUDA device based on gpu_ids argument
    torch.cuda.set_device(gpu_ids[rank])  # Assign the correct GPU for this process

def cleanup():
    dist.destroy_process_group()

def run_experiment(rank, world_size, args, gpu_ids):
    setup(rank, world_size, gpu_ids)

    # Set run directory and load config
    load_cfg(cfg, args)
    set_out_dir(cfg.out_dir, args.cfg_file)

    if torch.cuda.is_available():
        # Select the GPU to use based on the process rank and gpu_id
        cfg.device = 'cuda:{}'.format(gpu_ids[rank])  # Each process gets its own GPU
    else:
        cfg.device = 'cpu'

    dump_cfg(cfg)

    for i in range(args.repeat):
        set_run_dir(cfg.out_dir)
        setup_printing()
        cfg.seed = cfg.seed + 1
        seed_everything(cfg.seed)

        # Create dataset, loader, logger
        datasets = create_dataset()
        loaders = create_loader(datasets)
        loggers = create_logger()

        # Create model and wrap with DDP
        model = create_model().to(cfg.device)
        gpu_id = int(cfg.device.split(':')[-1])
        model = DDP(model, device_ids=[gpu_id])

        # Create optimizer and scheduler
        optimizer = create_optimizer(model.parameters())
        scheduler = create_scheduler(optimizer)
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training
        if cfg.train.mode == 'standard':
            train(loggers, loaders, model, optimizer, scheduler)
        else:
            raise NotImplementedError
            #train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
            #                           scheduler)
    # Aggregate results from different seeds
    agg_runs(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
    cleanup()

if __name__ == '__main__':
    args = parse_args()

    # Define GPU IDs based on the `gpu_id` argument
    gpu_id_start = args.gpu_id  # Start GPU ID
    gpu_ids = [gpu_id_start, gpu_id_start + 1]  # Use two GPUs per experiment

    world_size = 2  # Number of GPUs per experiment (2 GPUs)

    # Spawn processes, passing in gpu_ids to control which GPUs are used
    mp.spawn(run_experiment,
             args=(world_size, args, gpu_ids),
             nprocs=world_size,
             join=True)

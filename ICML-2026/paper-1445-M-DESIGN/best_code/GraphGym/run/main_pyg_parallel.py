import logging
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch_geometric import seed_everything

from graphgym.cmd_args import parse_args
from graphgym.config import cfg, dump_cfg, load_cfg, set_run_dir, set_out_dir
from graphgym.loader_pyg import create_dataset, create_loader
from graphgym.logger import create_logger, setup_printing
from graphgym.model_builder_pyg import create_model
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.register import train_dict
from graphgym.train_pyg import train
from graphgym.utils.agg_runs import agg_runs
from graphgym.utils.comp_budget import params_count


def run(rank: int, world_size: int, dataset):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    loaders = create_loader(dataset, rank, world_size)
    model = create_model().to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
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
        train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                    scheduler)

if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)
    set_out_dir(cfg.out_dir, args.cfg_file)
    # Set Pytorch environment
    if cfg.device != 'cpu' and torch.cuda.is_available():
        cfg.device = 'cuda:{}'.format(args.gpu_id)
    else:
        cfg.device = 'cpu'
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    world_size = min(torch.cuda.device_count(), 2)
    
    # Repeat for different random seeds
    for i in range(args.repeat):
        set_run_dir(cfg.out_dir)
        setup_printing()
        # Set configurations for each run
        cfg.seed = cfg.seed + 1
        seed_everything(cfg.seed)

        # Set machine learning pipeline
        dataset = create_dataset()
        loggers = create_logger()

        mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)

    # Aggregate results from different seeds
    agg_runs(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')

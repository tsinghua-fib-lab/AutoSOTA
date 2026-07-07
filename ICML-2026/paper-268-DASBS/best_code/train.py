import numpy as np
from datetime import datetime
import random
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta
import os
from utils_train import train
from model import get_model, EMA
from utils import Logger, model_info
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
if not OmegaConf.has_resolver('eval'): OmegaConf.register_new_resolver('eval', eval)
import traceback


@hydra.main(config_path="configs", config_name="ising4.yaml", version_base="1.1")
def main(args):
    OmegaConf.set_struct(args, False) # Allow dynamic updates to args

    # Set up working directory and run name
    if not args.logging.on_slurm: # on slurm system, run_name is set in loop.sh
        if args.ckpt.path is None:
            if args.logging.run_name is None:
                args.logging.run_name = datetime.now().strftime("%m%d%H%M%S")
        else:
            args.logging.run_name = os.path.basename(os.path.dirname(args.ckpt.path))
            # Reminder: provide WANDB_RUN_ID env variable for logging resumption on wandb

    os.makedirs(os.path.join(args.logging.dir), exist_ok=True)
    print(f">>> Working directory: {args.logging.dir}")

    # Save configs of current run
    config_dir = os.path.join(args.logging.dir, f".hydra_{datetime.now().strftime('%m%d%H%M%S')}")
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=args, f=f)
    hydra_cfg = HydraConfig.get()
    with open(os.path.join(config_dir, "hydra.yaml"), "w") as f:
        OmegaConf.save(config=hydra_cfg, f=f)
    with open(os.path.join(config_dir, "overrides.yaml"), "w") as f:
        f.write("\n".join(hydra_cfg.overrides.task))
    print(f">>> Configs saved to {config_dir}")

    # Reminder: if not on slurm system, always set CUDA_VISIBLE_DEVICES before run!
    args.world_size = world_size = torch.cuda.device_count()
    print(f">>> Using {world_size} GPU(s) for training")
    if world_size > 1:
        port = int(np.random.randint(10000, 20000))
        mp.set_start_method("forkserver")
        mp.spawn(run_multiprocess, args=(world_size, args, port), nprocs=world_size, join=True)
    else:
        run_multiprocess(0, 1, args, None)


def run_multiprocess(rank, world_size, args, port):
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=30))
        torch.cuda.set_device(rank)
    args.device = f'cuda:{rank}'
    
    torch.backends.cudnn.benchmark = True
    if args.seed is not None:
        seed = args.seed + rank
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
    
    logger = Logger(args, rank=rank)

    controller = get_model(args, require_time=True).to(args.device)
    corrector = get_model(args, require_time=False).to(args.device)
    optim_controller = torch.optim.AdamW(controller.parameters(), lr=args.optim.lr, weight_decay=args.optim.wd)
    optim_corrector = torch.optim.AdamW(corrector.parameters(), lr=args.optim.lr, weight_decay=args.optim.wd)
    ema_controller = EMA(controller.parameters(), decay=args.ema.decay)
    ema_corrector = EMA(corrector.parameters(), decay=args.ema.decay)

    logger.info(model_info(controller, name='Controller'))
    logger.info(model_info(corrector, name='Corrector'))

    if args.ckpt.path is not None:
        ckpt = torch.load(args.ckpt.path, weights_only=False, map_location=args.device)
        controller.load_state_dict(ckpt['controller_state_dict'])
        corrector.load_state_dict(ckpt['corrector_state_dict'])
        ema_controller = EMA(controller.parameters(), decay=args.ema.decay)
        ema_corrector = EMA(corrector.parameters(), decay=args.ema.decay)
        if args.ckpt.load_optim:
            optim_controller.load_state_dict(ckpt['optim_controller_state_dict'])
            optim_corrector.load_state_dict(ckpt['optim_corrector_state_dict'])
        if args.ckpt.load_ema:
            ema_controller.load_state_dict(ckpt['ema_controller_state_dict'])
            ema_corrector.load_state_dict(ckpt['ema_corrector_state_dict'])
        logger.info(f'Checkpoint loaded from {args.ckpt.path}')

        if not args.ckpt.start_from_zero:
            # ckpt.path should be like xxx/ckpt?.pth where ? starts from 1
            args.start_stage = int(args.ckpt.path.split('ckpt')[-1].split('.pth')[0])
            logger.info(f'Resume training from stage {args.start_stage}')
            args.step_count_controller = ckpt['args']['step_count_controller']
            args.step_count_corrector = ckpt['args']['step_count_corrector']
            logger.info(f'''Existing step counts - controller: {
                args.step_count_controller}, corrector: {args.step_count_corrector}''')
        else:
            args.start_stage = args.step_count_controller = args.step_count_corrector = 0
    else:
        args.start_stage = args.step_count_controller = args.step_count_corrector = 0

    if world_size > 1:
        controller = DDP(controller, device_ids=[rank], static_graph=True, find_unused_parameters=True)
        corrector = DDP(corrector, device_ids=[rank], static_graph=True, find_unused_parameters=True)

    try:
        train(controller, corrector, optim_controller, optim_corrector,
              ema_controller, ema_corrector, args, logger)
        logger.close()
    except Exception as e:
        error_info = traceback.format_exc()
        logger.info(f">>> Training failed with error:\n{error_info}")
        logger.close(exit_code=1)
        raise e
    

if __name__ == "__main__":
    main()

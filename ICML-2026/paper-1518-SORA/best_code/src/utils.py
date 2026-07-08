import torch
import numpy as np
import random
import os
from pathlib import Path
from architectures.get_model import get_model
from training.utils import get_optimizer, get_scheduler

def set_seed(seed: int):
    """
    Seed setting for result reproducibility. 
    Set the random seed for python, numpy, torch (cpu and cuda).
    
    Args:
        seed (int): Seed for the random number generator.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed set to {seed} for random, numpy, and torch (CPU & CUDA).")

def create_directories(args):
    """
    Create necessary directories for the project.
    
    Args:
        root_path (str): Path to the root directory of the project.
    """
    Path(f'{args.root_path}').mkdir(parents=True, exist_ok=True)
    Path(f'{args.root_path}/Datasets/{args.dataset}').mkdir(parents=True, exist_ok=True)
    Path(f'{args.root_path}/Results/{args.dataset}').mkdir(parents=True, exist_ok=True)
    Path(f'{args.root_path}/Results/{args.dataset}/{args.model}').mkdir(parents=True, exist_ok=True)
    Path(f'{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}').mkdir(parents=True, exist_ok=True)
    Path(f'{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/raw_results_{args.seed}').mkdir(parents=True, exist_ok=True)
    Path(f'{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/plots_{args.seed}').mkdir(parents=True, exist_ok=True)
    Path(f'{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/checkpoints_{args.seed}').mkdir(parents=True, exist_ok=True)
    Path(f'{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/final_checkpoints_{args.seed}').mkdir(parents=True, exist_ok=True)
    # Path(f'{args.root_path}/{args.dataset}/data').mkdir(parents=True, exist_ok=True)

def get_device(device_name):
    """
    Get the device to use for the training.
    
    Args:
        device_name (str): Name of the device to use.
    """
    match device_name:
        case "cuda":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Running on {torch.cuda.get_device_name(0)}")
        case "cpu":
            device = torch.device('cpu')
            print(f"Running on CPU")
        case _:
            raise ValueError("Invalid Device!")
    return device

def save_checkpoint(model, optimizer, scheduler, path:str):
    """
    Save the model checkpoint.
    
    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler to save.
        path (str): Path to save the checkpoint.
    """
    torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None
                   }, path)

def load_checkpoint(args, path:str, num_classes:int, image_size:int, num_channels:int, len_trainloader, device):
    """
    Load the model checkpoint.
    
    Args:
        args (argparse.Namespace): Arguments for the training.
        path (str): Path to load the checkpoint.
        num_classes (int): Number of classes in the dataset.
        len_trainloader (int): Length of the training data loader.
        device (torch.device): Device to use for the training.
    """
    model = get_model(args.model, num_classes, image_size, num_channels)
    model.to(device)
    
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer, len_trainloader)

    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return model, optimizer, scheduler

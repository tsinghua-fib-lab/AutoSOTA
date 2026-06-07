import os
os.environ["OMP_NUM_THREADS"] = "1" # Limit OpenMP (NumPy, MKL)
os.environ["MKL_NUM_THREADS"] = "1" # Limit MKL operations
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

os.environ["OMP_PLACES"] = "cores"
os.environ["OMP_PROC_BIND"] = "spread"

import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_descriptor")   # avoid big /dev/shm segments

from lib.core.config import cfg, update_config
from lib.models.model import FECO
from lib.core.base import compute_contact_loss
from data.dataset import MultipleDatasets
from lib.utils.train_utils import get_optim_groups, get_transform, worker_init_fn, set_seed
from lib.utils.log_utils import get_datetime
from lib.utils.train_utils import move_to_device, infinite_loader, load_training_setup


parser = argparse.ArgumentParser(description='Train FECO')
parser.add_argument('--backbone', type=str, default='vit-h-14', choices=['vit-h-14', 'vit-l-16', 'vit-b-16', 'vit-s-16', 'resnet-152', 'resnet-101', 'resnet-50', 'resnet-34', 'resnet-18'], help='backbone model')
parser.add_argument('--resume_training', type=str, default='', help='path to checkpoint to strictly resume from')
args = parser.parse_args()


# Import dataset
for i in range(len(cfg.DATASET.train_name)):
    exec(f'from data.{cfg.DATASET.train_name[i]}.dataset import {cfg.DATASET.train_name[i]}')
for i in range(len(cfg.DATASET.train_shoe_style_name)):
    exec(f'from data.{cfg.DATASET.train_shoe_style_name[i]}.dataset import {cfg.DATASET.train_shoe_style_name[i]}')


# Set device as CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(cfg.DATASET.workers) # Limit Torch
torch.set_num_interop_threads(1) # Limit Torch


# Initialize directories
dataset_name = "_".join([name.lower() for name in (cfg.DATASET.train_name)])

resume_checkpoint_path = None
if args.resume_training:
    resume_checkpoint_path = os.path.abspath(args.resume_training)

experiment_dir = os.path.join(f'experiments_train_{dataset_name}', 'full', f'exp_{get_datetime()}')
checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)


# Load config
update_config(backbone_type=args.backbone, exp_dir=experiment_dir)


# Set seed for minimal reproducibility
from lib.core.config import logger
set_seed(cfg.MODEL.seed)
logger.info(f"Using random seed: {cfg.MODEL.seed}")
logger.info(f"!!!!!!!!!!!!!!! Max epochs {cfg.TRAIN.epoch}")


############## Dataset ###############
transform = get_transform(args.backbone)

# Foot contact dataset
train_datasets = []
for i in range(len(cfg.DATASET.train_name)):
    train_datasets.append(eval(f'{cfg.DATASET.train_name[i]}')(transform, 'train'))
train_datasets = MultipleDatasets(train_datasets, make_same_len=False)

# Shoe dataset
train_shoe_datasets = []
for i in range(len(cfg.DATASET.train_shoe_style_name)):
    train_shoe_datasets.append(eval(f'{cfg.DATASET.train_shoe_style_name[i]}')(transform, 'train'))
train_shoe_datasets = MultipleDatasets(train_shoe_datasets, make_same_len=False)
############## Dataset ###############


############# Dataloader #############
train_dataloader = DataLoader(train_datasets, batch_size=cfg.TRAIN.batch, shuffle=True, num_workers=cfg.DATASET.workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=2, multiprocessing_context="fork")
train_shoe_dataloader = DataLoader(train_shoe_datasets, batch_size=cfg.TRAIN.batch, shuffle=True, num_workers=cfg.DATASET.workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=2, multiprocessing_context="fork")
############# Dataloader #############


logger.info(f"# of train batch: {len(train_dataloader)}")


############# Model #############
model = FECO().to(device)
############# Model #############


############# Optmizer #############
# Optimization group
optim_groups = get_optim_groups(model)

# Optimizer
optimizer = torch.optim.AdamW(optim_groups, lr=cfg.TRAIN.lr, betas=cfg.TRAIN.betas, weight_decay=cfg.TRAIN.weight_decay)

# Scheduler
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.milestones, gamma=cfg.TRAIN.gamma)
############# Optmizer #############


############################### Train Loop ###############################
# Set train parameters
best_checkpoint_path = ''
start_epoch = 0
global_step = 0

# Load optimizer and scheduler if resume training
start_epoch, global_step, best_checkpoint_path = load_training_setup(args.resume_training, model, optimizer, scheduler, device, logger)

# Start training
for epoch in range(start_epoch, cfg.TRAIN.epoch):
    # Make dataloader as iterator
    train_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)
    train_shoe_iterator = infinite_loader(train_shoe_dataloader)

    # Make model trainable
    torch.set_grad_enabled(True)
    model.train()

    # Iterate over samples to train the model
    for idx, data in train_iterator:
        # Move data to CUDA
        move_to_device(data, device)
        shoe_data = next(train_shoe_iterator)

        ############# Run model #############
        contact_data = {'input': data['input_data'], 'input_shoe': shoe_data['input_data'] ,'target': data['targets_data'], 'meta_info': data['meta_info']}
        outputs = model(contact_data, mode="train")
        ############# Run model #############

        ############# Loss Function #############
        train_loss, loss_dict = compute_contact_loss(outputs, data['targets_data'])
        contact_loss, style_contact_loss, adv_contact_loss, ground_loss, style_ground_loss = loss_dict['main_contact_loss'], loss_dict['style_contact_loss'], loss_dict['adv_contact_loss'], loss_dict['ground_loss'], loss_dict['style_ground_loss']
        mask_loss, ph_loss, normal_loss = loss_dict['foot_mask_loss'], loss_dict['pixel_height_loss'], loss_dict['ground_normal_loss']
        ############# Loss Function #############

        train_iterator.set_description(f"Epoch: {epoch} | Train Loss: {train_loss:.3f} | Contact Loss: {contact_loss:.3f} | Style Loss: {style_contact_loss:.3f} | Adv Loss: {adv_contact_loss:.3f} | Ground Loss: {ground_loss:.3f} | Style Ground Loss: {style_ground_loss:.3f}")        

        ############# Training process #############
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        ############# Training process #############

    # Scheduler
    scheduler.step()
    global_step += 1
    ############################### Training Loop ###############################


    ############# Save model checkpoint #############
    if epoch % cfg.TRAIN.print_freq == 0 or epoch == (cfg.TRAIN.epoch - 1):
        checkpoint_path = os.path.join(checkpoint_dir, f"feco_full_epoch{epoch}.ckpt")
        checkpoint_out = {
            'epoch': epoch,
            'global_step': global_step,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(checkpoint_out, checkpoint_path)
        best_checkpoint_path = checkpoint_path
        logger.info(f"Model trained, best model path: {best_checkpoint_path}")
    ############# Save model checkpoint #############
    

    logger.info(f"Epoch: {epoch} | Train Loss: {train_loss:.3f} | Contact Loss: {contact_loss:.3f} | Style Loss: {style_contact_loss:.3f} | Adv Loss: {adv_contact_loss:.3f} | Ground Loss: {ground_loss:.3f} | Style Ground Loss: {style_ground_loss:.3f}")
############################### Train Loop ###############################


# Let us know that training is finally finished
logger.info('Model Training Finished!!!!!')
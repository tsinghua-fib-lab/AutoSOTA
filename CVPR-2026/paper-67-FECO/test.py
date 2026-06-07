import os
os.environ["OMP_NUM_THREADS"] = "1" # Limit OpenMP (NumPy, MKL)
os.environ["MKL_NUM_THREADS"] = "1" # Limit MKL operations
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

os.environ["OMP_PLACES"] = "cores"
os.environ["OMP_PROC_BIND"] = "spread"

import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from lib.core.config import cfg, update_config
from lib.models.model import FECO
from lib.utils.contact_utils import get_contact_thres
from lib.utils.train_utils import get_transform, worker_init_fn, set_seed
from lib.utils.eval_utils import evaluation


parser = argparse.ArgumentParser(description='Test FECO')
parser.add_argument('--backbone', type=str, default='vit-h-14', choices=['vit-h-14', 'vit-l-16', 'vit-b-16', 'vit-s-16', 'resnet-152', 'resnet-101', 'resnet-50', 'resnet-34', 'resnet-18'], help='backbone model')
parser.add_argument('--test_name', type=str, default='MMVP', help='dataset name for evaluation')
parser.add_argument('--checkpoint', type=str, default='', help='model path for evaluation')
args = parser.parse_args()


# Import dataset
exec(f'from data.{args.test_name}.dataset import {args.test_name}')


# Set device as CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(cfg.DATASET.workers) # Limit Torch
torch.set_num_interop_threads(cfg.DATASET.workers) # Limit Torch


# Initialize directories
experiment_dir = f'experiments_test_{args.test_name.lower()}'
checkpoint_dir = os.path.join(experiment_dir, 'full', 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)


# Load config
update_config(backbone_type=args.backbone, test_name=args.test_name, exp_dir=experiment_dir)


# Set seed for reproducibility
from lib.core.config import logger
set_seed(cfg.MODEL.seed)
logger.info(f"Using random seed: {cfg.MODEL.seed}")


############## Dataset ###############
transform = get_transform(args.backbone)

test_dataset = eval(f'{args.test_name}')(transform, 'test')
############## Dataset ###############


############# Dataloader #############
test_dataloader = DataLoader(test_dataset, batch_size=cfg.TEST.batch, shuffle=False, num_workers=cfg.DATASET.workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
############# Dataloader #############


logger.info(f"# of test samples: {len(test_dataset)}")


############# Model #############
model = FECO().to(device)
model.eval()
############# Model #############


# Load model checkpoint if provided
if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])


############################### Test Loop ###############################
eval_result = {
    'cont_pre': [None for _ in range(len(test_dataset))],
    'cont_rec': [None for _ in range(len(test_dataset))],
    'cont_f1': [None for _ in range(len(test_dataset))],
    }

test_iterator = tqdm(enumerate(test_dataloader), total=len(test_dataloader), leave=False)
model.eval()


for idx, data in test_iterator:
    ############# Run model #############
    with torch.no_grad():
        outputs = model({'input': data['input_data'], 'target': data['targets_data'], 'meta_info': data['meta_info']}, mode="test")
    ############# Run model #############
    

    ############## Evaluation ###############
    # Compute evaluation metrics
    eval_thres = get_contact_thres(args.backbone)
    eval_out = evaluation(outputs, data['targets_data'], data['meta_info'], mode='test', thres=eval_thres)
    for key in [*eval_out]:
        eval_result[key][idx] = eval_out[key]

    # Hand Contact Estimator (HCE)
    total_cont_pre = np.mean([x if x is not None else 0.0 for x in eval_result['cont_pre'][:idx+1]])
    total_cont_rec = np.mean([x if x is not None else 0.0 for x in eval_result['cont_rec'][:idx+1]])
    total_cont_f1  = np.mean([x if x is not None else 0.0 for x in eval_result['cont_f1'][:idx+1]])
    ############## Evaluation ###############


    logger.info(f"C-Pre: {total_cont_pre:.3f} | C-Rec: {total_cont_rec:.3f} | C-F1: {total_cont_f1:.3f}")
############################### Test Loop ###############################


# Let us know that test is finally finished
logger.info('Test finished!!!!')
logger.info(f"Final Results --- C-Pre: {total_cont_pre:.3f} | C-Rec: {total_cont_rec:.3f} | C-F1: {total_cont_f1:.3f}")
import argparse
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from utils import *
from conformal import ConformalModel
from utils_ori import *
from conformal_ori import ConformalModel_ori
import torch.backends.cudnn as cudnn
import random
from PIL import ImageFile
import copy
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define the seed_torch function
def seed_torch(seed):
    """Fix the randomness for PyTorch."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

parser = argparse.ArgumentParser(description='Conformalize Torchvision Model on Imagenet')
parser.add_argument('data', metavar='IMAGENETVALDIR', help='path to Imagenet Val')
parser.add_argument('--batch_size', metavar='BSZ', help='batch size', default=32)
parser.add_argument('--num_workers', metavar='NW', help='number of workers', default=0)
parser.add_argument('--num_calib', metavar='NCALIB', help='number of calibration points', default=10000)
parser.add_argument('--seed', metavar='SEED', help='random seed', default=[0, 1, 2, 3, 4])

if __name__ == "__main__":
    args = parser.parse_args()

    Raps_top1_all, Raps_top5_all, Raps_coverage_all, Raps_size_all = [], [], [], []
    FFRaps_top1_all, FFRaps_top5_all, FFRaps_coverage_all, FFRaps_size_all = [], [], [], []

    Raps_top1_all1, Raps_top5_all1, Raps_coverage_all1, Raps_size_all1 = [], [], [], []
    FFRaps_top1_all1, FFRaps_top5_all1, FFRaps_coverage_all1, FFRaps_size_all1 = [], [], [], []

    for seed in tqdm(args.seed):
        seed_torch(seed)
        np.random.seed(seed=seed)
        random.seed(seed)

        # Transform as in https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L92
        transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std= [0.229, 0.224, 0.225])
                    ])

        # Get the conformal calibration dataset
        imagenet_calib_data, imagenet_val_data = torch.utils.data.random_split(torchvision.datasets.ImageFolder(args.data, transform), [args.num_calib,50000-args.num_calib])

        # Initialize loaders
        calib_loader = torch.utils.data.DataLoader(imagenet_calib_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(imagenet_val_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

        cudnn.benchmark = True

        # Get the model
        model = torchvision.models.inception_v3(pretrained=True,progress=True).cuda()
        model = torch.nn.DataParallel(model)
        model.eval()

        modelFF = copy.deepcopy(model)

        # optimize for 'size' or 'adaptiveness'
        lamda_criterion = 'size'
        # allow sets of size zero
        allow_zero_sets = False
        # use the randomized version of conformal
        randomized = True
        # Fixed softmax temperature parameter
        optional_T = None

        print("start Raps")
        # Conformalize model
        model = ConformalModel_ori(model, calib_loader, alpha=0.1, lamda=None, randomized=randomized, allow_zero_sets=allow_zero_sets)

        print("Model calibrated and conformalized! Now evaluate over remaining data.")
        Raps_top1, Raps_top5, Raps_coverage, Raps_size = validate_ori(val_loader, model, print_bool=True)

        Raps_top1_all.append(Raps_top1)
        Raps_top5_all.append(Raps_top5)
        Raps_coverage_all.append(Raps_coverage)
        Raps_size_all.append(Raps_size)

        print("Complete Raps!")

        print("start FFRaps")
        # Conformalize model
        model = ConformalModel(modelFF, calib_loader, alpha=0.1, lamda=None, randomized=randomized, allow_zero_sets=allow_zero_sets, delta=None, T=optional_T)

        print("Model calibrated and conformalized! Now evaluate over remaining data.")
        FFRaps_top1, FFRaps_top5, FFRaps_coverage, FFRaps_size = validate(val_loader, model, print_bool=True)

        FFRaps_top1_all.append(FFRaps_top1)
        FFRaps_top5_all.append(FFRaps_top5)
        FFRaps_coverage_all.append(FFRaps_coverage)
        FFRaps_size_all.append(FFRaps_size)

        print("Complete!")

print(f'top1: {np.mean(Raps_top1_all)} \\pm {np.std(Raps_top1_all)}',
      f'top5: {np.mean(Raps_top5_all)} \\pm {np.std(Raps_top5_all)}',
      f'Raps_coverage: {np.mean(Raps_coverage_all)} \\pm {np.std(Raps_coverage_all)}',
      f'Raps_size: {np.mean(Raps_size_all)} \\pm {np.std(Raps_size_all)}')

print(f'top1: {np.mean(FFRaps_top1_all)} \\pm {np.std(FFRaps_top1_all)}',
      f'top5: {np.mean(FFRaps_top5_all)} \\pm {np.std(FFRaps_top5_all)}',
      f'FFRaps_coverage: {np.mean(FFRaps_coverage_all)} \\pm {np.std(FFRaps_coverage_all)}',
      f'FFRaps_size: {np.mean(FFRaps_size_all)} \\pm {np.std(FFRaps_size_all)}')

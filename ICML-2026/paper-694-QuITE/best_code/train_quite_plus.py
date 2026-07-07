"""Entrypoint: train QuITE++ for IMTS forecasting.

Example:
    python train_quite_plus.py --dataset physionet --history 24 \
        --patch_size 6 --stride 6 --hid_dim 64 --nhead 4 --nlayer 2 \
        --seed 1 --gpu 0
"""
import os
import sys
import time
import math
import datetime
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from random import SystemRandom

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser('QuITE++ Forecasting')

# ----- Training -----
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--logmode', type=str, default='a')
parser.add_argument('--load', type=str, default=None,
                    help='Experiment ID to resume; if None, a fresh one is generated')

# ----- Dataset -----
parser.add_argument('--dataset', type=str, default='physionet',
                    choices=['physionet', 'mimic', 'ushcn', 'activity'])
parser.add_argument('--history', type=int, default=24,
                    help='Historical window length (hours / months / ms by dataset)')
parser.add_argument('--pred_window', type=int, default=1)
parser.add_argument('-n', type=int, default=int(1e8),
                    help='Cap on dataset size (debug); when < 12000 a debug log file is used')
parser.add_argument('--quantization', type=float, default=0.0,
                    help='Time quantization (PhysioNet only)')

# ----- Model (QuITE++) -----
parser.add_argument('-hd', '--hid_dim', type=int, default=64)
parser.add_argument('--nhead', type=int, default=4)
parser.add_argument('--nlayer', type=int, default=2)
parser.add_argument('-ps', '--patch_size', type=float, default=24)
parser.add_argument('--stride', type=float, default=24)
parser.add_argument('--dropout', type=float, default=0.1)

args = parser.parse_args()
args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import torch.optim as optim
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

import utils
from data.parse import parse_datasets
from evaluation import compute_all_losses, evaluation
from models.quite_plus import Model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_of_patches(n_patch):
    if n_patch == 1:
        return 1
    if n_patch % 2 == 0:
        return 1 + layer_of_patches(n_patch / 2)
    return layer_of_patches(n_patch + 1)


if __name__ == '__main__':
    utils.setup_seed(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.PID = os.getpid()
    print("PID, device:", args.PID, args.device)

    experimentID = args.load or int(SystemRandom().random() * 100000)

    input_command = sys.argv
    ind = [i for i, a in enumerate(input_command) if a == "--load"]
    if len(ind) == 1:
        i = ind[0]
        input_command = input_command[:i] + input_command[(i + 2):]
    input_command = " ".join(input_command)

    data_obj = parse_datasets(args, patch_ts=True, max_ts=True)
    args.ndim = data_obj["input_dim"]
    args.npatch = int(math.ceil((args.history - args.patch_size) / args.stride)) + 1
    args.patch_layer = layer_of_patches(args.npatch)
    args.scale_patch_size = args.patch_size / (args.history + args.pred_window)
    args.task = 'forecasting'

    model = Model(args).to(args.device)

    if args.n < 12000:
        log_path = f"logs/{args.dataset}_quite_plus_debug.log"
    else:
        log_path = (
            f"logs/{args.dataset}_quite_plus_{args.hid_dim}hdims_"
            f"{args.nlayer}nlayers_{args.npatch}npatch_{args.patch_size}patchsize_"
            f"{args.history}history_{args.pred_window}pred_{args.nhead}nhead_"
            f"{args.lr}lr_{args.seed}seed.log"
        )
    utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(input_command)
    logger.info(args)
    logger.info(model)
    logger.info(f'parameters:{count_parameters(model)}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_batches = data_obj["n_train_batches"]
    best_val_mse = np.inf
    test_res = None

    for itr in range(args.epoch):
        model.train()
        st = time.time()
        for _ in tqdm(range(num_batches)):
            optimizer.zero_grad()
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
            if batch_dict is None:
                continue
            train_res = compute_all_losses(model, batch_dict)
            train_res["loss"].backward()
            optimizer.step()
        logger.info(f'training time per epoch :{time.time() - st:.3f}s')

        model.eval()
        with torch.no_grad():
            st = time.time()
            val_res, _ = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])
            logger.info(f"val time per epoch :{time.time() - st:.2f}s")

            if val_res["mse"] < best_val_mse:
                best_val_mse = val_res["mse"]
                best_iter = itr
                st = time.time()
                test_res, _ = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])
                logger.info(f"test time per epoch :{time.time() - st:.2f}s")

            logger.info(f'- Epoch {itr:03d}, ExpID {experimentID}')
            logger.info(f"Train - Loss (one batch): {train_res['loss'].item():.5f}")
            logger.info(
                f"Val - Loss, MSE, RMSE, MAE, MAPE: {val_res['loss']:.5f}, "
                f"{val_res['mse']:.5f}, {val_res['rmse']:.5f}, {val_res['mae']:.5f}, "
                f"{val_res['mape'] * 100:.2f}%"
            )
            if test_res is not None:
                logger.info(
                    f"Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {best_iter}, "
                    f"{test_res['loss']:.5f}, {test_res['mse']:.5f}, "
                    f"{test_res['rmse']:.5f}, {test_res['mae']:.5f}, "
                    f"{test_res['mape'] * 100:.2f}%"
                )

        if itr - best_iter >= args.patience:
            logger.info("Exp has been early stopped!")
            sys.exit(0)

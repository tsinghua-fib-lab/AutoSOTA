import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import datetime
import random
import time
from pathlib import Path
import numpy as np
import torch
import util.misc as utils
from engine import train_one_epoch
from models import build_MixerCSeg as build_model
from datasets import create_dataset
import cv2
from eval.evaluate import eval
from util.logger import get_logger
from tqdm import tqdm
from mmengine.optim.scheduler.lr_scheduler import PolyLR

# from torch.utils.tensorboard import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('MixerCSeg FOR CRACK', add_help=False)

    parser.add_argument('--BCELoss_ratio', default=0.87, type=float,
                        help='Weight ratio for Binary Cross Entropy Loss (0.0-1.0), should sum to 1 with DiceLoss_ratio')
    parser.add_argument('--DiceLoss_ratio', default=0.13, type=float,
                        help='Weight ratio for Dice Loss (0.0-1.0), should sum to 1 with BCELoss_ratio')
    parser.add_argument('--Norm_Type', default='GN', type=str,
                        help='Normalization layer type [GN|BN], GN=GroupNorm')
    parser.add_argument('--dataset_path', default="/home/linux/code/sod/dataset/CrackMap",
                        help='Root directory path for dataset')
    parser.add_argument('--batch_size_train', type=int, default=1,
                        help='Number of samples per training batch (affects memory usage)')
    parser.add_argument('--batch_size_test', type=int, default=1,
                        help='Number of samples per batch')
    parser.add_argument('--lr_scheduler', type=str, default='PolyLR',
                        help='Learning rate scheduler type [PolyLR|StepLR|CosLR]')
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Initial learning rate (base value for schedulers)')
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='Minimum learning rate for PolyLR')
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help='Weight decay coefficient for regularization')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Total number of training epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Manual epoch number to start training (useful for resuming)')
    parser.add_argument('--lr_drop', default=30, type=int,
                        help='Epoch interval for dropping learning rate in StepLR scheduler')
    parser.add_argument('--sgd', action='store_true',
                        help='Use SGD optimizer instead of default AdamW')
    parser.add_argument('--output_dir', default='./checkpoints/weights',
                        help='Directory to save model checkpoints')
    parser.add_argument('--device', default='cuda',
                        help='Computation device [cuda|cpu] for training/inference')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')
    parser.add_argument('--dataset_mode', type=str, default='crack',
                        help='Dataset mode selector')
    parser.add_argument('--serial_batches', action='store_true',
                        help='Disable random shuffling and use sequential batch sampling if enabled')
    parser.add_argument('--num_threads', default=1, type=int,
                        help='Number of subprocesses for data loading')
    parser.add_argument('--phase', type=str, default='train',
                        help='Runtime phase selector')
    parser.add_argument('--load_width', type=int, default=512,
                        help='Input image width for preprocessing (will be resized)')
    parser.add_argument('--load_height', type=int, default=512,
                        help='Input image height for preprocessing (will be resized)')
    parser.add_argument('--use_tta', action='store_true',
                        help='Enable Test-Time Augmentation (multi-scale + flip ensemble) during inference')
    parser.add_argument('--use_morph', action='store_true',
                        help='Enable morphological post-processing during inference')
    return parser

def main(args):
    checkpoints_path = "../newcseg/checkpoints"
    cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))
    dataset_name = (args.dataset_path).split('/')[-1]
    process_folder_path = os.path.join(checkpoints_path, cur_time + '_' + dataset_name)
    args.phase = 'train'
    if not os.path.exists(process_folder_path):
        os.makedirs(process_folder_path)
    else:
        print("create process folder error!")

    log_train = get_logger(process_folder_path, 'train')
    log_test = get_logger(process_folder_path, 'test')
    log_eval = get_logger(process_folder_path, 'eval')

    # writer = SummaryWriter('./tensorboard_runs/{}'.format(os.path.join(cur_time + '_' + dataset_name)))

    log_train.info("args -> " + str(args))
    log_train.info("args: dataset -> " + str(args.dataset_path))
    log_train.info("args: BCELoss_ratio -> " + str(args.BCELoss_ratio))
    log_train.info("args: DiceLoss_ratio -> " + str(args.DiceLoss_ratio))
    print("args: BCELoss_ratio -> " + str(args.BCELoss_ratio))
    print("args: DiceLoss_ratio -> " + str(args.DiceLoss_ratio))

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    model.to(device)
    args.batch_size = args.batch_size_train
    train_dataLoader = create_dataset(args)
    dataset_size = len(train_dataLoader)
    print('The number of training images = %d' % dataset_size)
    log_train.info('The number of training images = %d' % dataset_size)

    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters()],
            "lr": args.lr,
        },
    ]
    if args.sgd:
        print('use SGD!')
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        print('use AdamW!')
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)

    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'CosLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)
    elif args.lr_scheduler == 'PolyLR':
        lr_scheduler = PolyLR(optimizer, eta_min=args.min_lr, begin=args.start_epoch, end=args.epochs)
    else:
        raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler}")
    output_dir = args.output_dir + '/' + cur_time + '_Dataset->' + dataset_name
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)

    print("Start processing! ")
    log_train.info("Start processing! ")
    start_time = time.time()
    max_mIoU = 0
    max_Metrics = {'epoch': 0, 'mIoU': 0, 'ODS': 0, 'OIS': 0, 'F1': 0, 'Precision': 0, 'Recall': 0}

    for epoch in range(args.start_epoch, args.epochs):
        print("---------------------------------------------------------------------------------------")
        print("training epoch start -> ", epoch)
        train_one_epoch(model, criterion, train_dataLoader, optimizer, epoch, args, log_train)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        print("training epoch finish -> ", epoch)
        print("---------------------------------------------------------------------------------------")

        print("testing epoch start -> ", epoch)
        results_path = cur_time + '_Dataset->' + dataset_name
        save_root = f'./results/{results_path}/results_' + str(epoch)
        args.phase = 'test'
        args.batch_size = args.batch_size_test
        test_dl = create_dataset(args)
        pbar = tqdm(total=len(test_dl), desc=f"Initial Loss: Pending")

        if not os.path.isdir(save_root):
            os.makedirs(save_root)
        with torch.no_grad():
            model.eval()
            for batch_idx, (data) in enumerate(test_dl):
                x = data["image"]
                target = data["label"]
                if device != 'cpu':
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                out = model(x)
                loss = criterion(out, target.float())
                target = target[0, 0, ...].cpu().numpy()
                out = out[0, 0, ...].cpu().numpy()
                root_name = data["A_paths"][0].split("/")[-1][0:-4]

                target = 255 * (target / np.max(target))
                out = 255 * (out / np.max(out))

                # out[out >= 0.5] = 255
                # out[out < 0.5] = 0

                log_test.info('----------------------------------------------------------------------------------------------')
                log_test.info("loss -> " + str(loss))
                log_test.info(str(os.path.join(save_root, "{}_lab.png".format(root_name))))
                log_test.info(str(os.path.join(save_root, "{}_pre.png".format(root_name))))
                log_test.info('----------------------------------------------------------------------------------------------')
                cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target)
                cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out)
                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)
        pbar.close()

        log_test.info("model -> " + str(epoch) + " test finish!")
        log_test.info('----------------------------------------------------------------------------------------------')
        print("testing epoch finish -> ", epoch)
        print("---------------------------------------------------------------------------------------")

        print("evalauting epoch start -> ", epoch)
        metrics = eval(log_eval, save_root, epoch)
        for key, value in metrics.items():
            print(str(key) + ' -> ' + str(value))
        if(max_mIoU < metrics['mIoU']):
            max_Metrics = metrics
            max_mIoU = metrics['mIoU']
            checkpoint_paths = [output_dir / f'checkpoint_best.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            log_train.info("\nupdate and save best model -> " + str(epoch))
            print("\nupdate and save best model -> ", epoch)

        print("evalauting epoch finish -> ", epoch)
        print('\nmax_mIoU -> ' + str(max_Metrics['mIoU']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
        print("---------------------------------------------------------------------------------------")

        log_eval.info("evalauting epoch finish -> " + str(epoch))
        log_eval.info('\nmax_mIoU -> ' + str(max_Metrics['mIoU']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
        log_eval.info("---------------------------------------------------------------------------------------")

    # writer.close()

    for key, value in max_Metrics.items():
        log_eval.info(str(key) + ' -> ' + str(value))
    log_eval.info('\nmax_mIoU -> ' + str(max_Metrics['mIoU']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Process time {}'.format(total_time_str))
    log_train.info('Process time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MixerCSeg FOR CRACK', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

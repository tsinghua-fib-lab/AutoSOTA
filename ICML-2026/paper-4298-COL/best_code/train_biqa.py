"""Training script for ConOrd on BIQA datasets (BID).

Implements:
- ViT-B/CLIP backbone
- AdamW optimizer with weight_decay=2e-3
- Cosine annealing scheduler with 5-epoch warmup
- Three-crops averaged features during inference (Appendix B.2)
- k-NN inference with SRCC/PCC evaluation
- 10 random 80:20 splits with median evaluation
"""

import os
import sys
import time
import argparse
from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random
from scipy import stats

from config.basic import ConfigBasic
from utils.util import (write_log, get_current_time, to_np, make_dir,
                        log_configs, save_ckpt, AverageMeter,
                        extract_embs, cal_srocc_plcc)
from utils.loss_util import ConOrdLoss, compute_center_loss
from utils.comparison_utils import find_kNN
from networks.util import prepare_model
from data.get_datasets_BIQA import get_datasets_BIQA


def parse_args():
    parser = argparse.ArgumentParser(description='ConOrd BIQA Training')
    parser.add_argument('--dataset', type=str, default='bid')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=2e-3)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--epsilon', type=float, default=1e-7)
    parser.add_argument('--k_nn', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--data_root', type=str, default='/datasets/BID')
    return parser.parse_args()


def set_local_config(cfg, args):
    cfg.dataset = args.dataset
    cfg.logscale = False
    cfg.set_biqa_dataset()
    cfg.data_root = args.data_root
    cfg.tau = 0

    # Model
    cfg.model = 'ConOrd'
    cfg.backbone = 'vitB16'

    # Evaluation
    cfg.k = args.k_nn  # Single k value as in paper
    cfg.scheduler = 'cosine_warmup'
    cfg.warmup_epochs = args.warmup_epochs
    cfg.lr_decay_epochs = [100, 200, 300]

    # Reference points
    cfg.margin = 0.05
    cfg.ref_mode = 'flex'
    cfg.ref_point_num = 60
    cfg.drct_wieght = 1
    cfg.start_norm = True
    cfg.epochs = args.epochs
    cfg.learning_rate = args.lr
    cfg.weight_decay = args.weight_decay

    # Loss
    cfg.metric = 'L2'
    cfg.label_diff = 'l2'
    cfg.similarity_type = 'L2'
    cfg.epsilon = args.epsilon
    cfg.temp = args.tau

    # Training
    cfg.lr_decay_rate = 0.0005
    cfg.batch_size = args.batch_size
    cfg.test_batch_size = 1000

    # Splits
    cfg.n_splits = args.n_splits
    cfg.split_seed = args.split_seed

    # Logging
    cfg.wandb = False
    cfg.experiment_name = (f'ConOrd_BIQA_{cfg.dataset}_lr{cfg.learning_rate}_'
                          f'bs{cfg.batch_size}_tau{cfg.temp}_eps{cfg.epsilon}_'
                          f'k{cfg.k}_warmup{cfg.warmup_epochs}')
    cfg.save_folder = (f'../results_biqa/{cfg.dataset}/{cfg.experiment_name}_'
                       f'{get_current_time()}')
    make_dir(cfg.save_folder)

    cfg.n_gpu = torch.cuda.device_count()
    cfg.num_workers = 0
    cfg.gpu_ids = [args.gpu]
    cfg.device = torch.device(f'cuda:{cfg.gpu_ids[0]}')
    return cfg


def generate_three_crops(images, cfg):
    """Generate three crops (top-left, bottom-right, center) for each image.
    Returns [3*B, C, H, W] tensor. Uses pure GPU tensor ops (no PIL)."""
    from torchvision import transforms
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    b, c, h, w = images.shape
    if h != 256:
        images = nn.functional.interpolate(images, size=256, mode="bilinear", align_corners=False)
    tl = images[:, :, :224, :224]
    br = images[:, :, -224:, -224:]
    cc = images[:, :, 16:240, 16:240]
    all_crops = torch.cat([tl, br, cc], dim=0)
    all_crops = normalize(all_crops)
    return all_crops


def train_one_epoch(epoch, train_loader, model, optimizer, criterion, cfg):
    """One epoch training with three-crops augmentation."""
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dist_losses = AverageMeter()
    center_losses = AverageMeter()
    end = time.time()

    for idx, (images, _, ranks, _) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.to(cfg.device)
            ranks = ranks.to(cfg.device)

        data_time.update(time.time() - end)

        # Generate three crops per image -> triples the batch
        images_3crop = generate_three_crops(images, cfg)  # [3*B, C, H, W]
        ranks_3crop = ranks.repeat(3)  # [3*B]

        bsz = ranks_3crop.shape[0] // 2

        # Encode all crops through the model
        features = model.encoder(images_3crop)
        features = nn.functional.normalize(features, dim=-1)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        dist_loss = criterion(features, ranks_3crop, cfg)
        center_loss = compute_center_loss(
            torch.cat(torch.unbind(features, dim=1), dim=0),
            ranks_3crop, model.ref_points, cfg)
        total_loss = dist_loss + center_loss

        losses.update(total_loss.item(), ranks.size(0))
        dist_losses.update(dist_loss.item(), ranks.size(0))
        center_losses.update(center_loss.item(), ranks.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % cfg.print_freq == 0:
            write_log(cfg.logfile,
                      f'Epoch [{epoch}][{idx}/{len(train_loader)}]\t'
                      f'Loss {losses.val:.4f}\t'
                      f'Dist {dist_losses.val:.4f}\t'
                      f'Center {center_losses.val:.4f}')
            sys.stdout.flush()

    return losses.avg


def evaluate_biqa(loader_dict, model, cfg):
    """Evaluate on BIQA dataset using SRCC and PCC.

    Uses three-crops averaged features during inference (Appendix B.2).
    """
    model.eval()

    # Extract features with three-crops averaging for test set
    embs_train = extract_embs_three_crop(model.encoder, loader_dict['train_for_val'], cfg)
    embs_train = embs_train.to(cfg.device)

    embs_test = extract_embs_three_crop(model.encoder, loader_dict['val'], cfg)
    embs_test = embs_test.to(cfg.device)

    test_labels = loader_dict['val'].dataset.mos
    train_labels = loader_dict['train_for_val'].dataset.mos

    if isinstance(test_labels, torch.Tensor):
        test_labels = to_np(test_labels)
    if isinstance(train_labels, torch.Tensor):
        train_labels = to_np(train_labels)

    n_test = len(embs_test)
    n_batch = int(np.ceil(n_test / cfg.test_batch_size))

    k = cfg.k

    with torch.no_grad():
        vals, inds = find_kNN(embs_test, embs_train, k=k, metric=cfg.metric)
        inds = np.squeeze(to_np(inds), 0)
        if inds.ndim == 1:
            inds = inds[np.newaxis, :]

        nn_labels = train_labels[inds[:, :k]]
        pred_scores = np.mean(nn_labels, axis=-1)

    srcc, plcc = cal_srocc_plcc(pred_scores, test_labels)
    write_log(cfg.logfile, f'SRCC: {srcc:.4f}, PLCC: {plcc:.4f}')
    return srcc, plcc


def extract_embs_three_crop(encoder, data_loader, cfg):
    """Extract features using three-crops averaging - optimized GPU version.

    For each image, extracts features from top-left, bottom-right, and center crops,
    then averages them as described in Appendix B.2.
    Uses pure tensor operations (no PIL roundtrips).
    """
    from torchvision import transforms
    encoder.eval()

    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )

    embs = []
    inds = []

    with torch.no_grad():
        for batch in data_loader:
            x_base = batch[0]
            item = batch[-1]
            bsz = x_base.size(0)

            x_base = x_base.to(cfg.device)
            if x_base.shape[-1] != 256:
                x_base = nn.functional.interpolate(
                    x_base, size=256, mode="bilinear", align_corners=False
                )

            tl = x_base[:, :, :224, :224]
            br = x_base[:, :, -224:, -224:]
            cc = x_base[:, :, 16:240, 16:240]

            all_crops = torch.cat([tl, br, cc], dim=0)
            all_crops = normalize(all_crops)

            feats = encoder(all_crops)
            feats = feats.view(bsz, 3, -1).mean(dim=1)
            embs.append(feats.cpu())
            inds.append(item)

    embs = torch.cat(embs)
    inds = torch.cat(inds)
    embs_temp = deepcopy(embs)
    embs[inds] = embs_temp
    return embs


class WarmupCosineScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 eta_min_factor=0.0005):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.eta_min_factor = eta_min_factor
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup from 0.2x to 1x base lr
            warmup_factor = (0.2 + 0.8 * self.current_epoch / self.warmup_epochs)
            for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = blr * warmup_factor
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            eta_min = self.eta_min_factor
            for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = blr * (eta_min + (1 - eta_min) * cosine_factor)

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


def run_single_split(split_idx, args, cfg_template):
    """Run training and evaluation for a single data split."""
    cfg = deepcopy(cfg_template)

    # Set split-specific seed
    split_seed = args.split_seed + split_idx
    random_seed = split_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)

    cfg.split_seed = split_seed
    cfg.save_folder = (f'{cfg_template.save_folder}/split_{split_idx:02d}')
    make_dir(cfg.save_folder)
    cfg.logfile = log_configs(cfg, log_file='train_log.txt')

    # Data
    loader_dict = get_datasets_BIQA(cfg)
    cfg.n_ranks = int(cfg.rank_max - cfg.rank_min) + 1
    cfg.ref_point_num = 60  # keep 60 ref points for BIQA continuous MOS
    cfg.fiducial_point_num = 60  # keep 60 fiducial points for BIQA continuous MOS
    write_log(cfg.logfile, f'[*] MOS range: [{cfg.rank_min:.2f}, {cfg.rank_max:.2f}], '
              f'n_ranks: {cfg.n_ranks}')

    # Model
    model = prepare_model(cfg)
    model = model.to(cfg.device)

    # Optimizer: AdamW as in paper
    param_groups = []
    for key, value in dict(model.named_children()).items():
        param_groups += [{"params": value.parameters(), "lr": cfg.learning_rate}]
    param_groups += [{"params": model.ref_points, "lr": cfg.learning_rate * 10}]

    optimizer = optim.AdamW(params=param_groups,
                            lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)

    # Scheduler: cosine with warmup
    scheduler = WarmupCosineScheduler(optimizer, cfg.warmup_epochs, cfg.epochs,
                                      eta_min_factor=cfg.lr_decay_rate)

    # Loss
    criterion = ConOrdLoss(label_diff=cfg.label_diff,
                          feature_sim=cfg.similarity_type,
                          temperature=cfg.temp)

    # Training loop
    best_srcc = 0.0
    best_plcc = 0.0

    for epoch in range(cfg.epochs):
        write_log(cfg.logfile, f'==> Split {split_idx}, Epoch {epoch} training...')
        time1 = time.time()
        train_loss = train_one_epoch(epoch, loader_dict['train'], model,
                                     optimizer, criterion, cfg)
        scheduler.step()
        time2 = time.time()
        write_log(cfg.logfile, f'Epoch {epoch}, loss {train_loss:.4f}, '
                  f'time {time2 - time1:.2f}s, '
                  f'lr {scheduler.get_lr()[0]:.2e}')

        write_log(cfg.logfile, '==> validation...')
        srcc, plcc = evaluate_biqa(loader_dict, model, cfg)

        if srcc > best_srcc:
            best_srcc = srcc
            best_plcc = plcc
            save_ckpt(cfg, model, f'best_SRCC_{srcc:.4f}_PLCC_{plcc:.4f}.pth')

    write_log(cfg.logfile, f'[Split {split_idx}] Best SRCC: {best_srcc:.4f}, '
              f'Best PLCC: {best_plcc:.4f}')
    return best_srcc, best_plcc


def main():
    args = parse_args()

    # Create base config
    cfg = ConfigBasic()
    cfg = set_local_config(cfg, args)
    base_save_folder = cfg.save_folder

    write_log(open(os.path.join(base_save_folder, 'summary.txt'), 'w'),
              f'ConOrd BIQA Training on {args.dataset}')
    write_log(open(os.path.join(base_save_folder, 'summary.txt'), 'a'),
              f'Hyperparams: lr={args.lr}, bs={args.batch_size}, '
              f'tau={args.tau}, eps={args.epsilon}, k={args.k_nn}, '
              f'epochs={args.epochs}, warmup={args.warmup_epochs}, '
              f'n_splits={args.n_splits}')

    cfg_template = deepcopy(cfg)

    # Run over multiple splits
    srcc_results = []
    plcc_results = []

    for split_idx in range(args.n_splits):
        srcc, plcc = run_single_split(split_idx, args, cfg_template)
        srcc_results.append(srcc)
        plcc_results.append(plcc)
        write_log(open(os.path.join(base_save_folder, 'summary.txt'), 'a'),
                  f'Split {split_idx}: SRCC={srcc:.4f}, PLCC={plcc:.4f}')

    # Report median (as in paper)
    median_srcc = np.median(srcc_results)
    median_plcc = np.median(plcc_results)
    mean_srcc = np.mean(srcc_results)
    std_srcc = np.std(srcc_results)
    mean_plcc = np.mean(plcc_results)
    std_plcc = np.std(plcc_results)

    summary = (f'\n=== FINAL RESULTS ===\n'
               f'SRCC (median): {median_srcc:.4f}\n'
               f'SRCC (mean±std): {mean_srcc:.4f}±{std_srcc:.4f}\n'
               f'PLCC (median): {median_plcc:.4f}\n'
               f'PLCC (mean±std): {mean_plcc:.4f}±{std_plcc:.4f}\n'
               f'All SRCC: {[f"{v:.4f}" for v in srcc_results]}\n'
               f'All PLCC: {[f"{v:.4f}" for v in plcc_results]}\n')

    write_log(open(os.path.join(base_save_folder, 'summary.txt'), 'a'), summary)
    print(summary)

    return median_srcc, median_plcc


if __name__ == '__main__':
    main()

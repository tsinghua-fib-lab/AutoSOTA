#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anti-Curriculum Fine-tuning Stage (Standalone Script)
=====================================================

✅ Purpose:
- Load your trained model (e.g., bestMAE ~ 0.031)
- Compute difficulty = 1 - softIoU on FULL training set
- Select HARDEST top hard_ratio samples
- Fine-tune model only on these hard samples (Anti-curriculum)
- (Optional) SBFT: low-pass blur to suppress high-frequency shortcut
- Validate every epoch & save best checkpoint

This script is standalone: it DOES NOT include curriculum selection stage.
It assumes your repo already has:
  - lib/Network.py  -> Network
  - utils/data_val.py -> get_loader, test_dataset
  - utils/utils.py -> clip_gradient, get_coef, cal_ual
"""

import os
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

from lib.Network import Network
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, get_coef, cal_ual


# -----------------------------
# Globals
# -----------------------------
device_ids = [0]
best_mae = 1.0
best_epoch = 0
step = 0


# -----------------------------
# Losses (SAME AS YOUR CURRENT)
# -----------------------------
def dice_loss(predict, target, smooth=1.0, p=2.0):
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def structure_loss(pred_logits, mask):
    """Original structure loss (NO extra weighting)"""
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )

    bce = F.binary_cross_entropy_with_logits(pred_logits, mask, reduction='none')
    wbce = (weit * bce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-6)

    pred = torch.sigmoid(pred_logits)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1.0) / (union - inter + 1.0)

    return wbce + wiou


# -----------------------------
# Difficulty = 1 - softIoU
# -----------------------------
def batch_soft_iou_from_logits(logits, gts, eps=1e-6):
    p = torch.sigmoid(logits)
    g = gts.float()
    inter = (p * g).sum(dim=(1, 2, 3))
    union = (p + g - p * g).sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    return iou


@torch.no_grad()
def compute_difficulty_by_model(model, full_loader, device, final_idx=4):
    """
    difficulty = 1 - softIoU
    return: dict idx -> difficulty
    """
    model.eval()
    d_map = {}

    for images, gts, edges, idxs in full_loader:
        images = images.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)

        preds = model(images)
        iou = batch_soft_iou_from_logits(preds[final_idx], gts)
        d = 1.0 - iou

        if torch.is_tensor(idxs):
            idxs = idxs.detach().cpu().tolist()

        for j, idx in enumerate(idxs):
            d_map[int(idx)] = float(d[j].item())

    return d_map


def summarize_difficulty(d_map, tag="difficulty"):
    ds = np.array(list(d_map.values()), dtype=np.float32)
    if ds.size == 0:
        print(f"[DIFF STAT] {tag}: empty")
        return

    print(
        f"[DIFF STAT] {tag}: N={len(ds)} "
        f"min={ds.min():.6f} p10={np.quantile(ds, 0.1):.6f} p50={np.quantile(ds, 0.5):.6f} "
        f"p90={np.quantile(ds, 0.9):.6f} max={ds.max():.6f} mean={ds.mean():.6f} std={ds.std():.6f}"
    )


# -----------------------------
# Robust idx -> dataset_index mapping
# -----------------------------
def build_idx_to_dataset_index(dataset, max_print=5):
    """
    dataset[i] returns (img, gt, edge, idx)
    We build mapping:
      returned idx  -> actual dataset index i
    This makes Subset safe even if idx != i
    """
    idx2i = {}
    dup = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        idx = int(sample[-1])
        if idx in idx2i:
            dup += 1
            if dup <= max_print:
                print(f"[IDX MAP] duplicate idx found: idx={idx} already mapped to {idx2i[idx]}, new={i}")
        else:
            idx2i[idx] = i

    print(f"[IDX MAP] built: {len(idx2i)}/{len(dataset)} (dup={dup})")
    return idx2i


def select_hardest_subset(d_map, hard_ratio=0.2):
    """
    Pick hardest top hard_ratio by difficulty descending
    returns: list of idx (the returned idx from dataset)
    """
    pairs = [(k, float(v)) for k, v in d_map.items()]
    pairs.sort(key=lambda x: x[1], reverse=True)

    k = max(1, int(np.ceil(len(pairs) * hard_ratio)))
    hard_idxs = [p[0] for p in pairs[:k]]
    return hard_idxs


# -----------------------------
# SBFT (Low-pass blur in spatial domain)
# -----------------------------
def gaussian_kernel1d(kernel_size: int, sigma: float, device):
    """Create 1D Gaussian kernel"""
    x = torch.arange(kernel_size, device=device).float() - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / (kernel.sum() + 1e-8)
    return kernel


def lowpass_blur(images, kernel_size=11, sigma=3.0):
    """
    Simple Gaussian blur (separable conv) as low-pass filter.
    images: [B,C,H,W]
    """
    device = images.device
    B, C, H, W = images.shape
    k1d = gaussian_kernel1d(kernel_size, sigma, device=device)

    # [1,1,k,1] and [1,1,1,k]
    kx = k1d.view(1, 1, kernel_size, 1)
    ky = k1d.view(1, 1, 1, kernel_size)

    # apply per-channel (groups=C)
    # first vertical then horizontal
    images = F.pad(images, (0, 0, kernel_size // 2, kernel_size // 2), mode="reflect")
    images = F.conv2d(images, kx.expand(C, 1, kernel_size, 1), groups=C)

    images = F.pad(images, (kernel_size // 2, kernel_size // 2, 0, 0), mode="reflect")
    images = F.conv2d(images, ky.expand(C, 1, 1, kernel_size), groups=C)

    return images


# -----------------------------
# Validation (SAME STYLE)
# -----------------------------
def val(test_loader, model, epoch, save_path, writer):
    global best_mae, best_epoch

    model.eval()
    with torch.no_grad():
        mae_sum = 0.0

        for _ in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)

            image = image.cuda(device=device_ids[0], non_blocking=True)

            result = model(image)
            res = F.interpolate(result[4], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            mae_sum += np.sum(np.abs(res - gt)) / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / float(test_loader.size)
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)

        print(f'[Val] Epoch: {epoch}, MAE: {mae:.6f}, bestMAE: {best_mae:.6f}, bestEpoch: {best_epoch}')
        logging.info(f'[Val Info]:Epoch:{epoch} MAE:{mae} bestEpoch:{best_epoch} bestMAE:{best_mae}')

        if epoch == 1:
            best_mae = mae
            best_epoch = 1
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_path, 'Net_epoch_best.pth'))
                print(f'[Val] Save best state_dict! Best epoch: {epoch}')
                logging.info(f'[Val] Save best state_dict! Best epoch: {epoch}')


# -----------------------------
# Anti-curriculum training loop
# -----------------------------
def train_one_epoch_anti_curri(
    hard_loader,
    model,
    optimizer,
    epoch,
    writer,
    use_sbft=False,
    sbft_prob=0.7,
    sbft_kernel=11,
    sbft_sigma=3.0,
):
    global step

    model.train()
    device = next(model.parameters()).device
    total_step = max(len(hard_loader), 1)

    loss_all = 0.0
    epoch_step = 0

    for it, (images, gts, edges, idxs) in enumerate(hard_loader, start=1):
        optimizer.zero_grad(set_to_none=True)

        images = images.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)
        edges = edges.to(device, non_blocking=True)

        # ✅ SBFT: probabilistic low-pass blur
        if use_sbft and (np.random.rand() < sbft_prob):
            images = lowpass_blur(images, kernel_size=sbft_kernel, sigma=sbft_sigma)

        preds = model(images)

        # ---- UAL ----
        ual_coef = get_coef(iter_percentage=it / float(total_step), method='cos')
        ual_loss = cal_ual(seg_logits=preds[4], seg_gts=gts)
        ual_loss = ual_loss * ual_coef

        # ---- loss (same as your current) ----
        loss_init = (
            structure_loss(preds[0], gts).mean() * 0.0625 +
            structure_loss(preds[1], gts).mean() * 0.125 +
            structure_loss(preds[2], gts).mean() * 0.25 +
            structure_loss(preds[3], gts).mean() * 0.5
        )
        loss_final = structure_loss(preds[4], gts).mean()
        loss_edge = (
            dice_loss(preds[6], edges) * 0.125 +
            dice_loss(preds[7], edges) * 0.25 +
            dice_loss(preds[8], edges) * 0.5
        )

        loss = loss_init + loss_final + loss_edge + 2.0 * ual_loss

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        epoch_step += 1
        loss_all += float(loss.item())
        step += 1

        if it % 20 == 0 or it == 1 or it == len(hard_loader):
            print(f'{datetime.now()} [Anti-curri] Epoch [{epoch}/{opt.epoch}] Step [{it}/{len(hard_loader)}] '
                  f'Loss: {loss.item():.4f}')
            writer.add_scalars('Loss_Statistics', {
                'Loss_total': loss.item(),
                'Loss_init': float(loss_init.item()),
                'Loss_final': float(loss_final.item()),
                'Loss_edge': float(loss_edge.item()),
            }, global_step=step)

    loss_all /= max(epoch_step, 1)
    writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
    logging.info(f'[Anti-curri Train] Epoch [{epoch}/{opt.epoch}] Loss_AVG: {loss_all:.6f}')
    return loss_all


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # training basic
    parser.add_argument('--epoch', type=int, default=100, help='anti-curri total epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='finetune learning rate')
    parser.add_argument('--batchsize', type=int, default=36, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training image size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

    # paths
    parser.add_argument('--train_root', type=str, default='',
                        help='training dataset root (contains Imgs/ GT/ Edge/)')
    parser.add_argument('--val_root', type=str, default='',
                        help='validation dataset root (contains Imgs/ GT/)')
    parser.add_argument('--save_path', type=str, default='',
                        help='path to save model and log')
    parser.add_argument('--load', type=str, default='',
                        help='path to load ckpt (your bestMAE model, e.g., Net_epoch_best.pth)')

    # anti-curri params
    parser.add_argument('--hard_ratio', type=float, default=0.2,
                        help='hardest ratio used for anti-curri')
    parser.add_argument('--recompute_diff_every', type=int, default=0,
                        help='recompute difficulty every N epochs')
    parser.add_argument('--num_workers', type=int, default=16, help='dataloader workers')

    # SBFT
    parser.add_argument('--use_sbft', action='store_true', help='enable SBFT low-pass blur')
    parser.add_argument('--sbft_prob', type=float, default=0.7, help='probability of SBFT per batch')
    parser.add_argument('--sbft_kernel', type=int, default=11, help='gaussian blur kernel size (odd)')
    parser.add_argument('--sbft_sigma', type=float, default=3.0, help='gaussian blur sigma')

    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    cudnn.benchmark = True

    save_path = opt.save_path
    os.makedirs(save_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(save_path, 'anti_curri.log'),
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO,
        filemode='a',
        datefmt='%Y-%m-%d %I:%M:%S %p'
    )

    logging.info('Anti-Curriculum Stage Start')
    logging.info(
        f'Config: epoch={opt.epoch} lr={opt.lr} batchsize={opt.batchsize} trainsize={opt.trainsize} '
        f'hard_ratio={opt.hard_ratio} use_sbft={opt.use_sbft} load={opt.load}'
    )

    # Build model
    model = Network(channels=192).cuda(device=device_ids[0])

    # Load checkpoint
    ckpt = torch.load(opt.load, map_location='cuda')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    new_state = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(new_state, strict=False)
    print(f"[Load] Loaded ckpt: {opt.load}")
    logging.info(f"[Load] Loaded ckpt: {opt.load}")

    # Optimizer (fine-tune)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=1e-4
    )

    # Data
    print('[Data] Loading...')
    train_loader = get_loader(
        image_root=os.path.join(opt.train_root, 'Imgs/'),
        gt_root=os.path.join(opt.train_root, 'GT/'),
        edge_root=os.path.join(opt.train_root, 'Edge/'),
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        num_workers=opt.num_workers
    )
    val_loader = test_dataset(
        image_root=os.path.join(opt.val_root, 'Imgs/'),
        gt_root=os.path.join(opt.val_root, 'GT/'),
        testsize=opt.trainsize
    )

    writer = SummaryWriter(os.path.join(save_path, 'summary'))

    # Full loader for difficulty compute
    full_loader = DataLoader(
        train_loader.dataset,
        batch_size=opt.batchsize,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Build idx->dataset index mapping for safe Subset
    idx2i = build_idx_to_dataset_index(train_loader.dataset)

    # Compute difficulty once at start
    print("[Anti-curri] Computing difficulty (start)...")
    d_map = compute_difficulty_by_model(model, full_loader, device='cuda', final_idx=4)
    summarize_difficulty(d_map, tag="start")

    hard_idxs = select_hardest_subset(d_map, hard_ratio=opt.hard_ratio)

    # Convert returned idx -> dataset index for Subset
    hard_dataset_indices = []
    miss = 0
    for idx in hard_idxs:
        if idx in idx2i:
            hard_dataset_indices.append(idx2i[idx])
        else:
            miss += 1
    print(f"[Anti-curri] Hard subset: idxN={len(hard_idxs)} -> datasetN={len(hard_dataset_indices)} (miss={miss})")
    logging.info(f"[Anti-curri] Hard subset datasetN={len(hard_dataset_indices)} miss={miss}")

    hard_loader = DataLoader(
        Subset(train_loader.dataset, hard_dataset_indices),
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Train anti-curri
    print('[Anti-curri] Start fine-tuning...')
    for epoch in range(1, opt.epoch + 1):
        cur_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning_rate/lr', cur_lr, global_step=epoch)

        # Optionally recompute difficulty and refresh hard subset
        if opt.recompute_diff_every > 0 and (epoch > 1) and (epoch % opt.recompute_diff_every == 0):
            print(f"[Anti-curri] Recompute difficulty @epoch={epoch} ...")
            d_map = compute_difficulty_by_model(model, full_loader, device='cuda', final_idx=4)
            hard_idxs = select_hardest_subset(d_map, hard_ratio=opt.hard_ratio)

            hard_dataset_indices = []
            for idx in hard_idxs:
                if idx in idx2i:
                    hard_dataset_indices.append(idx2i[idx])

            hard_loader = DataLoader(
                Subset(train_loader.dataset, hard_dataset_indices),
                batch_size=opt.batchsize,
                shuffle=True,
                num_workers=opt.num_workers,
                pin_memory=True,
                drop_last=True
            )

            summarize_difficulty(d_map, tag=f"epoch={epoch}")
            print(f"[Anti-curri] Refreshed hard subset size={len(hard_dataset_indices)}")

        # Train one epoch
        train_one_epoch_anti_curri(
            hard_loader=hard_loader,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            writer=writer,
            use_sbft=opt.use_sbft,
            sbft_prob=opt.sbft_prob,
            sbft_kernel=opt.sbft_kernel,
            sbft_sigma=opt.sbft_sigma
        )

        # Validate every epoch
        val(val_loader, model, epoch, save_path, writer)

        # Save periodic checkpoint
        if epoch % 1 == 0:
            ckpt_path = os.path.join(save_path, f'Net_epoch_{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"[CKPT] Saved: {ckpt_path}")
            logging.info(f"[CKPT] Saved: {ckpt_path}")

    writer.close()
    print('[Done] Anti-curriculum fine-tuning finished.')

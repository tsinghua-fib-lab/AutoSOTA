import torch.nn as nn
import os
import logging
from datetime import datetime
from collections import deque

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


def load_pretrained(model, ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        elif "net" in ckpt:
            state = ckpt["net"]
        else:
            state = ckpt
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)

    print(f"[Load] Loaded checkpoint: {ckpt_path}")
    print(f"[Load] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("[Load] Missing examples:", missing[:5])
    if len(unexpected) > 0:
        print("[Load] Unexpected examples:", unexpected[:5])


# -----------------------------
# Debug utilities
# -----------------------------
def _to_int(x):
    if torch.is_tensor(x):
        return int(x.detach().cpu().item())
    return int(x)


def runtime_check_batch_idxs(idxs, dataset_size, prefix="[IDX RUNTIME]"):
    if torch.is_tensor(idxs):
        arr = idxs.detach().cpu().numpy().astype(np.int64)
    else:
        arr = np.array(idxs, dtype=np.int64)

    mn, mx = int(arr.min()), int(arr.max())
    uq = int(len(np.unique(arr)))
    B = int(len(arr))
    oor = int(np.sum((arr < 0) | (arr >= dataset_size)))

    print(f"{prefix} min={mn} max={mx} unique={uq}/{B} out_of_range={oor}")
    if oor > 0 or uq != B:
        print(f"{prefix} ❌ abnormal idx batch detected (oor={oor}, unique={uq}, B={B})")


def summarize_difficulty(d_map, tag="difficulty", print_hist=False):
    ds = np.array(list(d_map.values()), dtype=np.float32)
    if ds.size == 0:
        print(f"[DIFF STAT] {tag}: empty")
        return {
            "N": 0, "min": None, "p10": None, "p50": None, "p90": None,
            "max": None, "mean": None, "std": None
        }

    stat = {
        "N": int(ds.size),
        "min": float(ds.min()),
        "p10": float(np.quantile(ds, 0.1)),
        "p50": float(np.quantile(ds, 0.5)),
        "p90": float(np.quantile(ds, 0.9)),
        "max": float(ds.max()),
        "mean": float(ds.mean()),
        "std": float(ds.std()),
    }
    print(
        f"[DIFF STAT] {tag}: N={stat['N']} "
        f"min={stat['min']:.6f} p10={stat['p10']:.6f} p50={stat['p50']:.6f} "
        f"p90={stat['p90']:.6f} max={stat['max']:.6f} mean={stat['mean']:.6f} std={stat['std']:.6f}"
    )

    if stat["std"] < 1e-3:
        print("[DIFF STAT] ⚠️ std extremely small -> difficulty collapsed; curriculum may be meaningless.")

    if print_hist:
        hist, bins = np.histogram(ds, bins=10, range=(0, 1))
        print("[DIFF HIST] bins:", bins.tolist())
        print("[DIFF HIST] hist:", hist.tolist())

    return stat


def easy_subset_from_dmap(d_map, p=0.5):
    idxs = np.array(list(d_map.keys()), dtype=np.int64)
    ds = np.array(list(d_map.values()), dtype=np.float32)
    if ds.size == 0:
        return set()
    thr = float(np.quantile(ds, p))
    chosen = idxs[ds <= thr]
    return set(chosen.tolist())


def jaccard(a: set, b: set):
    if len(a) == 0 and len(b) == 0:
        return 1.0
    return float(len(a & b) / (len(a | b) + 1e-6))


# -----------------------------
# Losses
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
    """
    Original structure loss (NO weighting) - kept for reference.
    """
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
# PUE: Pixel-level Uncertainty Estimation
# -----------------------------
def pixel_entropy_from_logits(logits, eps=1e-6):
    """
    Stable entropy in [0,1]
    H = -p log2 p - (1-p) log2 (1-p)
    """
    p = torch.sigmoid(logits)
    p = p.clamp(min=eps, max=1 - eps)

    # use natural log for stability, then convert to log2
    ln2 = np.log(2.0)
    H = -(p * torch.log(p) + (1.0 - p) * torch.log1p(-p)) / ln2

    # safety clamp
    H = torch.clamp(H, 0.0, 1.0)
    H = torch.nan_to_num(H, nan=1.0, posinf=1.0, neginf=0.0)
    return H



def pue_pixel_weight(logits, epoch, burnin_epochs, Tc, W_min=0.1):
    """
    W_{h,w}(t) = W_min + (1-W_min) * (1 - beta(t)*H_{h,w})
    beta(t) = max(0, 1 - (t/Tc)), t counts from curriculum phase start
    """

    # ✅ burn-in 阶段：关闭 PUE（直接返回全1权重）
    if epoch <= burnin_epochs:
        return torch.ones_like(logits)

    # curriculum 阶段才启用 PUE
    t = epoch - burnin_epochs
    beta = max(0.0, 1.0 - float(t) / float(max(Tc, 1)))

    H = pixel_entropy_from_logits(logits)  # [B,1,H,W]
    W = W_min + (1.0 - W_min) * (1.0 - beta * H)
    return W


def structure_loss_pue(pred_logits, mask, epoch, burnin_epochs, Tc, pue_wmin=0.1):
    """
    structure loss + PUE pixel weighting
    return: per-sample loss vector [B]
    """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )

    Wpix = pue_pixel_weight(pred_logits, epoch, burnin_epochs, Tc, W_min=pue_wmin)
    W_all = weit * Wpix

    bce = F.binary_cross_entropy_with_logits(pred_logits, mask, reduction='none')
    wbce = (W_all * bce).sum(dim=(2, 3)) / (W_all.sum(dim=(2, 3)) + 1e-6)  # [B,1]

    pred = torch.sigmoid(pred_logits)
    inter = ((pred * mask) * W_all).sum(dim=(2, 3))
    union = ((pred + mask) * W_all).sum(dim=(2, 3))
    wiou = 1 - (inter + 1.0) / (union - inter + 1.0)  # [B,1]

    loss_vec = (wbce + wiou).squeeze(1)  # [B]
    return loss_vec


# -----------------------------
# Curriculum utilities
# -----------------------------
def batch_soft_iou_from_logits(logits, gts, eps=1e-6):
    p = torch.sigmoid(logits)
    g = gts.float()
    inter = (p * g).sum(dim=(1, 2, 3))
    union = (p + g - p * g).sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    return iou  # [B]


@torch.no_grad()
def compute_difficulty_by_model(model, full_loader, device, final_idx=4):
    """
    difficulty = 1 - soft IoU
    """
    model.eval()
    d_map = {}

    for images, gts, edges, idxs in full_loader:
        images = images.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)

        preds = model(images)
        iou = batch_soft_iou_from_logits(preds[final_idx], gts)
        d = 1.0 - iou  # [B]

        if torch.is_tensor(idxs):
            idxs = idxs.detach().cpu().tolist()

        for j, idx in enumerate(idxs):
            d_map[int(idx)] = float(d[j].item())

    return d_map


def curriculum_percentile(epoch, burnin_epochs=10):
    """
    burn-in (1..burnin_epochs): full set
    after that:
      start at 0.50
      +10% every 10 epochs (as in your code)
    """
    if epoch <= burnin_epochs:
        return 1.0
    inc = (epoch - (burnin_epochs + 1)) // 10
    p = 0.50 + 0.10 * inc
    return float(min(max(p, 0.50), 1.0))


def select_indices_by_percentile(d_map, p):
    idxs = np.array(list(d_map.keys()), dtype=np.int64)
    ds = np.array([d_map[i] for i in idxs], dtype=np.float32)
    if len(ds) == 0:
        return [], 0.0
    thr = float(np.quantile(ds, p))
    chosen = idxs[ds <= thr]
    return chosen.tolist(), thr


# -----------------------------
# TSSW: Temporal Statistics-based Sample Weighting
# -----------------------------
class TSSWBuffer:
    """
    Store sample-wise error history e_i = 1 - IoU over last K epochs.
    Use temporal mean/var to build sample weights omega_i.
    """
    def __init__(self, K=10, wmin=0.1, sigma_star=0.5, gamma=0.2):
        self.K = int(K)
        self.wmin = float(wmin)
        self.sigma_star = float(sigma_star)
        self.gamma = float(gamma)
        self.hist = {}  # idx -> deque(maxlen=K)

    def update_batch(self, idxs, errors):
        if torch.is_tensor(errors):
            errors = errors.detach().cpu().tolist()
        for i, e in zip(idxs, errors):
            i = int(i)
            if i not in self.hist:
                self.hist[i] = deque(maxlen=self.K)
            self.hist[i].append(float(e))

    def get_mu_var(self, indices):
        mu = {}
        var = {}
        for i in indices:
            i = int(i)
            if i in self.hist and len(self.hist[i]) > 0:
                arr = np.array(self.hist[i], dtype=np.float32)
                mu[i] = float(arr.mean())
                var[i] = float(((arr - arr.mean()) ** 2).mean())
        return mu, var

    def compute_weights(self, indices):
        """
        Your paper formula (with error mean/var):
        - normalize mu (error) and var
        - omega_mu = 1 - mu~
        - omega_sigma = exp(-((var~ - sigma*)^2)/(2 gamma^2))
        - omega_out = 1 - mu~ * (1 - var~)
        - omega = wmin + (1-wmin) * omega_mu * omega_sigma * omega_out
        """
        indices = [int(i) for i in indices]
        mu_map, var_map = self.get_mu_var(indices)

        # too few history -> fallback weights=1
        if len(mu_map) < 8:
            return {i: 1.0 for i in indices}

        mu_vals = np.array(list(mu_map.values()), dtype=np.float32)
        var_vals = np.array(list(var_map.values()), dtype=np.float32)

        mu_min, mu_max = float(mu_vals.min()), float(mu_vals.max())
        v_min, v_max = float(var_vals.min()), float(var_vals.max())

        def _minmax(x, mn, mx):
            return (x - mn) / (mx - mn + 1e-8)

        omega = {}
        for i in indices:
            if i not in mu_map:
                omega[i] = 1.0
                continue

            mu_t = _minmax(mu_map[i], mu_min, mu_max)   # \tilde{\mu}_i (error mean)
            v_t = _minmax(var_map[i], v_min, v_max)     # \tilde{\sigma}_i^2 (error var)

            w_mu = 1.0 - mu_t
            w_sigma = np.exp(-((v_t - self.sigma_star) ** 2) / (2.0 * (self.gamma ** 2) + 1e-12))
            w_out = 1.0 - mu_t * (1.0 - v_t)

            w = self.wmin + (1.0 - self.wmin) * (w_mu * w_sigma * w_out)
            omega[i] = float(np.clip(w, self.wmin, 1.0))

        return omega


# -----------------------------
# Validation
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

        if epoch == 1:
            best_mae = mae
            best_epoch = 1
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_path, 'Net_epoch_best.pth'))
                print(f'[Val] Save best state_dict! Best epoch: {epoch}')

        logging.info(f'[Val Info]:Epoch:{epoch} MAE:{mae} bestEpoch:{best_epoch} bestMAE:{best_mae}')


# -----------------------------
# Training (Curriculum + TSSW + PUE)
# -----------------------------
def train_with_curriculum_learning(
    train_loader,
    model,
    optimizer,
    epoch,
    save_path,
    writer,
    K=10,
    burnin_epochs=10,
    difficulty_cache=None,
    curriculum_state=None,
    tssw_buffer=None,
    num_workers=8,
    debug_runtime_idx_every=200,
    debug_print_diff_hist=False,
):
    global step

    model.train()
    device = next(model.parameters()).device
    mse_loss = nn.MSELoss()

    train_dataset = train_loader.dataset
    dataset_size = len(train_dataset)

    # loader for difficulty computation (no shuffle)
    full_loader = DataLoader(
        train_dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if difficulty_cache is None:
        difficulty_cache = {}

    if curriculum_state is None:
        curriculum_state = {"active_set": None, "last_k_use": None}

    if tssw_buffer is None:
        tssw_buffer = TSSWBuffer(
            K=opt.tssw_K, wmin=opt.tssw_wmin,
            sigma_star=opt.tssw_sigma_star, gamma=opt.tssw_gamma
        )

    # ---- Decide active loader ----
    active_indices = None

    if epoch <= burnin_epochs:
        active_loader = train_loader
        active_indices = None
        print(f'[Burn-in] epoch={epoch:03d} using FULL set |S_t|={dataset_size}')
        logging.info(f'[Burn-in] epoch={epoch} using FULL set |S_t|={dataset_size}')
    else:
        k_use = ((epoch - 1) // K) * K  # epoch=11..20 -> 10, epoch=21..30 -> 20 ...

        # Load difficulty map for k_use
        if k_use in difficulty_cache:
            d_map = difficulty_cache[k_use]
        else:
            d_path = os.path.join(save_path, f'difficulty_epoch_{k_use}.npy')
            if os.path.exists(d_path):
                d_map = np.load(d_path, allow_pickle=True).item()
                difficulty_cache[k_use] = d_map
            else:
                print(f"[Difficulty] ⚠️ Missing {d_path}, compute difficulty ONLINE now (fallback).")
                logging.info(f"[Difficulty] Missing {d_path}, compute ONLINE now (fallback).")
                d_map = compute_difficulty_by_model(model, full_loader, device, final_idx=4)
                difficulty_cache[k_use] = d_map
                np.save(d_path, d_map, allow_pickle=True)
                print(f"[Difficulty] Fallback saved: {d_path} (#samples={len(d_map)})")

        # Monotonic expansion update at block boundary
        if curriculum_state["last_k_use"] != k_use:
            p = curriculum_percentile(epoch, burnin_epochs=burnin_epochs)
            chosen, thr = select_indices_by_percentile(d_map, p)
            chosen_set = set(chosen)

            if curriculum_state["active_set"] is None:
                curriculum_state["active_set"] = set()
            curriculum_state["active_set"] |= chosen_set  # UNION

            curriculum_state["last_k_use"] = k_use

            print(f"[Curriculum-Update] epoch={epoch:03d} use_diff@{k_use} p={p:.2f} "
                  f"new_add={len(chosen_set)} |S_t|={len(curriculum_state['active_set'])}/{dataset_size} thr={thr:.6f}")
            logging.info(f"[Curriculum-Update] epoch={epoch} use_diff@{k_use} p={p:.2f} "
                         f"new_add={len(chosen_set)} |S_t|={len(curriculum_state['active_set'])}/{dataset_size} thr={thr:.6f}")

        active_indices = sorted(list(curriculum_state["active_set"]))
        if len(active_indices) == 0:
            active_indices = list(range(dataset_size))

        active_loader = DataLoader(
            Subset(train_dataset, active_indices),
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        print(f"[Curriculum] epoch={epoch:03d} |S_t|={len(active_indices)}/{dataset_size} (monotonic union)")
        logging.info(f"[Curriculum] epoch={epoch} |S_t|={len(active_indices)}/{dataset_size} (monotonic union)")

    # ---- Recon switch ----
    if epoch <= burnin_epochs:
        print(f"[TSSW/PUE] epoch={epoch:03d} burn-in: PUE=OFF, omega=1")
    else:
        print(f"[TSSW/PUE] epoch={epoch:03d} curriculum: PUE=ON, omega=ON")

    is_full_set = (epoch > burnin_epochs) and (active_indices is not None) and (len(active_indices) >= dataset_size)
    print(f"[Recon Switch] epoch={epoch:03d} | is_full_set={is_full_set} | "
          f"|S_t|={dataset_size if active_indices is None else len(active_indices)}/{dataset_size}")

    # ---- TSSW weights map for current subset ----
    if epoch <= burnin_epochs:
        cur_indices_for_weight = list(range(dataset_size))
    elif active_indices is None or len(active_indices) == 0:
        cur_indices_for_weight = list(range(dataset_size))
    else:
        cur_indices_for_weight = active_indices

    # ---- TSSW omega map for current subset ----
    if epoch <= burnin_epochs:
        # ✅ burn-in：关闭 TSSW（所有样本权重=1）
        omega_map = {}
    else:
        omega_map = tssw_buffer.compute_weights(cur_indices_for_weight)

    # ---- Train loop ----
    loss_all = 0.0
    epoch_step = 0

    for it, (images, gts, edges, idxs) in enumerate(active_loader, start=1):
        optimizer.zero_grad(set_to_none=True)

        if (it == 1) or (debug_runtime_idx_every > 0 and it % debug_runtime_idx_every == 0):
            runtime_check_batch_idxs(idxs, dataset_size, prefix=f"[IDX RUNTIME][E{epoch:03d}][It{it:04d}]")

        images = images.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)
        edges = edges.to(device, non_blocking=True)

        preds = model(images)

        total_step = max(len(active_loader), 1)

        # ---- UAL ----
        ual_coef = get_coef(iter_percentage=it / float(total_step), method='cos')
        ual_loss = cal_ual(seg_logits=preds[4], seg_gts=gts)
        ual_loss = ual_loss * ual_coef

        # batch idx list
        if torch.is_tensor(idxs):
            idx_list = idxs.detach().cpu().tolist()
        else:
            idx_list = list(idxs)

        # ---- Update TSSW history using current batch error = 1 - IoU ----
        with torch.no_grad():
            cur_iou = batch_soft_iou_from_logits(preds[4], gts)  # [B]
            cur_err = 1.0 - cur_iou
            tssw_buffer.update_batch(idx_list, cur_err)

        # ---- TSSW omega for batch ----
        omega_batch = torch.tensor(
            [omega_map.get(int(i), 1.0) for i in idx_list],
            dtype=torch.float32, device=device
        )  # [B]

        # ---- PUE structure loss vectors ----
        L0 = structure_loss_pue(preds[0], gts, epoch, burnin_epochs, opt.pue_Tc, pue_wmin=opt.pue_wmin)
        L1 = structure_loss_pue(preds[1], gts, epoch, burnin_epochs, opt.pue_Tc, pue_wmin=opt.pue_wmin)
        L2 = structure_loss_pue(preds[2], gts, epoch, burnin_epochs, opt.pue_Tc, pue_wmin=opt.pue_wmin)
        L3 = structure_loss_pue(preds[3], gts, epoch, burnin_epochs, opt.pue_Tc, pue_wmin=opt.pue_wmin)
        L4 = structure_loss_pue(preds[4], gts, epoch, burnin_epochs, opt.pue_Tc, pue_wmin=opt.pue_wmin)

        # ---- apply TSSW sample weights ----
        loss_init_vec = (L0 * 0.0625 + L1 * 0.125 + L2 * 0.25 + L3 * 0.5)  # [B]
        loss_final_vec = L4  # [B]

        loss_init = (loss_init_vec * omega_batch).mean()
        loss_final = (loss_final_vec * omega_batch).mean()

        # ---- Edge loss (unchanged, as requested) ----
        loss_edge = (dice_loss(preds[5], edges) * 0.125 +
                     dice_loss(preds[6], edges) * 0.25 +
                     dice_loss(preds[7], edges) * 0.5)

        # ---- total loss ----
        loss = loss_init + loss_final + loss_edge + 2.0 * ual_loss

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        epoch_step += 1
        loss_all += float(loss.item())
        step += 1

        if it % 20 == 0 or it == len(active_loader) or it == 1:
            print(f'{datetime.now()} Epoch [{epoch}/{opt.epoch}] Step [{it}/{len(active_loader)}] '
                  f'Loss: {loss.item():.4f}')
            logging.info(f'[Train Info]:Epoch [{epoch}/{opt.epoch}] Step [{it}/{len(active_loader)}] '
                         f'Loss: {loss.item():.6f}')

            writer.add_scalars('Loss_Statistics', {
                'Loss_total': loss.item(),
                'Loss_init': float(loss_init.item()),
                'Loss_final': float(loss_final.item()),
                'Loss_edge': float(loss_edge.item()),
            }, global_step=step)

    loss_all /= max(epoch_step, 1)
    writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
    logging.info(f'[Train Info]: Epoch [{epoch}/{opt.epoch}] Loss_AVG: {loss_all:.6f}')

    # ---- END-OF-EPOCH: save checkpoint & compute difficulty every K epochs ----
    if epoch % K == 0:
        ckpt_path = os.path.join(save_path, f'Net_epoch_{epoch}.pth')
        torch.save(model.state_dict(), ckpt_path)
        logging.info(f'[CKPT] Saved checkpoint: {ckpt_path}')
        print(f'[CKPT] Saved checkpoint: {ckpt_path}')

        d_map = compute_difficulty_by_model(model, full_loader, device, final_idx=4)
        difficulty_cache[epoch] = d_map

        d_save = os.path.join(save_path, f'difficulty_epoch_{epoch}.npy')
        np.save(d_save, d_map, allow_pickle=True)

        print(f'[Difficulty] Saved: {d_save} (#samples={len(d_map)})')
        logging.info(f'[Difficulty] Saved: {d_save} (#samples={len(d_map)})')

        # ✅ dump tssw history for later visualization (paper figures)
        tssw_path = os.path.join(save_path, f"tssw_hist_epoch_{epoch}.npy")
        np.save(tssw_path, tssw_buffer.hist, allow_pickle=True)
        print(f"[TSSW] Saved history: {tssw_path}")
        logging.info(f"[TSSW] Saved history: {tssw_path}")

        stat = summarize_difficulty(d_map, tag=f"epoch={epoch}", print_hist=debug_print_diff_hist)

        with open(os.path.join(save_path, "difficulty_stats.txt"), "a") as f:
            f.write(
                f"epoch={epoch} N={stat['N']} min={stat['min']:.6f} p10={stat['p10']:.6f} p50={stat['p50']:.6f} "
                f"p90={stat['p90']:.6f} max={stat['max']:.6f} mean={stat['mean']:.6f} std={stat['std']:.6f}\n"
            )

        prev_epoch = epoch - K
        if prev_epoch in difficulty_cache:
            S_prev = easy_subset_from_dmap(difficulty_cache[prev_epoch], p=0.5)
            S_now = easy_subset_from_dmap(d_map, p=0.5)
            jac = jaccard(S_prev, S_now)
            print(f"[CURRI STUCK CHECK] Jaccard(easy@{prev_epoch}, easy@{epoch}) = {jac:.4f}")
            logging.info(f"[CURRI STUCK CHECK] Jaccard(easy@{prev_epoch}, easy@{epoch}) = {jac:.6f}")

            if jac > 0.98:
                print("[CURRI STUCK CHECK] ⚠️ Easy subset almost unchanged -> curriculum may be stuck.")
            if jac < 0.30:
                print("[CURRI STUCK CHECK] ⚠️ Easy subset changed too much -> check idx consistency / difficulty quality.")

    return difficulty_cache, curriculum_state, tssw_buffer


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=36, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training image size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

    parser.add_argument('--train_root', type=str, default='',
                        help='training dataset root (contains Imgs/ GT/ Edge/)')
    parser.add_argument('--val_root', type=str, default='',
                        help='validation dataset root (contains Imgs/ GT/)')
    parser.add_argument('--save_path', type=str,
                        default='',
                        help='path to save model and log')

    # Curriculum params
    parser.add_argument('--K', type=int, default=10, help='checkpoint interval K')
    parser.add_argument('--burnin_epochs', type=int, default=10, help='burn-in epochs using FULL set')
    parser.add_argument('--num_workers', type=int, default=16, help='dataloader workers')

    # Debug params
    parser.add_argument('--debug_idx_every', type=int, default=200, help='runtime idx check interval (iters)')
    parser.add_argument('--debug_diff_hist', action='store_true', help='print difficulty histogram')

    # -------- TSSW params --------
    parser.add_argument('--tssw_K', type=int, default=10, help='TSSW history window length')
    parser.add_argument('--tssw_wmin', type=float, default=0.1, help='TSSW min sample weight')
    parser.add_argument('--tssw_sigma_star', type=float, default=0.5, help='TSSW optimal variance level (normalized)')
    parser.add_argument('--tssw_gamma', type=float, default=0.2, help='TSSW gaussian tolerance')

    # -------- PUE params --------
    parser.add_argument('--pue_wmin', type=float, default=0.1, help='PUE min pixel weight')
    parser.add_argument('--pue_Tc', type=int, default=200, help='PUE curriculum length Tc (epochs)')

    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    cudnn.benchmark = True

    save_path = opt.save_path
    os.makedirs(save_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(save_path, 'log.log'),
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO,
        filemode='a',
        datefmt='%Y-%m-%d %I:%M:%S %p'
    )

    logging.info('Network-Train (CurriSeg Union + TSSW + PUE)')
    logging.info(
        'Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; '
        'K: {}; burnin_epochs: {}; train_root: {}; val_root: {}; save_path: {}; '
        'tssw_K: {}; tssw_wmin: {}; tssw_sigma_star: {}; tssw_gamma: {}; '
        'pue_wmin: {}; pue_Tc: {}'.format(
            opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
            opt.K, opt.burnin_epochs,
            opt.train_root, opt.val_root, save_path,
            opt.tssw_K, opt.tssw_wmin, opt.tssw_sigma_star, opt.tssw_gamma,
            opt.pue_wmin, opt.pue_Tc
        )
    )

    # Build model
    model = Network(channels=192).cuda(device=device_ids[0])

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

    difficulty_cache = {}
    curriculum_state = {"active_set": None, "last_k_use": None}
    tssw_buffer = TSSWBuffer(
        K=opt.tssw_K, wmin=opt.tssw_wmin,
        sigma_star=opt.tssw_sigma_star, gamma=opt.tssw_gamma
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=opt.epoch, eta_min=1e-5
    )

    print('[Train] Start...')
    for epoch in range(1, opt.epoch + 1):
        cur_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning_rate/lr', cur_lr, global_step=epoch)
        logging.info(f'>>> current lr: {cur_lr}')

        difficulty_cache, curriculum_state, tssw_buffer = train_with_curriculum_learning(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            save_path=save_path,
            writer=writer,
            K=opt.K,
            burnin_epochs=opt.burnin_epochs,
            difficulty_cache=difficulty_cache,
            curriculum_state=curriculum_state,
            tssw_buffer=tssw_buffer,
            num_workers=opt.num_workers,
            debug_runtime_idx_every=opt.debug_idx_every,
            debug_print_diff_hist=opt.debug_diff_hist,
        )

        # Validation schedule
        if epoch <= 70:
            do_val = (epoch % 1 == 0)
        else:
            do_val = True

        if do_val:
            val(val_loader, model, epoch, save_path, writer)

        scheduler.step()

    writer.close()
    print('[Done] Training finished.')

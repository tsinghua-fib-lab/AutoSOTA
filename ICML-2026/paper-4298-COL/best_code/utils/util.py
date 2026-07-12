import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import datetime
import wandb
from copy import deepcopy
from scipy import stats


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ClassWiseAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, n_cls):
        self.n_cls = n_cls
        self.reset()

    def reset(self):
        self.val = np.zeros([self.n_cls,])
        self.avg = np.zeros([self.n_cls,])
        self.sum = np.zeros([self.n_cls,])
        self.count = np.ones([self.n_cls,]) * 1e-7
        self.total_avg = 0

    def update(self, val, n=[1,1,1]):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.total_avg = np.sum(self.sum) / np.sum(self.count)


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def write_log(log_file, out_str):
    log_file.write(out_str + '\n')
    log_file.flush()
    print(out_str)



def load_one_image(img_path, width=256, height=256):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))
    return img


def load_images(img_root, img_name_list, width=256, height=256):
    num_images = len(img_name_list)
    images = np.zeros([num_images, height, width, 3], dtype=np.uint8)
    for idx, img_path in enumerate(img_name_list):
        img = cv2.imread(os.path.join(img_root, img_path), cv2.IMREAD_COLOR)
        images[idx] = cv2.resize(img, (width, height))
    return images

def to_np(x):
    return x.cpu().detach().numpy()


def get_current_time():
    _now = datetime.now()
    _now = str(_now)[:-7]
    return _now


def display_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['lr'], param_group['initial_lr'])

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def log_configs(cfg, log_file='log.txt'):
    if os.path.exists(f'{cfg.save_folder}/{log_file}'):
        log_file = open(f'{cfg.save_folder}/{log_file}', 'a')
    else:
        log_file = open(f'{cfg.save_folder}/{log_file}', 'w')
    opt_dict = vars(cfg)
    for key in opt_dict.keys():
        write_log(log_file, f'{key}: {opt_dict[key]}')
    return log_file


def save_ckpt(cfg, model, postfix):
    state = {
        'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
    }
    save_file = os.path.join(cfg.save_folder, f'{postfix}')
    torch.save(state, save_file)
    print(f'ckpt saved to {save_file}.')


def set_wandb(cfg, key='private_key'):
    wandb.login(key=key)
    wandb.init(project=cfg.experiment_name, tags=[cfg.dataset])
    wandb.config.update(cfg)
    wandb.save('*.py')
    wandb.run.save()


def extract_embs(encoder, data_loader, cfg):
    encoder.eval()
    embs = []
    inds = []
    with torch.no_grad():
        for x_base, _, item in data_loader:
            x_base = x_base.to(cfg.device)
            feat = encoder(x_base)
            embs.append(feat.cpu())
            inds.append(item)
    embs = torch.cat(embs)
    inds = torch.cat(inds)
    embs_temp = deepcopy(embs)
    embs[inds] = embs_temp

    return embs

def to_dtype(x, tensor=None, dtype=None):
    if not torch.is_autocast_enabled():
        dt = dtype if dtype is not None else tensor.dtype
        if x.dtype != dt:
            x = x.type(dt)
    return x

def to_device(x, tensor=None, device=None, dtype=None):
    dv = device if device is not None else tensor.device
    if x.device != dv:
        x = x.to(dv)
    if dtype is not None:
        x = to_dtype(x, dtype=dtype)
    return x


def print_eval_result_by_groups_and_k(gt, ref_gt, preds_all, log_file, interval=10):
    test_cls_arr, cnt = np.unique(gt, return_counts=True)
    test_cls_min = test_cls_arr.min()
    test_cls_max = test_cls_arr.max()
    n_groups = int((test_cls_max - test_cls_min + 1) / interval + 0.5)

    title = 'Group \\ K |'
    for k in preds_all.keys():
        title += f" {k:<4} "
    title = title + ' | Best K | #Test | #Train '
    write_log(log_file, title)
    for i_group in range(n_groups):
        min_rank = interval * i_group
        max_rank = min(test_cls_max + 1, min_rank + interval)
        sample_idx_in_group = np.argwhere(np.logical_and(gt >= min_rank, gt < max_rank)).flatten()
        ref_sample_idx_in_group = np.argwhere(np.logical_and(ref_gt >= min_rank, ref_gt < max_rank)).flatten()

        if len(sample_idx_in_group) < 1:
            continue
        to_print = f' {min_rank:<3}~ {max_rank - 1:<3} |'

        best_k = -1
        best_mae = 1000
        for k in preds_all.keys():
            i_group_errors_at_k = np.abs(preds_all[k][sample_idx_in_group] - gt[sample_idx_in_group])
            i_group_mean_at_k = np.mean(i_group_errors_at_k)
            to_print += f' {i_group_mean_at_k:.3f}' if i_group_mean_at_k<10 else f' {i_group_mean_at_k:.2f}'
            if i_group_mean_at_k < best_mae:
                best_mae = i_group_mean_at_k
                best_k = k
        to_print += f' |   {best_k:<2}   | {len(sample_idx_in_group):<4}  | {len(ref_sample_idx_in_group):<4} '
        write_log(log_file, to_print)

    mean_all = '  Total   |'
    best_k = -1
    best_mae = 1000
    for k in preds_all.keys():
        mean_at_k = np.mean(np.abs(preds_all[k] - gt))
        mean_all += f' {mean_at_k:.3f}'
        if mean_at_k < best_mae:
            best_mae = mean_at_k
            best_k = k
    mean_all += f' |   {best_k:<2}   | {len(gt):<5} | {len(ref_gt):<5}'
    write_log(log_file, mean_all)
    write_log(log_file, f'Best Total MAE : {best_mae:.3f}\n')
    return best_mae, best_k


def evaluate_metric(pred_age, gt_age, cs_th=5):
    MAE = np.mean(np.abs(np.subtract(gt_age, pred_age)))
    CS = np.sum(np.abs(pred_age - gt_age) <= cs_th) / float(len(gt_age))
    acc = np.sum(gt_age == pred_age) / len(gt_age)
    return MAE, CS, acc

def cal_srocc_plcc(pred_score, gt_score):
    try:
        srocc, _ = stats.spearmanr(pred_score, gt_score)
        plcc, _ = stats.pearsonr(pred_score, gt_score)
    except:
        srocc, plcc = 0, 0

    return srocc, plcc



def print_eval_result_by_groups_and_k_IQA(gt, ref_gt, preds_all, log_file, interval=10):
    test_cls_arr, cnt = np.unique(gt, return_counts=True)
    test_cls_min = test_cls_arr.min()
    test_cls_max = test_cls_arr.max()
    n_groups = int((test_cls_max - test_cls_min + 1) / interval + 0.5)

    title = 'Group \\ K |'
    for k in preds_all.keys():
        title += f" {k:<4} "
    title = title + ' | Best K | #Test | #Train '
    write_log(log_file, title)
    for i_group in range(n_groups):
        min_rank = interval * i_group
        max_rank = min(test_cls_max + 1, min_rank + interval)
        sample_idx_in_group = np.argwhere(np.logical_and(gt >= min_rank, gt < max_rank)).flatten()
        ref_sample_idx_in_group = np.argwhere(np.logical_and(ref_gt >= min_rank, ref_gt < max_rank)).flatten()

        if len(sample_idx_in_group) < 1:
            continue
        to_print = f' {min_rank:<3}~ {max_rank - 1:<3} |'

        best_k = -1
        best_srcc = 0
        for k in preds_all.keys():
            i_group_metrics_at_k = cal_srocc_plcc(preds_all[k][sample_idx_in_group], gt[sample_idx_in_group])

            to_print += f' {i_group_metrics_at_k[0]:.4f}'
            if i_group_metrics_at_k[0] > best_srcc:
                best_srcc = i_group_metrics_at_k[0]
                best_plcc =  i_group_metrics_at_k[1]
                best_k = k
        to_print += f' |   {best_k:<2}   | {len(sample_idx_in_group):<4}  | {len(ref_sample_idx_in_group):<4} '
        write_log(log_file, to_print)

    mean_all = '  Total   |'
    best_k = -1
    best_srcc = 0
    for k in preds_all.keys():
        metrics_at_k = cal_srocc_plcc(preds_all[k], gt)
        mean_all += f'{metrics_at_k[0]:.4f}'
        if metrics_at_k[0] > best_srcc:
            best_srcc = metrics_at_k[0]
            best_plcc = metrics_at_k[1]
            best_k = k
    mean_all += f' |   {best_k:<2}   | {len(gt):<5} | {len(ref_gt):<5}'
    write_log(log_file, mean_all)
    write_log(log_file, f'Best Total SRCC : {best_srcc:.4f}\n')
    return best_srcc, best_plcc, best_k


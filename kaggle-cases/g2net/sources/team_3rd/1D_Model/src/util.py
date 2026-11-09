import random
import os
import numpy as np
import torch
import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR
from transformers import get_cosine_schedule_with_warmup
from torch.optim.swa_utils import AveragedModel
from numba import jit


@jit(forceobj=True)
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def get_swa(model, optimizer, epochs, swa_start_step_epoch, swa_lr, train_size, batch_size):
    swa_model = AveragedModel(model)
    epoch_step = train_size / batch_size
    swa_start_step = epoch_step * swa_start_step_epoch
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer,
                                                anneal_strategy="cos",
                                                anneal_epochs=int(epoch_step * (epochs - 1)),
                                                swa_lr=swa_lr)
    return swa_model, swa_scheduler


def do_swa_scheduler(step, swa_scheduler, swa_start_step):
    if (swa_scheduler is not None) and (step >= swa_start_step):
        return True
    else:
        return False


def get_score(y_true, y_pred):
    score = fast_auc(y_true, y_pred)
    return score


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_scheduler(optimizer, train_size, Config):
    if Config.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=Config.factor,
                                      patience=Config.patience, verbose=True, eps=Config.eps)
    elif Config.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=Config.T_max,
                                      eta_min=Config.min_lr, last_epoch=-1)
    elif Config.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=Config.T_0,
                                                T_mult=1,
                                                eta_min=Config.min_lr,
                                                last_epoch=-1)
    elif Config.scheduler == 'CyclicLR':
        iter_per_ep = train_size / Config.batch_size
        step_size_up = int(iter_per_ep * Config.step_up_epochs)
        step_size_down = int(iter_per_ep * Config.step_down_epochs)
        scheduler = CyclicLR(optimizer,
                             base_lr=Config.base_lr,
                             max_lr=Config.max_lr,
                             step_size_up=step_size_up,
                             step_size_down=step_size_down,
                             mode=Config.mode,
                             gamma=Config.cycle_decay ** (1 / (step_size_up + step_size_down)),
                             cycle_momentum=False)
    elif Config.scheduler == 'cosineWithWarmUp':
        epoch_step = train_size / Config.batch_size
        num_warmup_steps = int(Config.warmup * epoch_step * Config.epochs)
        num_training_steps = int(epoch_step * Config.epochs)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)
    return scheduler


def mixed_criterion(loss_fn, pred, y_a, y_b, lam):
    return lam * loss_fn(pred, y_a) + (1 - lam) * loss_fn(pred, y_b)


def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size, requires_grad=False).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


class Dict2Class():
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def id_2_path_wave(idx, Config, train=True) -> str:
    if Config.use_raw_wave:
        if train:
            return "{}/train/{}/{}/{}/{}.npy".format(Config.kaggleDataFolder, idx[0], idx[1], idx[2], idx)
        else:
            return "{}/test/{}/{}/{}/{}.npy".format(Config.kaggleDataFolder, idx[0], idx[1], idx[2], idx)
    else:
        if train:
            return "{}/{}.npy".format(Config.whiten_train_folder, idx)
        else:
            return "{}/{}.npy".format(Config.whiten_test_folder, idx)


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print('Number of device:', torch.cuda.device_count())
    return device


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def get_key(path):
    f = open(path, "r")
    key = f.read().strip()
    f.close()
    return key


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

import os
import time
import sys
from copy import deepcopy

import numpy as np
import wandb
import random
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
from collections import defaultdict

from config.basic import ConfigBasic
from utils.util import write_log, get_current_time, to_np, make_dir, log_configs, save_ckpt, set_wandb
from utils.util import AverageMeter, extract_embs, print_eval_result_by_groups_and_k, evaluate_metric
from utils.loss_util import ConOrdLoss, compute_center_loss
from utils.comparison_utils import find_kNN
from networks.util import prepare_model
from data.get_datasets_AGE import get_datasets_AGE

import argparse

def parse_args():
    ############ cmd process ##############
    parser = argparse.ArgumentParser(description='SOL')
    parser.add_argument('--dataset', type=str, default='clap',
                        help='dataset')
    parser.add_argument('--gpu', type=int, default=0,
                        help="train on which cuda device")
    parser.add_argument('--epochs', type=int, default=30,
                        help="total number of epochs to train")
    parser.add_argument('--lr', type=float, default=5e-6,
                        help='learning_rate')
    ########## cmd end ############

    args = parser.parse_args()

    return args

def set_local_config(cfg, args):
    # Dataset
    cfg.dataset = args.dataset
    cfg.logscale = False
    cfg.set_dataset()
    cfg.tau = 0

    # Model
    cfg.model = 'ConOrd'
    cfg.backbone = 'vitB16'

    cfg.k = np.arange(2, 60, 2)
    cfg.scheduler = 'cosine'
    cfg.lr_decay_epochs = [100, 200, 300]

    cfg.margin = 0.05
    cfg.ref_mode = 'flex'
    cfg.ref_point_num = 60  # 60 Fold1, 58 Fold0 setting D // 56 setting c // 58 setting B // 55 setting A
    cfg.drct_wieght = 1
    cfg.start_norm = True
    cfg.epochs = args.epochs
    cfg.learning_rate = args.lr
      
    cfg.metric = 'L2'
    cfg.label_diff = 'l2'
    cfg.similarity_type = 'L2'
    cfg.epsilon = 1e-7
    cfg.temp = 0.07

    cfg.lr_decay_rate = 0.0005
    cfg.batch_size = 128
    cfg.test_batch_size = 1000
    # Log
    cfg.wandb = False
    cfg.experiment_name = f'ContrastiveOrder+centerLoss_centerMetric{cfg.metric}_FeatSim{cfg.similarity_type}_epsilon{cfg.epsilon}_Temp{cfg.temp}_lrDecay{cfg.lr_decay_rate}_labelDiff{cfg.label_diff}'
    cfg.save_folder = f'../results_v3_ConOrd/{cfg.dataset}/{cfg.experiment_name}/Epochs{cfg.epochs}_{cfg.model}_{cfg.backbone}_lr{cfg.learning_rate}_{get_current_time()}'
    make_dir(cfg.save_folder)


    cfg.n_gpu = torch.cuda.device_count()
    cfg.num_workers = 1
    cfg.gpu_ids = [args.gpu]
    cfg.device = torch.device(f'cuda:{cfg.gpu_ids[0]}')
    return cfg


def main():

    random_seed = 999
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    cfg = ConfigBasic()
    args = parse_args()
    cfg = set_local_config(cfg, args)
    cfg.logfile = log_configs(cfg, log_file='train_log.txt')

    # dataloader
    loader_dict = get_datasets_AGE(cfg)
    cfg.n_ranks = loader_dict['train'].dataset.ranks.max() + 1
    print(f'[*] {cfg.n_ranks} ranks exist. ')
    ####################################################
    cfg.ref_point_num = cfg.n_ranks
    cfg.fiducial_point_num = cfg.n_ranks
    #####################################################

    # model
    model = prepare_model(cfg)
    if cfg.wandb:
        set_wandb(cfg)
        wandb.watch(model)

    param_groups = []
    for key, value in dict(model.named_children()).items():
        param_groups += [{"params": value.parameters(), "lr": cfg.learning_rate}]
    param_groups += [{"params": model.ref_points, "lr": cfg.learning_rate * 10}]

    optimizer = optim.Adam(params=param_groups,
                           lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)

    if cfg.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs, eta_min=cfg.learning_rate*cfg.lr_decay_rate)
    elif cfg.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_decay_epochs, gamma=cfg.lr_decay_rate)
    else:
        scheduler = None

    criterion = ConOrdLoss(label_diff=cfg.label_diff, feature_sim=cfg.similarity_type, temperature=cfg.temp)

    if torch.cuda.is_available():
        # if cfg.n_gpu > 1:
        #     model = nn.DataParallel(model)
        model = model.to(cfg.device)

    val_mae_best = np.inf
    log_dict = dict()
    # init loss matrix
    loss_record = dict()
    loss_record['angle'] = [np.zeros([cfg.n_ranks, cfg.n_ranks]), np.zeros([cfg.n_ranks, cfg.n_ranks])]

    for epoch in range(cfg.epochs):
        print("==> training...")

        time1 = time.time()
        train_loss, loss_record = train(epoch, loader_dict['train'], model, optimizer,criterion, cfg,
                                                     prev_loss_record=loss_record)

        if cfg.scheduler:
            scheduler.step()
        time2 = time.time()
        print('epoch {}, loss {:.4f}, total time {:.2f}'.format(epoch, train_loss, time2 - time1))

        print('==> validation...')
        val_mae, val_cs, best_k = validate_AGE(loader_dict, model, cfg)
        if val_mae < val_mae_best:
            val_mae_best = val_mae
            save_ckpt(cfg, model, f'ep_{epoch}_MAE_{val_mae:.3f}_CS_{val_cs:.4f}_k{best_k}.pth')

        if cfg.wandb:
            log_dict['Epoch'] = epoch
            log_dict['Train Loss'] = train_loss
            log_dict['Val Mae'] = val_mae
            log_dict['LR'] = scheduler.get_lr()[0] if scheduler else cfg.learning_rate
            wandb.log(log_dict)

    print('[*] Training ends')


def train(epoch, train_loader, model, optimizer, criterion, cfg, prev_loss_record):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dist_losses = AverageMeter()
    center_losses = AverageMeter()

    loss_record = deepcopy(prev_loss_record)
    end = time.time()


    for idx, (images, _, ranks, _) in enumerate(train_loader):
        bsz = ranks.shape[0] // 2

        if torch.cuda.is_available():
            images = images.to(cfg.device)
            ranks = ranks.to(cfg.device)

        data_time.update(time.time() - end)

        # ===================forward=====================
        features = model.encoder(images)
        features = nn.functional.normalize(features, dim=-1)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        # =====================loss======================
        # compute loss

        start = time.time()
        dist_loss = criterion(features, ranks, cfg)
        center_loss = compute_center_loss(torch.cat(torch.unbind(features, dim=1), dim=0), ranks, model.ref_points, cfg)
        total_loss = dist_loss + center_loss

        losses.update(total_loss.item(), ranks.size(0))
        dist_losses.update(dist_loss.item(), ranks.size(0))
        center_losses.update(center_loss.item(), ranks.size(0))
        # ===================backward=====================
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % cfg.print_freq == 0:
            write_log(cfg.logfile,
                      f'Epoch [{epoch}][{idx}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f}\t'
                      f'Data {data_time.val:3f}\t'
                      f'Loss {losses.val:.4f}\t'
                      f'Dist-Loss {dist_losses.val:.4f}\t'
                      f'Center-Loss {center_losses.val:.4f}\t'
                      )
            sys.stdout.flush()

    return losses.avg, loss_record

def validate_AGE(loader_dict, model, cfg):
    model.eval()
    data_time = AverageMeter()

    embs_train = extract_embs(model.encoder, loader_dict['train_for_val'], cfg)
    embs_train = embs_train.to(cfg.device)

    embs_test = extract_embs(model.encoder, loader_dict['val'], cfg)
    embs_test = embs_test.to(cfg.device)
    
    n_test = len(embs_test)
    n_batch = int(np.ceil(n_test / cfg.test_batch_size))
    
    test_labels = loader_dict['val'].dataset.labels
    train_labels = loader_dict['train_for_val'].dataset.labels

    if isinstance(cfg.k, int):
        preds_all = defaultdict(list)
        k = cfg.k
        with torch.no_grad():
            end = time.time()
            for idx in range(n_batch):
                data_time.update(time.time() - end)
                i_st = idx * cfg.test_batch_size
                i_end = min(i_st + cfg.test_batch_size, n_test)

                # ===================meters=====================
                vals, inds = find_kNN(embs_test[i_st:i_end].view(i_end - i_st, -1), embs_train, k=k,
                                      metric=cfg.metric)
                inds = np.squeeze(to_np(inds), 0)

                nn_labels = train_labels[inds[:, :k]]
                pred_mean = np.round(np.mean(nn_labels, axis=-1, dtype=np.float32))
                preds_all[k].append(pred_mean)
        preds_all[k] = np.concatenate(preds_all[k])
        pred_age = preds_all[k]

        p_age = pred_age
        t_labels = test_labels
        mae, cs, acc = evaluate_metric(p_age, t_labels)
        best_k = k
    else:
        preds_all = defaultdict(list)
        with torch.no_grad():
            end = time.time()
            for idx in range(n_batch):
                data_time.update(time.time() - end)
                i_st = idx * cfg.test_batch_size
                i_end = min(i_st + cfg.test_batch_size, n_test)
                # ===================meters=====================
                vals, inds = find_kNN(embs_test[i_st:i_end].view(i_end - i_st, -1), embs_train, k=max(cfg.k),
                                      metric=cfg.metric)
                inds = np.squeeze(to_np(inds), 0)
                for k in cfg.k:
                    nn_labels = train_labels[inds[:, :k]]
                    pred_mean = np.round(np.mean(nn_labels, axis=-1, dtype=np.float32))
                    preds_all[k].append(pred_mean)

        for key in preds_all.keys():
            preds_all[key] = np.concatenate(preds_all[key])

        best_mae, best_k = print_eval_result_by_groups_and_k(test_labels, train_labels, preds_all, cfg.logfile, interval=3)
        pred_age = preds_all[best_k]

        p_age = pred_age
        t_labels = test_labels
        mae, cs, acc = evaluate_metric(p_age, t_labels)

    write_log(cfg.logfile, f'MAE: {mae:.3f}, CS:{cs:.4f}, Acc : {acc * 100:.2f}')
    sys.stdout.flush()

    return mae, cs, best_k



if __name__ == "__main__":
    main()

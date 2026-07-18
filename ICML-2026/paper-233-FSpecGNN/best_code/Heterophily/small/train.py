import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns
import numpy as np
import time
from dataset_loader import DataLoader
from utils import random_splits,fixed_splits, set_seed
from copy import deepcopy

from models import *

# Deterministic
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def train(model, optimizer, data):
    model.train()
    # for n,p in model.named_parameters():
    #     print(f'n:{ n}--p:{p.data[0:5]}')

    optimizer.zero_grad(set_to_none=True)
    out = model(data)[data.train_mask]
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    del out


def test(model, data):
    model.eval()
    accs, losses, preds = [], [], []
    with torch.no_grad():
        logits = model(data)
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            logit_m = logits[mask]
            pred = logit_m.argmax(dim=1)
            acc = (pred.eq(data.y[mask]).sum().item()) / mask.sum().item()
            loss = F.nll_loss(logit_m, data.y[mask])
            preds.append(pred.cpu())
            accs.append(acc)
            losses.append(loss.item())
    return accs, preds, losses


def RunExp(args, dataset, data, Net, percls_trn, val_lb):
    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')

    tmp_net = Net(dataset, args)
    #Using the dataset splits described in the paper.
    if args.full:
        data = random_splits(data, dataset.num_classes, percls_trn, val_lb,args.seed)
    elif args.semi_rnd:
        data = random_splits(data, dataset.num_classes, percls_trn, val_lb,args.seed)
    elif args.semi_fix and args.dataset in ["Chameleon","Squirrel","Texas","Wisconsin"]:
        data = fixed_splits(data, dataset.num_classes, percls_trn,val_lb, args.dataset)
    
    model, data = tmp_net.to(device), data.to(device)

    if args.net in ['Cheb_GAT']:
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.wd)
    elif args.net in ['Bern_GAT','ChebII_GAT']:
        lin_params = [
            p for name, p in model.named_parameters()
            if name.startswith("lin1") or name.startswith("lin2")
        ]
        other_params = [
            p for name, p in model.named_parameters()
            if not (name.startswith("lin1") or name.startswith("lin2"))
        ]
        optimizer = torch.optim.Adam([
            {"params": other_params, "lr": args.prop_lr, "weight_decay": args.prop_wd},
            {"params": lin_params,   "lr": args.lr,      "weight_decay": args.wd}
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.wd)



    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=1, eta_min=1e-6)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    best_train_acc = 0
    best_test_acc = 0
    best_train_loss = float('inf')
    best_test_loss = float('inf')
    best_step = -1

    val_loss_history = []
    val_acc_history = []
    train_loss_history = []

    time_run=[]
    for epoch in range(args.epochs):
        t_st=time.time()
        train(model, optimizer, data)
        scheduler.step()

        time_epoch=time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)
        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(model, data)
        train_loss_history.append(train_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc

            best_train_acc = train_acc
            best_test_acc = tmp_test_acc
            best_train_loss = train_loss
            best_test_loss = tmp_test_loss
            best_step = epoch


        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

    print(f'best_step:{best_step} train_acc:{best_train_acc} test_acc:{best_test_acc} val_acc:{best_val_acc} '
          f'train_loss:{best_train_loss} test_loss:{best_test_loss} val_loss:{best_val_loss}')
    return best_step, best_train_acc, best_train_loss, best_test_acc, best_test_loss, best_val_acc, best_val_loss, time_run


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42, help='seed.')
    ap.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    ap.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    ap.add_argument('--wd', type=float, default=0.0005, help='weight decay.')  
    ap.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    ap.add_argument('--hidden', type=int, default=64, help='hidden units.')
    ap.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')

    ap.add_argument('--train_rate', type=float, default=0.025, help='train set rate.')
    ap.add_argument('--val_rate', type=float, default=0.025, help='val set rate.')
    ap.add_argument('--K', type=int, default=10, help='propagation steps.')
    ap.add_argument("--heads", type=int, default=1, help='numbers of heads for GAT')
    ap.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
    ap.add_argument('--alpha_init', type=float, default=-6.0, help='initial value for alpha logit.')

    ap.add_argument('--dataset', type=str, default='Texas', help='Texas, Wisconsin, Chameleon, Squirrel, ...')
    ap.add_argument('--device', type=int, default=0, help='GPU device.')
    ap.add_argument('--runs', type=int, default=10, help='number of runs.')
    ap.add_argument('--net', type=str, default='Cheb_GAT', help='Cheb_GAT, Bern_GAT, ChebII_GAT')
    ap.add_argument('--prop_lr', type=float, default=0.01, help='learning rate for propagation layer.')
    ap.add_argument('--prop_wd', type=float, default=0.0005, help='learning rate for propagation layer.')

    ap.add_argument('--full', action='store_true', help='full-supervise with random splits')
    ap.add_argument('--semi_rnd', action='store_true', help='semi-supervised with random splits')
    ap.add_argument('--semi_fix', action='store_true', help='semi-supervised with fixed splits')

    args = ap.parse_args()
    set_seed(args.seed)
    #10 fixed seeds for random splits from BernNet
    SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]

    print(args)
    print("---------------------------------------------")

    MODEL_MAP = {
        "Cheb_GAT": Cheb_GAT,
        "Bern_GAT": Bern_GAT,
        "ChebII_GAT": ChebII_GAT,
    }

    gnn_name = args.net
    if gnn_name not in MODEL_MAP:
        raise ValueError(f"Unknown model name: {gnn_name}")
    Net = MODEL_MAP[gnn_name]
    
    dataset = DataLoader(args.dataset)
    data = dataset[0]

    #print stats of dataset
    print(f'nodes:{data.x.size(0)} edges:{data.edge_index.size(1)} num_class:{dataset.num_classes} feat dim:{dataset.num_features}')

    if args.full:
        args.train_rate = 0.6
        args.val_rate = 0.2
    else:
        args.train_rate = 0.025
        args.val_rate = 0.025
    percls_trn = int(round(args.train_rate*len(data.y)/dataset.num_classes))
    val_lb = int(round(args.val_rate*len(data.y)))
    
    # results = []
    time_results=[]
    best_step_list = []
    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    for RP in tqdm(range(args.runs)):
        print(f'run:{RP}')
        args.seed=SEEDS[RP]
        best_step, best_train_acc, best_train_loss, best_test_acc, best_test_loss, best_val_acc, best_val_loss, time_run = RunExp(args, dataset, deepcopy(data), Net, percls_trn, val_lb)

        time_results.append(time_run)
        best_step_list.append(best_step)
        train_acc_list.append(best_train_acc)
        train_loss_list.append(best_train_loss)
        test_acc_list.append(best_test_acc)
        test_loss_list.append(best_test_loss)

        # # results.append([test_acc, best_val_acc, theta_0])
        # results.append([best_test_acc, best_val_acc])
        # print(f'run_{str(RP+1)} \t test_acc: {test_acc:.4f}')

    print(f'adNet name: {gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    for i in range(args.runs):
        print('{} : {} ---- {}'.format(i, test_acc_list[i], best_step_list[i]))
    print('{} on {}'.format(args.net, args.dataset))

    test_acc_mean = np.mean(np.array(test_acc_list))
    test_acc_std = np.max(np.abs(
        sns.utils.ci(sns.algorithms.bootstrap(np.array(test_acc_list), func=np.mean, n_boot=1000), 95) - np.array(
            test_acc_list).mean()))
    print(f'test acc mean = {test_acc_mean * 100:.4f} ± {test_acc_std * 100:.4f}')

    train_acc_mean = np.mean(np.array(train_acc_list))
    train_acc_std = uncertainty = np.max(np.abs(
        sns.utils.ci(sns.algorithms.bootstrap(np.array(train_acc_list), func=np.mean, n_boot=1000), 95) - np.array(
            train_acc_list).mean()))
    print(f'train acc mean = {train_acc_mean * 100:.4f} ± {train_acc_std * 100:.4f}')

    print('train acc list:{}\ntest acc list:{}'.format(train_acc_list, test_acc_list))

    gap_acc_tensor = (np.array(train_acc_list) - np.array(test_acc_list)) * 100.0
    gap_loss_tensor = np.array(test_loss_list) - np.array(train_loss_list)
    print('gap_acc:{}'.format(gap_acc_tensor.tolist()))
    print('gap loss:{}'.format(gap_loss_tensor.tolist()))
    mean_gap_acc, std_gap_acc = np.mean(gap_acc_tensor), np.max(np.abs(
        sns.utils.ci(sns.algorithms.bootstrap(gap_acc_tensor, func=np.mean, n_boot=1000), 95) - gap_acc_tensor.mean()))
    mean_gap_loss, std_gap_loss = np.mean(gap_loss_tensor), np.max(np.abs(
        sns.utils.ci(sns.algorithms.bootstrap(gap_loss_tensor, func=np.mean, n_boot=1000),
                     95) - gap_loss_tensor.mean()))

    print(f'gap acc mean = {mean_gap_acc:.4f} ± {std_gap_acc:.4f}')
    print(f'gap loss mean = {mean_gap_loss:.4f} ± {std_gap_loss:.4f}')

    run_sum=0
    epochsss=0
    for i in time_results:
        run_sum+=sum(i)
        epochsss+=len(i)
    print("each run avg_time:",run_sum/(args.runs),"s")
    print("each epoch avg_time:",1000*run_sum/epochsss,"ms")

    print(f'finish')

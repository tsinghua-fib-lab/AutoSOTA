import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from logger import *
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits, class_rand_splits
from eval import *
from parse import parse_method, parser_add_main_args

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

### Parse args ###
parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [class_rand_splits(
        dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)


dataset.label = dataset.label.to(device)

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]
original_d = d

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

### FAF feature augmentation ###
model_name = str(args.model).lower()
if model_name in ('faf', 'fafmlp', 'faf-mlp'):
    extra_args = {
        'multi_agg': args.multi_agg,
        'sum_agg': args.sum_agg,
        'mean_agg': args.mean_agg,
        'max_agg': args.max_agg,
        'std_agg': args.std_agg,
        'ka_agg': args.ka_agg,
        'last_agg': args.last_agg,
        'last_agg_only': args.last_agg_only,
        'all_agg': args.all_agg,
        'exp_agg': args.exp_agg,
        'meansumall_agg': args.meansumall_agg,
        'ka_args': {
            'ka_order': args.ka_order,
            'ka_D_max': args.ka_D_max,
            'ka_truncate': args.ka_truncate,
            'ka_pad_value': args.ka_pad_value,
            'ka_transform': args.ka_transform,
            'ka_temperature': args.ka_temperature,
            'ka_n_bits': args.ka_n_bits,
        },
        'bin_agg': args.bin_agg,
        'bin_args': {
            'bin_num': args.bin_num,
            'bin_edges': args.bin_edges,
            'bin_cdf': args.bin_cdf,
        },
        'sim_agg': args.sim_agg,
        'sim_args': {
            'sim_mode': args.sim_mode,
            'sim_slice': args.sim_slice,
            'sim_clamp_negatives': args.sim_clamp_negatives,
            'sim_clamp_positives': args.sim_clamp_positives,
            'sim_normalize': args.sim_normalize,
            'sim_temperature': args.sim_temperature,
            'sim_eps': args.sim_eps,
            'sim_type': args.sim_type,
        },
        'rewire': args.rewire,
        'split_comp': args.split_comp,
        'q_agg': args.q_agg,
        'q_args': {
            'q_include': args.q_include.split(','),
            'q_interpolation': args.q_interpolation,
        },
        'ns_agg': args.ns_agg,
        'ns_args': {
            'ns_include': args.ns_include.split(','),
            'ns_cc_k': args.ns_cc_k,
            'ns_ev_max_iter': args.ns_ev_max_iter,
            'ns_ev_tol': args.ns_ev_tol,
            'ns_betweenness_cpu': args.ns_betweenness_cpu,
            'ns_bc_k': args.ns_bc_k,
        }
    }
    with torch.no_grad():
        from aggregation import aggregate_faf_features
        H = aggregate_faf_features(
            dataset.graph['node_feat'],
            dataset.graph['edge_index'],
            args.local_layers,
            extra_args
        )
    dataset.graph['node_feat'] = H.to(device)
    d = H.size(1)
    print(f"FAF features | num node feats {d} | added {d - original_d} features")
    dataset.graph['edge_index'] = None

### Load method ###
model = parse_method(args, n, c, d, device)

if args.shap:
    import shap
    import os
    import time
    directory_shap = args.info_dir + '/' + args.dataset + '/' + args.gnn + '/'
    if not os.path.exists(directory_shap):
        try:
            os.makedirs(directory_shap, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create directory {directory_shap}: {e}")
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    model,optimizer = load_model(args, model, optimizer, 0, 'best')
    model.eval()
    background = dataset.graph['node_feat'][torch.randperm(n)[:args.shap_background]].to(device)
    # print current time: 
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    explainer = shap.GradientExplainer(model, background)
    print("Background set for SHAP explainer created")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    shap_values = explainer.shap_values(dataset.graph['node_feat'])
    print("SHAP values calculated")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    np.save(directory_shap + f'{args.project_name}_0_shap_values.npy', shap_values)
    exit(0)

### Loss function (Single-class, Multi-class) ###
if args.dataset in ('questions'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

args.method = args.gnn
logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

import wandb

### Training loop ###
for run in range(args.runs):
    wandb.init(
        project=args.project,
        config={
            **vars(args),
            'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'input_dim': d,
            'original_dim': original_d,
            'n_classes': c,
            'n_nodes': n,
            'n_edges': e,
        },
        name=args.project_name + '_' + str(run),
    )
    if args.dataset in ('coauthor-cs', 'coauthor-physics', 'amazon-computer', 'amazon-photo', 'cora', 'citeseer', 'pubmed'):
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    if args.use_scheduler:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.lr * 1e-4)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])
        print(f'Scheduler: warmup={args.warmup_epochs} epochs + cosine annealing to {args.lr * 1e-4:.2e}')
    best_val = float('-inf')
    best_test = float('-inf')
    if args.swa:
        swa_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
        swa_n = 0
    if args.save_model:
        save_model(args, model, optimizer, run, 'init')

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
        if args.dataset in ('questions'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            loss = criterion(out[train_idx], true_label.squeeze(1)[
                train_idx].to(torch.float))
        else:
            out = F.log_softmax(out, dim=1)
            loss = criterion(
                out[train_idx], dataset.label.squeeze(1)[train_idx])
        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if args.swa and epoch >= args.swa_start_epoch:
            swa_n += 1
            for k in swa_state:
                swa_state[k] = swa_state[k] * (swa_n - 1) / swa_n + model.state_dict()[k].detach() / swa_n
        if args.use_scheduler:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)

        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            best_test = result[2]
            if args.save_model:
                save_model(args, model, optimizer, run, 'best')

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%, '
                  f'Best Valid: {100 * best_val:.2f}%, '
                  f'Best Test: {100 * best_test:.2f}%')
            
        wandb.log({
            'loss': loss,
            'train_acc': result[0],
            'valid_acc': result[1],
            'test_acc': result[2],
            'best_valid_acc': best_val,
            'best_test_acc': best_test
        })

    wandb.finish()
    if args.save_model:
        save_model(args, model, optimizer, run, 'final')

    # SWA evaluation
    if args.swa:
        orig_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(swa_state)
        model.eval()
        with torch.no_grad():
            swa_out = F.log_softmax(model(dataset.graph['node_feat'], None), dim=1)
        swa_test = eval_func(dataset.label[split_idx['test']], swa_out[split_idx['test']])
        swa_train = eval_func(dataset.label[split_idx['train']], swa_out[split_idx['train']])
        print(f'SWA (epochs {args.swa_start_epoch}-{args.epochs}) | Train: {100 * swa_train:.2f}% | Test: {100 * swa_test:.2f}%')
        best_test = swa_test
        model.load_state_dict(orig_state)

    logger.print_statistics(run)

results = logger.print_statistics()
# ### Save results ###
# save_result(args, results)


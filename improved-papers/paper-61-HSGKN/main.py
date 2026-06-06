from utils import *
import dgl
from model import *
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from opt import args
import warnings
from sklearn.model_selection import StratifiedKFold
import random
import yaml

warnings.filterwarnings("ignore")

# seed = 0
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

def train(lr, weight_decay, hid_paths, norm, cutoff, dropout, norm_attr):
    if args.dataset in ('PROTEINS', 'PROTEINS_full', 'ENZYMES', 'SYNTHETIC', 'BZR', 'COX2', 'Synthie', 'DHFR'):
        dataset = dgl.data.TUDataset(args.dataset)
        dataset = cat_attr_label(dataset)
        node_feat_dim = dataset[0][0].ndata['node_attr'].shape[1]
    elif args.dataset in ('NCI1', 'DD', 'MUTAG', 'PTC_MR', 'NCI109'):
        dataset = dgl.data.TUDataset(args.dataset)
        set_attr_as_label(dataset)
        node_feat_dim = dataset[0][0].ndata['node_attr'].shape[1]
    elif args.dataset in (
            'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'FRANKENSTEIN', 'REDDIT-BINARY', 'REDDIT-MULTI-5K',
            'REDDIT-MULTI-12K', 'reddit_threads'):
        dataset = dgl.data.TUDataset(args.dataset)
        if args.dataset in ('IMDB-BINARY', 'IMDB-MULTI', 'reddit_threads', 'COLLAB'):
            set_label_as_degree(dataset)
            set_attr_as_label(dataset)
        elif args.dataset in ('REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K', 'reddit_threads'):
            set_node_degree_as_feature(dataset)
        node_feat_dim = dataset[0][0].ndata['node_attr'].shape[1]

    if norm == 'layer':
        norm = torch.nn.LayerNorm
    elif norm == 'batch_norm':
        norm = torch.nn.BatchNorm1d

    attrs_filename = 'cache/{}.pt'.format(args.dataset)
    len_count_filename = 'cache/{}_len_count.pt'.format(args.dataset)
    if os.path.exists(attrs_filename):
        len_count = torch.load(len_count_filename)
        graphs_paths_attrs = torch.load(attrs_filename)
    else:
        graphs_paths_attrs, len_count = process_dataset_matrix_ig(dataset, cutoff)
        torch.save(graphs_paths_attrs, attrs_filename)
        torch.save(len_count, len_count_filename)

    for l in list(len_count.keys()):
        if cutoff is not None:
            if l > cutoff:
                del len_count[l]

    p = int(sum(len_count.values()) / hid_paths)
    hiden_path_channel = {k: max((v + p - 1) // p, 5) for k, v in len_count.items()}
    a = sum(hiden_path_channel.values())
    hid_dims = [int(a / 2), int(a / 4)]

    x = graphs_paths_attrs.to(args.device)
    if cutoff is not None:
        x = x[:, :int(cutoff * (cutoff + 1) / 2) * node_feat_dim]
    if norm_attr:
        x = normalize_attr(x)

    y = dataset.graph_labels.to(args.device)
    num_classes = dataset.num_labels

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    val_acc_list = []
    for fold, (train_index, val_index) in enumerate(skf.split(x.cpu().clone(), y.cpu().clone())):

        model = Model(hiden_path_channel, node_feat_dim, num_classes, hid_dims, norm, dropout).to(args.device)
        x_train = x[train_index]
        x_test = x[val_index]
        y_train = y[train_index].view(-1)
        y_test = y[val_index].view(-1)

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_acc = 0
        start = time.time()
        for epoch in range(args.epoch):
            model.train()
            optimizer.zero_grad()
            outputs = model(x_train)

            loss = loss_function(outputs, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            outputs = model(x_test)
            _, pred_test = outputs.max(1)

            val_acc = (pred_test.eq(y_test)).sum().float() / len(y_test)
            if val_acc > best_acc:
                best_acc = val_acc

        end = time.time() - start
        # print('AVG time:',(end)/args.epoch)
        val_acc_list.append(best_acc.item())
        print('Fold{}, Best Acc: {:.4f}'.format(fold, best_acc))
    print('Avg Acc: {:.4f} Std: {:.4f}'.format(np.mean(val_acc_list), np.std(val_acc_list)))
    return np.mean(val_acc_list)


if __name__ == '__main__':
    with open('config.yml', 'r') as file:
        data = yaml.safe_load(file)
    config = data['dataset'][args.dataset]

    train(config['lr'], config['l2'], config['hidden_paths'], config['norm'], config['cutoff'],
          config['dropout'], config['norm_attr'])

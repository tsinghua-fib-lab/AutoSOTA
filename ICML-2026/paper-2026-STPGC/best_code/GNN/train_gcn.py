import argparse
import os
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch import tensor
from network import Net, GCN, net_gcn
from torch_geometric.datasets import Planetoid, CitationFull
from PyGdataset import PygNodePropPredDataset, Evaluator
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import SparseTensor

torch.manual_seed(42)

PLANETOID_DATASETS = {'Cora', 'Citeseer', 'pubmed'}
CITATION_DATASETS = {'dblp'}

EMBEDDING_DIM_DICT = {
    'ogbn-arxiv': [128, 512, 40],
}

COARSENED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'coarsened_graph')

LR_DICT = {'ogbn-arxiv': 0.01, 'ogbn-products': 0.01}
WD_DICT = {'ogbn-arxiv': 5e-4, 'ogbn-products': 5e-4}

NUM_FEATURES_CLASSES = {
    'Cora': (1433, 7), 'Citeseer': (3703, 6), 'pubmed': (500, 3), 'dblp': (1639, 4),
}

EPOCHS_DEFAULT = {'ogbn-products': 300}


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def build_coarsened_labels_and_masks(data_mol, m):
    num_nodes = data_mol.new_x.shape[0]
    train_label = torch.zeros(num_nodes, dtype=torch.int64)
    train_mask = torch.zeros(num_nodes).bool()
    val_label = torch.zeros(num_nodes, dtype=torch.int64)
    val_mask = torch.zeros(num_nodes).bool()
    train_m = {}
    val_m = {}
    for key in m.keys():
        if data_mol.train_mask[key] == 1:
            train_mask[m[key]] = True
            if m[key] not in train_m:
                train_m[m[key]] = []
            train_m[m[key]].append(data_mol.y[key])
        if data_mol.val_mask[key] == 1:
            val_mask[m[key]] = True
            if m[key] not in val_m:
                val_m[m[key]] = []
            val_m[m[key]].append(data_mol.y[key])
    for key in train_m.keys():
        train_label[key] = data_mol.node_label[key]
    for key in val_m.keys():
        val_label[key] = data_mol.node_label[key]
    return train_label, train_mask, val_label, val_mask


def build_sparse_norm_adj(edge_index, num_nodes):
    adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                       sparse_sizes=(num_nodes, num_nodes))
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    return adj


def test_with_evaluator(model, data, split_idx, evaluator):
    model.eval()
    out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)
    if len(data.y.shape) == 1:
        data.y = data.y.unsqueeze(1)
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    return test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--coarsening_ratio', type=float, default=0.3)
    args = parser.parse_args()

    path = "params/"
    if not os.path.isdir(path):
        os.mkdir(path)

    dataname = args.dataset
    is_planetoid = dataname in PLANETOID_DATASETS
    is_citation = dataname in CITATION_DATASETS
    is_arxiv = dataname == 'ogbn-arxiv'
    is_products = dataname == 'ogbn-products'

    # All non-OGB datasets use the same Net + nll_loss + val_loss logic
    is_train_py_style = is_planetoid or is_citation

    if not (is_train_py_style or is_arxiv or is_products):
        raise ValueError(f"Unsupported dataset: {dataname}")

    v = float(args.coarsening_ratio)

    # Load coarsened data from graph_coarsening output
    coarsened_file = os.path.join(COARSENED_DIR, f'{dataname}_{v:.2f}.npy')
    data_mol, m = np.load(coarsened_file, allow_pickle=True)

    # Load original dataset
    evaluator = None
    split_idx = None
    if is_planetoid:
        dataset = Planetoid(root='./dataset', name=dataname)
        data = dataset[0]
    elif is_citation:
        dataset = CitationFull(root='./dataset', name=dataname)
        data = dataset[0]
        indices = []
        num_classes = torch.unique(data.y, return_counts=True)[0].shape[0]
        for ci in range(num_classes):
            index = (data.y == ci).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
        train_index = torch.cat([idx[:int(len(idx)*0.7)] for idx in indices], dim=0)
        val_index = torch.cat([idx[int(len(idx)*0.7):int(len(idx)*0.8)] for idx in indices], dim=0)
        test_index = torch.cat([idx[int(len(idx)*0.8):] for idx in indices], dim=0)
        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)
        data.train_mask = data_mol['train_mask']
        data.val_mask = data_mol['val_mask']
        data.test_mask = data_mol['test_mask']
    elif is_arxiv:
        dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                         root='./dataset/arxiv')
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator('ogbn-arxiv')
        data = dataset[0]
        data.edge_index = to_undirected(data.edge_index)
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
        data.train_mask = index_to_mask(split_idx["train"], size=data.num_nodes)
        data.val_mask = index_to_mask(split_idx["valid"], size=data.num_nodes)
        data.test_mask = index_to_mask(split_idx["test"], size=data.num_nodes)
        data.y = data.y.view(-1)
    elif is_products:
        dataset = PygNodePropPredDataset(name='ogbn-products', root='/mnt/ssd2/products/raw')
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator('ogbn-products')
        data = dataset[0]
        data.edge_index = to_undirected(data.edge_index)
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
        data.train_mask = index_to_mask(split_idx["train"], size=data.num_nodes)
        data.val_mask = index_to_mask(split_idx["valid"], size=data.num_nodes)
        data.test_mask = index_to_mask(split_idx["test"], size=data.num_nodes)
        data.y = data.y.view(-1)

    # Copy original graph attributes into data_mol
    data_mol.x = data.x
    data_mol.y = data.y
    data_mol.edge_index = data.edge_index
    data_mol.train_mask = data.train_mask
    data_mol.val_mask = data.val_mask
    data_mol.test_mask = data.test_mask
    data_mol.new_edge = data_mol.new_edge_index

    # Set num_features and num_classes
    if dataname in NUM_FEATURES_CLASSES:
        args.num_features, args.num_classes = NUM_FEATURES_CLASSES[dataname]
    else:
        args.num_features = data_mol.x.size()[1]
        if is_arxiv:
            args.num_classes = 40
        elif is_products:
            args.num_classes = 47

    # Override epochs for products
    if is_products:
        args.epochs = EPOCHS_DEFAULT.get(dataname, args.epochs)

    # Build model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if is_train_py_style:
        model = Net(args).to(device)
    elif is_arxiv:
        model = net_gcn(embedding_dim=EMBEDDING_DIM_DICT[dataname]).to(device)
    elif is_products:
        model = GCN(in_channels=args.num_features, hidden_channels=256,
                     out_channels=args.num_classes, num_layers=2, dropout=0.5).to(device)

    args.coarsening_method = 'Ours'

    # Prepare products original graph SparseTensor (only needs to be done once)
    if is_products:
        ori_adj_mat = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                                   sparse_sizes=(data.x.size(0), data.x.size(0)))
        ori_adj_mat = ori_adj_mat.set_diag()
        deg = ori_adj_mat.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * ori_adj_mat * deg_inv_sqrt.view(1, -1)
        data.adj_t = ori_adj_mat

    # Training
    f = open('./parameter_experiment.txt', 'a')
    all_acc = []
    all_val_loss = []

    for i in range(args.runs):
        train_label, train_mask, val_label, val_mask = build_coarsened_labels_and_masks(data_mol, m)

        if is_train_py_style:
            data = data_mol.to(device)
        elif is_arxiv:
            data = data_mol.to(device)
        elif is_products:
            data = data_mol

        coarsen_features = data_mol.new_x.to(device)
        coarsen_train_labels = train_label.to(device)
        coarsen_train_mask = train_mask.to(device)
        coarsen_val_labels = val_label.to(device)
        coarsen_val_mask = val_mask.to(device)

        print(torch.unique(coarsen_train_labels[coarsen_train_mask], return_counts=True))
        print(torch.unique(coarsen_val_labels[coarsen_val_mask], return_counts=True))

        # Prepare coarsened graph input
        if is_products:
            coarsen_edge = data_mol.new_edge
            coarsen_edge = add_self_loops(coarsen_edge, num_nodes=coarsen_features.size(0))[0]
            coarsen_edge = to_undirected(coarsen_edge)
            coarsen_adj_mat = build_sparse_norm_adj(coarsen_edge, coarsen_features.size(0))
            coarsen_adj_mat = coarsen_adj_mat.to(device)
            coarsen_graph_input = coarsen_adj_mat
        else:
            coarsen_edge = data_mol.new_edge.to(device)
            coarsen_edge = add_self_loops(coarsen_edge, num_nodes=coarsen_features.size(0))[0]
            coarsen_graph_input = coarsen_edge

        # Optimizer and loss
        if is_train_py_style:
            lr, wd = args.lr, args.weight_decay
        else:
            lr, wd = LR_DICT[dataname], WD_DICT[dataname]

        if is_train_py_style:
            model.reset_parameters()
        elif is_products:
            model.reset_parameters()
        # arxiv (net_gcn) has no explicit reset_parameters call in original

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        if is_arxiv:
            loss_func = torch.nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        best_val_acc = 0.0
        val_history = []

        for epoch in range(args.epochs):
            # Train
            model.train()
            optimizer.zero_grad()

            if is_arxiv:
                output = model(coarsen_features, coarsen_graph_input, val_test=False)
                loss = loss_func(output[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
            elif is_products:
                output = model(coarsen_features, coarsen_graph_input)
                loss = F.nll_loss(output[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
            else:
                output = model(coarsen_features, coarsen_graph_input)
                loss = F.nll_loss(output[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])

            loss.backward()
            optimizer.step()

            # Validate
            if is_products:
                with torch.no_grad():
                    model.eval()
                    pred = model(coarsen_features, coarsen_graph_input)
                    y_pred = pred.argmax(dim=-1, keepdim=True)
                    valid_acc = evaluator.eval({
                        'y_true': coarsen_val_labels[coarsen_val_mask].unsqueeze(1),
                        'y_pred': y_pred[coarsen_val_mask],
                    })['acc']
                val_metric = valid_acc
                if best_val_acc < valid_acc:
                    best_val_acc = valid_acc
                    torch.save(model.state_dict(), path + 'checkpoint-best-acc3.pkl')
            elif is_arxiv:
                model.eval()
                pred = model(coarsen_features, coarsen_graph_input, val_test=True)
                y_pred = torch.log_softmax(pred, dim=-1)
                y_pred = y_pred.argmax(dim=-1, keepdim=True)
                valid_acc = evaluator.eval({
                    'y_true': coarsen_val_labels[coarsen_val_mask].unsqueeze(1),
                    'y_pred': y_pred[coarsen_val_mask],
                })['acc']
                val_metric = valid_acc
                if best_val_acc < valid_acc:
                    best_val_acc = valid_acc
                    torch.save(model.state_dict(), path + 'checkpoint-best-acc.pkl')
            else:
                model.eval()
                pred = model(coarsen_features, coarsen_graph_input)
                val_loss = F.nll_loss(pred[coarsen_val_mask], coarsen_val_labels[coarsen_val_mask]).item()
                val_metric = val_loss
                if val_loss < best_val_loss and epoch > args.epochs // 2:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), path + 'checkpoint-best-acc.pkl')

            # Early stopping
            val_history.append(val_metric)
            if args.early_stopping > 0 and epoch > args.epochs // 2:
                tmp = tensor(val_history[-(args.early_stopping + 1):-1])
                if is_train_py_style:
                    if val_metric > tmp.mean().item():
                        break
                else:
                    if val_metric < tmp.mean().item():
                        break

        all_val_loss.append(best_val_loss)

        # Test
        if is_products:
            model.load_state_dict(torch.load(path + 'checkpoint-best-acc3.pkl'))
            model.cpu()
            coarsen_features = coarsen_features.to("cpu")
            device1 = torch.device("cpu")
            data = data.to(device1)
            model.eval()
            with torch.no_grad():
                test_acc = test_with_evaluator(model, data, split_idx, evaluator)
                print(test_acc)
                all_acc.append(test_acc)
        elif is_arxiv:
            gc.collect()
            torch.cuda.empty_cache()
            model.load_state_dict(torch.load(path + 'checkpoint-best-acc.pkl'))
            model.eval()
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.y = data.y.to(device)
            data.test_mask = data.test_mask.to(device)
            pred = model(data.x, data.edge_index, val_test=True).max(1)[1]
            test_acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())
            print(test_acc)
            all_acc.append(test_acc)
        else:
            model.load_state_dict(torch.load(path + 'checkpoint-best-acc.pkl'))
            model.eval()
            pred = model(data.x, data.edge_index).max(1)[1]
            test_acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())
            print(test_acc)
            all_acc.append(test_acc)

    if len(all_acc) == 0:
        f.write('%s  ' % args.coarsening_method)
        f.write('unable to Coarse.\n')
    else:
        print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))
        print('val_loss: {:.4f}'.format(np.mean(all_val_loss)), '+/- {:.4f}'.format(np.std(all_val_loss)))
        f.write(f"dataset {dataname} , ratio {v}\n ")
        f.write('ave_acc: {:.4f}'.format(np.mean(all_acc)) + ' +/- {:.4f}'.format(np.std(all_acc)) + '\n')

    f.write('\n')
    f.close()

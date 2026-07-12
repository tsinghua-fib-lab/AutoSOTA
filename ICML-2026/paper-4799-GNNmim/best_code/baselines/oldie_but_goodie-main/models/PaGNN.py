from embedder import embedder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import datetime
import utils
from tqdm import trange

from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_geometric.nn import GCNConv
from utils import get_symmetrically_normalized_adjacency

class PaGNN():
    def __init__(self, args):
        self.args = args
    
    def training(self):
        file = utils.set_filename(self.args)
        logger = utils.setup_logger('./', '-', file)

        seed_result = {}
        seed_result['acc'] = []
        seed_result['macro_F'] = []
        
        for seed in trange(0, 0+self.args.n_runs):
            print(f'============== seed:{seed} ==============')
            utils.seed_everything(seed)
            print('seed:', seed, file)
            self.args.seed = seed
            self = embedder(self.args, seed)

            # Main training
            model = modeler(self.args, self.missing_feature_mask, self.edge_index).to(self.args.device)

            optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

            acc_vals = []
            test_results = []
            best_metric = 0

            for epoch in range(0, self.args.epochs):
                model.train()
                optimizer.zero_grad()

                loss = model(self.x, self.edge_index, self.labels, self.train_mask)

                loss.backward()
                optimizer.step()

                # Valid
                model.eval()
                for conv in model.convs[:-1]:
                    x = conv(self.x, self.edge_index).relu_()

                output = model.convs[-1](x, self.edge_index)
                acc_val, macro_F_val = utils.performance(output[self.val_mask], self.labels[self.val_mask], pre='valid', evaluator=self.evaluator)

                acc_vals.append(acc_val)

                if best_metric <= acc_val:
                    best_metric = acc_val
                    max_idx = acc_vals.index(max(acc_vals))
                    best_output = output[:]

                # Test
                acc_test, macro_F_test = utils.performance(output[self.test_mask], self.labels[self.test_mask], pre='test', evaluator=self.evaluator)

                test_results.append([acc_test, macro_F_test])
                best_test_result = test_results[max_idx]

                if epoch % 100 == 0:
                    st = "[seed {}][{}-{}][{}][Epoch {}]".format(seed, self.args.dataset, self.args.missing_rate, self.args.embedder, epoch)
                    st += "[Val] ACC: {:.2f}, Macro-F1: {:.2f}|| ".format(acc_val, macro_F_val)
                    st += "[Test] ACC: {:.2f}, Macro-F1: {:.2f}\n".format(acc_test, macro_F_test)
                    st += "  [*Best Test Result*][Epoch {}] ACC: {:.2f}, Macro-F1: {:.2f}".format(max_idx, best_test_result[0], best_test_result[1])
                    print(st)
                      
                if (epoch - max_idx > self.args.patience) or (epoch+1 == self.args.epochs):
                    if epoch - max_idx > self.args.patience:
                        print("Early stop")
                    output = best_output
                    best_test_result[0], best_test_result[1] = utils.performance(output[self.test_mask], self.labels[self.test_mask], pre='test', evaluator=self.evaluator)
                    print("[Best Test Result] ACC: {:.2f}, Macro-F1: {:.2f}".format(best_test_result[0], best_test_result[1]))
                    break

            seed_result['acc'].append(float(best_test_result[0]))
            seed_result['macro_F'].append(float(best_test_result[1]))

        acc = seed_result['acc']
        f1 = seed_result['macro_F']

        print('[Averaged result] ACC: {:.2f}+{:.2f}, Macro-F: {:.2f}+{:.2f}'.format(np.mean(acc), np.std(acc), np.mean(f1), np.std(f1)))
        print('{:.2f}+{:.2f} {:.2f}+{:.2f}'.format(np.mean(acc), np.std(acc), np.mean(f1), np.std(f1)))
        
        logger.info('')
        logger.info(datetime.datetime.now())
        logger.info(file)
        logger.info(f'----------- missing rate: {self.args.missing_rate} -----------')
        logger.info('{:.2f}+{:.2f} {:.2f}+{:.2f}'.format(np.mean(acc), np.std(acc), np.mean(f1), np.std(f1)))
        logger.info('{:.2f}+{:.2f}'.format(np.mean(acc), np.std(acc)))
        logger.info('{:.2f}+{:.2f}'.format(np.mean(f1), np.std(f1)))
        logger.info(self.args)
        logger.info(f'=================================')
        # print(self.args)

class modeler(torch.nn.Module):
    def __init__(
        self, args, mask=None, edge_index=None,
    ):
        super(modeler, self).__init__()
        # NOTE: It not specified in their paper (https://arxiv.org/pdf/2003.10130.pdf), but the only way for their model to work is to have only the first layer
        # to be what they describe and the others to be standard GCN layers. Otherwise, the feature matrix would change dimensionality, and it couldn't be
        # multiplied elmentwise with the mask anymore
        self.convs = ModuleList([PaGNNConv(args.n_feat, args.n_hid, mask, edge_index)])
        for i in range(args.n_layer - 2):
            self.convs.append(GCNConv(args.n_hid, args.n_hid))
        self.convs.append(GCNConv(args.n_hid, args.n_class))
        self.num_layers = args.n_layer
        self.dropout = args.dropout
        self.args = args
        
    def forward(self, x, edge_index, labels, idx_train):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)

        output = self.convs[-1](x, edge_index)
        if 'OGBN' in self.args.dataset:
            labels = labels.squeeze(1)
        loss_nodeclassification = F.cross_entropy(output[idx_train], labels[idx_train])

        return loss_nodeclassification


class PaGNNConv(torch.nn.Module):
    def __init__(self, in_features, out_features, mask, edge_index):
        super(PaGNNConv, self).__init__()
        self.lin = torch.nn.Linear(in_features, out_features)
        self.mask = mask.float()
        edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, mask.shape[0])
        self.adj = torch.sparse.FloatTensor(edge_index, values=edge_weight).to(edge_index.device)

    def forward(self, x, edge_index):
        x[x.isnan()] = 0
        numerator = torch.sparse.mm(self.adj, torch.ones_like(x)) * torch.sparse.mm(self.adj, self.mask * x)
        denominator = torch.sparse.mm(self.adj, self.mask)
        ratio = torch.nan_to_num(numerator / denominator)
        x = self.lin(ratio)

        return x

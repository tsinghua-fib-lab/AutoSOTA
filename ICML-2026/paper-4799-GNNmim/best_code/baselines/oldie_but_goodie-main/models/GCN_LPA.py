from embedder import embedder
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import torch
import datetime
import utils
from tqdm import trange
from layers import GCN, GAT, SAGE, SGC, MLP, GCNConv
from sklearn.decomposition import PCA
from torch_geometric.nn.models import LabelPropagation
from filling_strategies import filling
from torch_geometric.nn.inits import reset, glorot, zeros


class GCN_LPA():
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
            model = modeler(self.args, self.edge_index.shape[1]).to(self.args.device)
            optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
            self.x = torch.nan_to_num(self.x, 0)
            # tmp_edge_index = self.edge_index
            # filled_features_fp = filling('fp', tmp_edge_index, self.x, self.missing_feature_mask, self.args.hop, self.args.replace, self.args.num_iterations, self.args.normalize_feature)
            # self.x = torch.where(self.missing_feature_mask, self.x, filled_features_fp)

            acc_vals = []
            test_results = []
            best_metric = 0

            for epoch in range(0, self.args.epochs):
                model.train()
                optimizer.zero_grad()

                output, y_hat = model(self.x, self.edge_index, self.labels, self.train_mask)
                if 'OGBN' in self.args.dataset:
                    labels = self.labels.squeeze(1)
                loss_train = F.cross_entropy(output[self.train_mask], labels[self.train_mask]) \
                    + self.args.lamb * F.cross_entropy(y_hat[self.train_mask], labels[self.train_mask])
                loss_train.backward(retain_graph=True)
                optimizer.step()

                # Valid
                model.eval()
                output, y_hat = model(self.x, self.edge_index, self.labels, self.train_mask)
                acc_val, macro_F_val = utils.performance(output[self.val_mask], self.labels[self.val_mask], pre='valid', evaluator=self.evaluator)

                acc_vals.append(acc_val)
                max_idx = acc_vals.index(max(acc_vals))

                if best_metric <= acc_val:
                    best_metric = acc_val
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


class modeler(nn.Module):
    def __init__(self, args, num_edges):
        super(modeler, self).__init__()
        self.edge_weight = nn.Parameter(torch.ones(num_edges))
        gc = nn.ModuleList()
        gc.append(GCNConv(args.n_feat, args.hidden_dim))
        for i in range(args.num_layers-2):
            gc.append(GCNConv(args.hidden_dim, args.hidden_dim))
        gc.append(GCNConv(args.hidden_dim, args.n_class))
        self.gc = gc
        self.lpa = LabelPropagation(num_layers=50, alpha=args.lp_alpha)
        self.dropout_rate = args.dropout

    def forward(self, x, edge_index, labels, idx_train):
        for i in range(len(self.gc)-1):
            x, _ = self.gc[i](x, edge_index, self.edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout_rate, training=self.training)

        x, _ = self.gc[-1](x, edge_index, self.edge_weight)

        y_hat = self.lpa(labels, edge_index, idx_train, self.edge_weight)

        return x, y_hat
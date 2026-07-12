from embedder import embedder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import datetime
import utils
from layers import GCN, GAT, SAGE, SGC
from torch_geometric.nn import Node2Vec
from tqdm import trange
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

class node2vec_gnn():
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
            # self.adj = self.adj.to(self.args.device)

            model = modeler(self.args, self.edge_index).to(self.args.device)
            
            best_params = None
            val_accs = []
            test_results = []
            train_loss = []
            patience = 20

            for epoch in range(0, 301):
                loss = model._train()
                train_loss.append(loss)
                if epoch % 10 == 0:
                    print(f'[seed {seed}][{self.args.dataset}-{self.args.missing_rate}][{self.args.embedder}] Epoch: {epoch:02d}, Loss: {loss:.4f}')
                if epoch > patience and min(train_loss[-patience :]) >= min(train_loss[: -patience]):
                    break

            # use embedding as initial feature
            model.node2vec.eval()
            self.x = model.node2vec().detach()

            optimizer = optim.Adam(model.classifier.parameters(), lr=self.args.lr)
            
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
                output = model.classifier(self.x, self.edge_index)
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
                    st = "[seed {}][{}-{}][{}-{}][Epoch {}]".format(seed, self.args.dataset, self.args.missing_rate, self.args.gnn, self.args.filling_method, epoch)
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

class modeler(nn.Module):
    def __init__(self, args, edge_index):
        super(modeler, self).__init__()
        self.args = args
        self.node2vec = Node2Vec(edge_index, embedding_dim=args.hidden_dim, walk_length=20, context_size=10, walks_per_node=10,p=args.p, q=args.q, sparse=True)
        self.loader = self.node2vec.loader(batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        self.optimizer = optim.SparseAdam(list(self.node2vec.parameters()), lr=0.1)

        ## Model Selection ##
        if args.gnn == 'GCN':
            classifier = GCN(args.n_layer, args.n_hid, args.n_hid, args.n_class, normalize=True, is_add_self_loops=False)
        elif args.gnn == 'GAT':
            classifier = GAT(args.n_layer, args.n_hid, args.n_hid, args.n_class, args.n_head, is_add_self_loops=False)
        elif args.gnn == "SAGE":
            classifier = SAGE(args.n_layer, args.n_hid, args.n_hid, args.n_class)
        elif args.gnn == "SGC":
            classifier = SGC(args.n_hid, args.n_class, args.n_layer, is_add_self_loops=False)
        else:
            raise NotImplementedError("Not Implemented Architecture!")        
        self.classifier = classifier

    def _train(self):
        self.node2vec.train()
        total_loss = 0
        for pos_rw, neg_rw in self.loader:
            self.optimizer.zero_grad()
            loss = self.node2vec.loss(pos_rw.to(self.args.device), neg_rw.to(self.args.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.loader)

    def forward(self, x, edge_index, labels, idx_train):
        output = self.classifier(x, edge_index)
        if 'OGBN' in self.args.dataset:
            labels = labels.squeeze(1)
        loss_nodeclassification = F.cross_entropy(output[idx_train], labels[idx_train])

        return loss_nodeclassification
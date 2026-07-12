from embedder import embedder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import datetime
import utils
from layers import GCN, GAT, SAGE
from torch_geometric.nn import Node2Vec
from tqdm import trange
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

class node2vec_x():
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
            self.x = torch.nan_to_num(self.x, 0)

            model = modeler(self.args, self.edge_index).to(self.args.device)
            
            best_params = None
            val_accs = []
            test_results = []
            patience = 20

            for epoch in range(0, 301):
                loss = model._train()
                val_acc, cls_params = model._val(self.x, self.labels, self.train_mask, self.val_mask)
                if epoch == 0 or val_acc > max(val_accs):
                    best_params = cls_params
                    test_acc = model._test(self.x, self.labels, self.test_mask, best_params)

                val_accs.append(val_acc)
                if epoch > patience and max(val_accs[-patience :]) <= max(val_accs[: -patience]):
                    break
                    
                print(f'[seed {seed}][{self.args.dataset}-{self.args.missing_rate}][{self.args.embedder}] Epoch: {epoch:02d}, Loss: {loss:.4f}, Val_Acc: {val_acc*100:.2f}, Test_Acc: {test_acc*100:.2f}')

            seed_result['acc'].append(float(test_acc)*100)
            seed_result['macro_F'].append(float(0))

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
        self.classifier = Node2Vec(edge_index, embedding_dim=args.hidden_dim, walk_length=20, context_size=10, walks_per_node=10,p=args.p, q=args.q, sparse=True)
        self.loader = self.classifier.loader(batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        self.optimizer = optim.SparseAdam(list(self.classifier.parameters()), lr=0.1)

    def _train(self):
        self.classifier.train()
        total_loss = 0
        for pos_rw, neg_rw in self.loader:
            self.optimizer.zero_grad()
            loss = self.classifier.loss(pos_rw.to(self.args.device), neg_rw.to(self.args.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.loader)

    @torch.no_grad()
    def _val(self, x, labels, train_mask, val_mask, solver='lbfgs', multi_class='auto', *args, **kwargs):
        self.classifier.eval()
        z = self.classifier()
        z = torch.cat([x,z],1)
        train_z = z[train_mask]
        train_y = labels[train_mask]
        val_z = z[val_mask]
        val_y = labels[val_mask]
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                            **kwargs).fit(train_z.detach().cpu().numpy(),
                                        train_y.detach().cpu().numpy())

        val_acc = clf.score(val_z.detach().cpu().numpy(),
                        val_y.detach().cpu().numpy())
        
        return val_acc, clf.coef_

    @torch.no_grad()
    def _test(self, x, labels, test_mask, cls_params):
        self.classifier.eval()
        z = self.classifier()
        z = torch.cat([x,z],1)
        out = z @ torch.FloatTensor(cls_params).to(z.device).T
        pred = out.argmax(1)
        
        if 'OGBN' in self.args.dataset:
            test_acc =  pred[test_mask].eq(labels[test_mask].squeeze()).sum().item() / len(test_mask)
        else:
            test_acc = pred[test_mask].eq(labels[test_mask].squeeze()).sum().item() / test_mask.sum().item()

        return test_acc
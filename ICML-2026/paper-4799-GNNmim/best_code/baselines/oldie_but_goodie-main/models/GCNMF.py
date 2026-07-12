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

from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import GCNConv

class GCNMF():
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
            self.edge_weight=torch.ones((self.edge_index.size(1), ), device=self.edge_index.device)
            self.adj = torch.sparse.FloatTensor(self.edge_index, values=self.edge_weight).to(self.edge_index.device)

            # Main training
            model = GCNmf(self.args, self.x, self.adj).to(self.args.device)
            optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

            acc_vals = []
            test_results = []
            best_metric = 0

            for epoch in range(0, self.args.epochs):
                model.train()
                optimizer.zero_grad()

                loss = model(self.x, self.adj, self.edge_index, self.labels, self.train_mask)

                loss.backward()
                optimizer.step()

                # Valid
                model.eval()
                x = model.gc1(self.x, self.adj)
                output = model.gc2(x, self.edge_index)

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

class GCNmf(torch.nn.Module):
    def __init__(
        self, args, x, adj,
    ):
        super(GCNmf, self).__init__()

        self.gc1 = GCNmfConv(
            args=args,
            in_features=args.n_feat,
            out_features=args.n_hid,
            x=x,
            adj=adj,
            n_components=5,
            dropout=args.dropout,
        )
        self.gc2 = GCNConv(args.n_hid, args.n_class, args.dropout)
        self.dropout = args.dropout
        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, x, adj, edge_index, labels, idx_train):
        x = self.gc1(x, adj)
        output = self.gc2(x, edge_index)
        if 'OGBN' in self.args.dataset:
            labels = labels.squeeze(1)
        loss_nodeclassification = F.cross_entropy(output[idx_train], labels[idx_train])

        return loss_nodeclassification

def ex_relu(mu, sigma):
    is_zero = (sigma == 0)
    sigma[is_zero] = 1e-10
    sqrt_sigma = torch.sqrt(sigma)
    w = torch.div(mu, sqrt_sigma)
    nr_values = sqrt_sigma * (torch.div(torch.exp(torch.div(- w * w, 2)), np.sqrt(2 * np.pi)) +
                              torch.div(w, 2) * (1 + torch.erf(torch.div(w, np.sqrt(2)))))
    nr_values = torch.where(is_zero, F.relu(mu), nr_values)
    return nr_values


def init_gmm(features, n_components):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    init_x = imp.fit_transform(features)
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag').fit(init_x)
    return gmm


class GCNmfConv(nn.Module):
    def __init__(self, args, in_features, out_features, x, adj, n_components, dropout, bias=True):
        super(GCNmfConv, self).__init__()
        self.device = args.device
        self.in_features = in_features
        self.out_features = out_features
        self.n_components = n_components
        self.dropout = dropout
        self.features = x.cpu().numpy()
        self.logp = Parameter(torch.FloatTensor(n_components))
        self.means = Parameter(torch.FloatTensor(n_components, in_features))
        self.logvars = Parameter(torch.FloatTensor(n_components, in_features))
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.adj2 = torch.mul(adj, adj).to(self.device)
        self.x = x
        self.adj = adj
        self.args = args
        self.gmm = None
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        self.gmm = init_gmm(self.features, self.n_components)
        self.logp.data = torch.FloatTensor(np.log(self.gmm.weights_)).to(self.args.device)
        self.means.data = torch.FloatTensor(self.gmm.means_).to(self.args.device)
        self.logvars.data = torch.FloatTensor(np.log(self.gmm.covariances_)).to(self.args.device)

    def calc_responsibility(self, mean_mat, variances):
        dim = self.in_features
        log_n = (- 1 / 2) *\
            torch.sum(torch.pow(mean_mat - self.means.unsqueeze(1), 2) / variances.unsqueeze(1), 2)\
            - (dim / 2) * np.log(2 * np.pi) - (1 / 2) * torch.sum(self.logvars)
        log_prob = self.logp.unsqueeze(1) + log_n
        return torch.softmax(log_prob, dim=0)

    def forward(self, x, adj):
        x_imp = x.repeat(self.n_components, 1, 1)
        x_isnan = torch.isnan(x_imp)
        variances = torch.exp(self.logvars)
        mean_mat = torch.where(x_isnan, self.means.repeat((x.size(0), 1, 1)).permute(1, 0, 2), x_imp)
        var_mat = torch.where(x_isnan,
                              variances.repeat((x.size(0), 1, 1)).permute(1, 0, 2),
                              torch.zeros(size=x_imp.size(), device=self.args.device, requires_grad=True))

        # dropout
        dropmat = F.dropout(torch.ones_like(mean_mat), self.dropout, training=self.training)
        mean_mat = mean_mat * dropmat
        var_mat = var_mat * dropmat

        transform_x = torch.matmul(mean_mat, self.weight)
        if self.bias is not None:
            transform_x = torch.add(transform_x, self.bias)
        transform_covs = torch.matmul(var_mat, self.weight * self.weight)
        conv_x = []
        conv_covs = []
        for component_x in transform_x:
            conv_x.append(torch.spmm(adj, component_x))
        for component_covs in transform_covs:
            conv_covs.append(torch.spmm(self.adj2, component_covs))
        transform_x = torch.stack(conv_x, dim=0)
        transform_covs = torch.stack(conv_covs, dim=0)
        expected_x = ex_relu(transform_x, transform_covs)

        # calculate responsibility
        gamma = self.calc_responsibility(mean_mat, variances)
        expected_x = torch.sum(expected_x * gamma.unsqueeze(2), dim=0)
        return expected_x
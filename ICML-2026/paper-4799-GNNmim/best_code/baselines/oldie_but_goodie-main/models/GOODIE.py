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
from layers import GCN, GAT, SAGE, SGC, MLP
from torch_geometric.nn.models import LabelPropagation
from torch_geometric.utils import to_dense_adj, remove_self_loops
from torch_geometric.nn.inits import reset, glorot, zeros
from sklearn.decomposition import PCA
from filling_strategies import filling
import pickle
from tqdm import trange

class GOODIE():
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
            print(self)
            exit()
            filled_features_fp = filling('fp', self.edge_index, self.x, self.missing_feature_mask, self.args.num_iterations)
            self.x = torch.where(self.missing_feature_mask, self.x, filled_features_fp)

            # Obtain Pseudo Labels
            lp_model = LabelPropagation(num_layers=50, alpha=self.args.lp_alpha)
            lp_output = lp_model(y=self.labels, edge_index=self.edge_index, mask=self.train_mask)
            self.pseudo_labels = lp_output.argmax(1)

            if self.args.lamb != 0.0: 
                lp_prediction = torch.softmax(lp_output, 1).max(1)[0]
                lp_prediction[self.train_mask] = 1.0
                if self.args.scaled:
                    lp_pred_mat = lp_prediction
                else:
                    lp_pred_mat = lp_prediction.unsqueeze(1) @ lp_prediction.unsqueeze(1).T
            else:
                lp_pred_mat = None

            # Leveraging partial labels
            self.edge_weight = None
            self.x = torch.nan_to_num(self.x, 0)

            # Main training
            model = modeler(self.args).to(self.args.device)
            optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

            acc_vals = []
            test_results = []
            best_metric = 0

            for epoch in range(0, self.args.epochs):
                model.train()
                optimizer.zero_grad()

                loss_ce, loss_pseudo = model(self.x, lp_output, self.edge_index, self.edge_weight, self.labels, self.pseudo_labels, self.train_mask, weight_mask=lp_pred_mat)
                loss = loss_ce + self.args.lamb * loss_pseudo

                loss.backward()
                optimizer.step()

                # Valid
                model.eval()

                fp_embed = model.classifier1(self.x, self.edge_index, self.edge_weight, embed=True)
                lp_embed = model.classifier2(lp_output, self.edge_index, self.edge_weight, embed=True)
                        
                fp_ = model.leakyrelu(torch.mm(fp_embed, model.attention))
                lp_ = model.leakyrelu(torch.mm(lp_embed, model.attention))
                values = torch.softmax(torch.cat((fp_, lp_), dim=1), dim=1)

                output_ = (values[:,0].unsqueeze(1) * fp_embed) + (values[:,1].unsqueeze(1) * lp_embed) 
                output = model.classifier3(output_, self.edge_index, self.edge_weight)

                acc_val, macro_F_val = utils.performance(output[self.val_mask], self.labels[self.val_mask], pre='valid', evaluator=self.evaluator)

                acc_vals.append(acc_val)

                if best_metric < acc_val:
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


class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
    
        ## Model Selection ##
        if args.gnn == 'GCN':
            classifier1 = GCN(1, args.n_feat, args.n_hid, args.n_class, normalize=True, is_add_self_loops=False)
            classifier2 = GCN(1, args.n_class, args.n_hid, args.n_class, normalize=True, is_add_self_loops=False)
            classifier3 = GCN(1, args.n_hid, args.n_hid, args.n_class, normalize=True, is_add_self_loops=False)
        else: # currently, GCN is only available
            raise NotImplementedError("Not Implemented Architecture!")

        self.classifier1 = classifier1
        self.classifier2 = classifier2
        self.classifier3 = classifier3
        self.leakyrelu = nn.LeakyReLU(args.leaky_alpha)
        self.attention = nn.Parameter(torch.empty(size=(args.n_hid, 1)))
        glorot(self.attention)

    def forward(self, x, lp_embed, edge_index, edge_weight, labels, pseudo_labels, idx_train, weight_mask=None):
        _fp_embed = self.classifier1(x, edge_index, edge_weight, embed=True)
        _lp_embed = self.classifier2(lp_embed, edge_index, edge_weight, embed=True)

        fp_ = self.leakyrelu(torch.mm(_fp_embed, self.attention))
        lp_ = self.leakyrelu(torch.mm(_lp_embed, self.attention))
        values = torch.softmax(torch.cat((fp_, lp_), dim=1), dim=1)
        output_ = (values[:,0].unsqueeze(1) * _fp_embed) + (values[:,1].unsqueeze(1) * _lp_embed)
        output = self.classifier3(output_, edge_index, edge_weight)
        
        pseudocon_loss = 0
        if self.args.lamb != 0.0:
            if self.args.scaled:
                centroids = torch.empty((self.args.n_class, output_.shape[1])).to(output_.device)
                for i in range(self.args.n_class):
                    idx = pseudo_labels == i
                    centroids[i, :] = (output_[idx] * weight_mask[idx].unsqueeze(1)).mean(0)
                    
                pseudocon_loss = self.pseducon_loss(centroids, pseudo_labels, weight_mask=weight_mask, scaled=True)
            
            else:
                pseudocon_loss = self.pseducon_loss(output_, pseudo_labels, weight_mask=weight_mask, scaled=False)


        if 'OGBN' in self.args.dataset:
            labels = labels.squeeze(1)
        
        loss_nodeclassification = F.cross_entropy(output[idx_train], labels[idx_train])
        
        return loss_nodeclassification, pseudocon_loss


    def pseducon_loss(self, features, labels=None, mask=None, temp=0.07, base_temp=0.07, weight_mask=None, scaled=False):
        # Normalize
        features = F.normalize(features, dim=-1)
        batch_size = features.shape[0]
        if scaled:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.args.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float()

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.args.temp)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(self.args.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        if scaled:
            mean_log_prob_pos = log_prob.sum(1)
        else:
            mean_log_prob_pos = (weight_mask * mask * log_prob).sum(1) / mask.sum(1)

        loss = - (temp / base_temp) * mean_log_prob_pos
        loss = loss.mean()

        return loss
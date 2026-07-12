# from embedder import embedder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from datetime import datetime
import utils_goodie
from tqdm import trange
from layers import GCN, GAT, SAGE, SGC, MLP
from torch_geometric.nn.models import LabelPropagation
from torch_geometric.utils import to_dense_adj, remove_self_loops
from torch_geometric.nn.inits import reset, glorot, zeros
from sklearn.decomposition import PCA
from filling_strategies import filling
import pickle
from tqdm import trange
from embedder import embedder
    
    
class GOODIE():
    def __init__(self, args):
        self.args = args
    
    def training(self, seed):
        file = utils_goodie.set_filename(self.args)
        logger = utils_goodie.setup_logger('./', '-', file)

        seed_result = {'acc': [], 'macro_F': []}
        
        utils_goodie.seed_everything(seed)
        self.args.seed = seed
        self = embedder(self.args)

        # Fill missing features
        filled_features_fp = filling('fp', self.edge_index, self.x, self.missing_feature_mask, self.args.num_iterations)
        self.x = torch.where(self.missing_feature_mask, self.x, filled_features_fp)
        num_imputed = (~self.missing_feature_mask).sum().item()
        total_values = self.missing_feature_mask.numel()
        perc_imputed = 100.0 * num_imputed / total_values

        # print(f"Valori imputati: {num_imputed}/{total_values} = {perc_imputed:.2f}%")

        lp_model = LabelPropagation(num_layers=50, alpha=self.args.lp_alpha)
        lp_output = lp_model(y=self.labels, edge_index=self.edge_index, mask=self.train_mask)
        self.pseudo_labels = lp_output.argmax(1)
    
        # LP Prediction matrix per pseudocon loss
        if self.args.lamb != 0.0: 
            lp_prediction = torch.softmax(lp_output, 1).max(1)[0]
            lp_prediction[self.train_mask] = 1.0
            lp_pred_mat = lp_prediction if self.args.scaled else lp_prediction.unsqueeze(1) @ lp_prediction.unsqueeze(1).T
        else:
            lp_pred_mat = None

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

            loss_ce, loss_pseudo = model(
                self.x, lp_output, self.edge_index, self.edge_weight,
                self.labels, self.pseudo_labels, self.train_mask,
                weight_mask=lp_pred_mat
            )
            loss = loss_ce + self.args.lamb * loss_pseudo
            loss.backward()
            
            optimizer.step()

            # Valid
            model.eval()
            fp_embed = model.classifier1(self.x, self.edge_index, self.edge_weight, embed=True)
            output = model.classifier3(fp_embed, self.edge_index, self.edge_weight)
            
            acc_val, macro_F_val = utils_goodie.performance(output[self.val_mask], self.labels[self.val_mask], pre='valid')
            acc_vals.append(acc_val)

            if best_metric < acc_val:
                best_metric = acc_val
                max_idx = acc_vals.index(max(acc_vals))
                best_output = output[:]

            # Test
            acc_test, macro_F_test = utils_goodie.performance(output[self.test_mask], self.labels[self.test_mask], pre='test')
            test_results.append([acc_test, macro_F_test])
            best_test_result = test_results[max_idx]

            if (epoch - max_idx > self.args.patience) or (epoch+1 == self.args.epochs):
                output = best_output
                best_test_result[0], best_test_result[1] = utils_goodie.performance(
                    output[self.test_mask], self.labels[self.test_mask], pre='test'
                )
                break

        seed_result['acc'].append(float(best_test_result[0]))
        seed_result['macro_F'].append(float(best_test_result[1]))

        acc = seed_result['acc']
        f1 = seed_result['macro_F']
        return acc, f1, self.x

def blockwise_matmul(features, block_size=2048):
    N = features.size(0)
    device = features.device
    result = torch.empty(N, N, device=device)
    for start in range(0, N, block_size):
        end = min(start+block_size, N)
        result[start:end] = features[start:end] @ features.T
    return result

class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
    
        # Model Selection (originale GOODIE)
        if args.gnn == 'GCN':
            # classifier1: usa feature imputate
            self.classifier1 = GCN(1, args.n_feat, args.n_hid, args.n_class, normalize=True, is_add_self_loops=False)
            # classifier2: usa embedding di label propagation
            self.classifier2 = GCN(1, args.n_class, args.n_hid, args.n_class, normalize=True, is_add_self_loops=False)
            # classifier3: combina i due embedding
            self.classifier3 = GCN(1, args.n_hid, args.n_hid, args.n_class, normalize=True, is_add_self_loops=False)
        else:
            raise NotImplementedError("Currently only GCN is supported in GOODIE.")

        self.leakyrelu = nn.LeakyReLU(args.leaky_alpha)
        self.attention = nn.Parameter(torch.empty(size=(args.n_hid, 1)))
        glorot(self.attention)

    def forward(self, x, lp_embed, edge_index, edge_weight, labels, pseudo_labels, idx_train, weight_mask=None):
        # 1. Ottieni embedding dalle feature imputate
        fp_embed = self.classifier1(x, edge_index, edge_weight, embed=True)
        # 2. Ottieni embedding dalla label propagation
        lp_embed = self.classifier2(lp_embed, edge_index, edge_weight, embed=True)

        # 3. Combina gli embedding con un meccanismo di attenzione
        fp_score = self.leakyrelu(torch.mm(fp_embed, self.attention))
        lp_score = self.leakyrelu(torch.mm(lp_embed, self.attention))
        weights = torch.softmax(torch.cat((fp_score, lp_score), dim=1), dim=1)

        combined_embed = (weights[:, 0].unsqueeze(1) * fp_embed) + (weights[:, 1].unsqueeze(1) * lp_embed)

        # 4. Classificazione finale
        output = self.classifier3(combined_embed, edge_index, edge_weight)

        # 5. Loss principale di classificazione
        if 'OGBN' in self.args.dataset:
            labels = labels.squeeze(1)

        loss_ce = F.cross_entropy(output[idx_train], labels[idx_train])

        # 6. Pseudocon loss opzionale
        pseudocon_loss = 0.0
        if self.args.lamb != 0.0:
            if self.args.scaled:
                centroids = torch.empty((self.args.n_class, combined_embed.shape[1])).to(output.device)
                for i in range(self.args.n_class):
                    idx = pseudo_labels == i
                    centroids[i, :] = (combined_embed[idx] * weight_mask[idx].unsqueeze(1)).mean(0)
                
                pseudocon_loss = self.pseducon_loss(centroids, pseudo_labels, weight_mask=weight_mask, scaled=True)
            else:
                pseudocon_loss = self.pseducon_loss(combined_embed, pseudo_labels, weight_mask=weight_mask, scaled=False)

        return loss_ce, pseudocon_loss

    def pseducon_loss(self, features, labels=None, mask=None, temp=0.07, base_temp=0.07, weight_mask=None, scaled=False):
        # Normalize features
        features = F.normalize(features, dim=-1)
        batch_size = features.shape[0]

        if scaled:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.args.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(self.args.device)

        # Similarità
        # anchor_dot_contrast = blockwise_matmul(features)
        # anchor_dot_contrast = anchor_dot_contrast / self.args.temp
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.args.temp)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(self.args.device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        if scaled:
            mean_log_prob_pos = log_prob.sum(1)
        else:
            mean_log_prob_pos = (weight_mask * mask * log_prob).sum(1) / mask.sum(1)

        loss = - (temp / base_temp) * mean_log_prob_pos
        return loss.mean()

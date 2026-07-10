from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch_sparse
from Params import args
from Utils.Utils import calcRegLoss, pairPredict
import math

if args.device == 'gpu':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

if args.init_type == 'xa_uni':
    init = nn.init.xavier_uniform_
elif args.init_type == 'norm':
    init = nn.init.normal_

zeroinit = nn.init.zeros_

def kernel_matrix(x, sigma):
    return torch.exp((torch.matmul(x, x.transpose(0,1)) - 1) / sigma)    ### real_kernel

def hsic(Kx, Ky, m):
    Kxy = torch.mm(Kx, Ky)
    h = torch.trace(Kxy) / m ** 2 + torch.mean(Kx) * torch.mean(Ky) - \
        2 * torch.mean(Kxy) / m
    return h * (m / (m - 1)) ** 2



class GBMB(nn.Module):
    # def __init__(self, targetG, auxG_list, allOne_auxG_list, sum_auxG):
    def __init__(self, targetG, auxG_list, allOne_auxG_list, sum_G):
        super().__init__()
        self.num_user = args.user
        self.num_item = args.item
        self.latdim = args.latdim
        self.gcn_layer = args.gcn_layer
        self.l2_reg = args.l2_reg
        self.alpha = args.alpha
        self.beta = args.beta
        self.sigma = args.sigma
        self.edge_bias = args.edge_bias
        self.batch_size = args.batch
        self.temperature1 = args.temperature1 
        self.temperature2 = args.temperature2
        self.layer_dropout = args.layer_dropout
        self.targetG = targetG
        self.auxG_list = auxG_list
        self.allOne_auxG_list = allOne_auxG_list
        self.sum_G = sum_G
        self.num_auxG = len(auxG_list)

        self.init_parameters()

    def init_parameters(self):
        self.user_embeddings = nn.Parameter(init(torch.empty(args.user, args.latdim)))
        self.item_embeddings = nn.Parameter(init(torch.empty(args.item, args.latdim)))
        self.MLP_list = nn.ModuleList()
        self.liner_agg = nn.Linear(args.latdim * (self.num_auxG + 1), (self.num_auxG + 1))
        self.liner_att = nn.Linear(args.latdim * 2, 1)
        nn.init.xavier_uniform_(self.liner_att.weight)
        self.dropout = nn.Dropout(1 / (self.num_auxG + 1))
        self.layer_weights = nn.Parameter(torch.ones(self.gcn_layer + 1))
        
        for _ in range(self.num_auxG + 1):
            MLP = nn.Sequential(
            nn.Linear(args.latdim * 2, args.latdim),
            nn.ReLU(),
            nn.Linear(args.latdim, 1)
        )
            nn.init.xavier_uniform_(MLP[0].weight)
            nn.init.xavier_uniform_(MLP[2].weight)
            self.MLP_list.append(MLP)
        return None


    def lightgcn(self, Graph_adj, user_embeddings, item_embeddings, layers):
        if Graph_adj.is_coalesced() == False:
            Graph_adj = Graph_adj.coalesce()
        ego_embedding = torch.cat([user_embeddings, item_embeddings], dim=0)
        emb = [ego_embedding]
        for i in range(layers):
            ego_embedding = torch.sparse.mm(Graph_adj, ego_embedding)
            ego_embedding = F.normalize(ego_embedding, p=2, dim=1)
            # Stochastic layer dropout during training
            if self.training and self.layer_dropout > 0:
                mask = torch.bernoulli(torch.full_like(ego_embedding[:, :1], 1 - self.layer_dropout))
                ego_embedding = ego_embedding * mask / (1 - self.layer_dropout)
            emb.append(ego_embedding)
        emb = torch.stack(emb, dim=0)
        weights = F.softmax(self.layer_weights[:layers + 1], dim=0)
        emb = (emb * weights.view(-1, 1, 1)).sum(dim=0)
        user_emb, item_emb = emb[:self.num_user], emb[self.num_user:]
        return user_emb, item_emb
    
    def G_learner(self, graph, user_emb, item_emb, view_Type):
        if graph.is_coalesced() == False:
            graph = graph.coalesce()
        row, col = graph.indices()[0], graph.indices()[1]
        half_edges = int(row.shape[0] / 2)
        row = row[:half_edges]
        col = col[:half_edges] - self.num_user
        row_emb = user_emb[row]
        col_emb = item_emb[col]
        cat_emb = torch.cat([row_emb, col_emb], dim=1)
        logit = self.MLP_list[view_Type](cat_emb).view(-1)
        eps = torch.rand(logit.shape).to(device)
        mask_gate_input = torch.log(eps) - torch.log(1 - eps)
        mask_gate_input = (mask_gate_input + logit) / self.temperature1
        mask_gate_input = torch.sigmoid(mask_gate_input) + self.edge_bias  # self.edge_bias
        weights = mask_gate_input.detach()
        half_values = weights * graph.values()[:half_edges]
        full_row = torch.cat([row, col + self.num_user], dim=0)
        full_col = torch.cat([col + self.num_user, row], dim=0)
        new_indices = torch.stack([full_row, full_col], dim=0)
        new_values = torch.cat([half_values, half_values], dim=0)
        masked_Graph = torch.sparse.FloatTensor(new_indices, new_values, torch.Size([self.num_user + self.num_item, self.num_user + self.num_item]))
        masked_Graph = masked_Graph.coalesce().to(device)
        return masked_Graph
    
    def bpr_loss(self, users, pos_items, neg_items, user_emb, item_emb):
        users_emb = user_emb[users]
        pos_items_emb = item_emb[pos_items]
        neg_items_emb = item_emb[neg_items]
        pos_scores = torch.sum(users_emb * pos_items_emb, dim=-1)
        neg_scores = torch.sum(users_emb * neg_items_emb, dim=-1)
        bpr_loss = - torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-6).mean()
        return bpr_loss

    def reg_loss(self, users, pos_items, neg_items):
        users_emb = self.user_embeddings[users]
        pos_items_emb = self.item_embeddings[pos_items]
        neg_items_emb = self.item_embeddings[neg_items]
        reg_loss = 1/2 * (users_emb.norm(2).pow(2) +
                    pos_items_emb.norm(2).pow(2) +
                    neg_items_emb .norm(2).pow(2)) / float(len(users))
        return reg_loss

    def InfoNce_loss(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, p=2, dim=1), F.normalize(view2, p=2, dim=1)
        score = (view1 @ view2.T) / temperature
        infonce_loss = -torch.diag(F.log_softmax(score, dim=1)).mean()
        return infonce_loss
    
    def hsic_graph(self, users, pos_items, user_emb, item_emb, user_emb_old, item_emb_old):
        ### user part ###
        users = torch.unique(users)
        items = torch.unique(pos_items)
        input_x = user_emb_old[users]
        input_y = user_emb[users]
        input_x = F.normalize(input_x, p=2, dim=1)
        input_y = F.normalize(input_y, p=2, dim=1)
        Kx = kernel_matrix(input_x, self.sigma)
        Ky = kernel_matrix(input_y, self.sigma)
        loss_user = hsic(Kx, Ky, self.batch_size)
        ### item part ###
        input_i = item_emb_old[items]
        input_j = item_emb[items]
        input_i = F.normalize(input_i, p=2, dim=1)
        input_j = F.normalize(input_j, p=2, dim=1)
        Ki = kernel_matrix(input_i, self.sigma)
        Kj = kernel_matrix(input_j, self.sigma)
        loss_item = hsic(Ki, Kj, self.batch_size)
        loss = loss_user + loss_item
        return loss



    def forward(self):
        global_user_emb, global_item_emb = self.lightgcn(self.sum_G, self.user_embeddings, self.item_embeddings, args.gcn_layer)

        target_user_emb, target_item_emb = self.lightgcn(self.targetG, global_user_emb, global_item_emb, args.gcn_layer)

        aux_user_old_emb, aux_item_old_emb = [], []
        aux_user_emb, aux_item_emb = [], []

        for i in range(self.num_auxG):
            user_emb, item_emb = self.lightgcn(self.allOne_auxG_list[i], global_user_emb, global_item_emb, args.mask_gcn_layer)
            aux_user_old_emb.append(user_emb)
            aux_item_old_emb.append(item_emb)

        for i in range(self.num_auxG):
            mask_auxG = self.G_learner(self.allOne_auxG_list[i], target_user_emb, target_item_emb, i)
            user_emb, item_emb = self.lightgcn(mask_auxG, global_user_emb, global_item_emb, args.mask_gcn_layer)
            aux_user_emb.append(user_emb)
            aux_item_emb.append(item_emb)

        all_aux_user_emb = torch.stack(aux_user_emb, dim=0).mean(dim=0)
        all_aux_item_emb = torch.stack(aux_item_emb, dim=0).mean(dim=0)
        user_emb = torch.stack([target_user_emb] + aux_user_emb, dim=0).mean(dim=0)
        item_emb = torch.stack([target_item_emb] + aux_item_emb, dim=0).mean(dim=0) 

        return target_user_emb, target_item_emb, aux_user_old_emb, aux_item_old_emb, aux_user_emb, aux_item_emb, all_aux_user_emb, all_aux_item_emb, user_emb, item_emb
    


    def cal_loss(self, users, pos_items, neg_items):

        target_user_emb, target_item_emb, aux_user_old_emb, aux_item_old_emb, aux_user_emb, aux_item_emb, all_aux_user_emb, all_aux_item_emb, user_emb, item_emb = self.forward()

        ib_loss = 0

        for i in range(self.num_auxG):
            i_ib_loss = self.hsic_graph(users, pos_items, aux_user_emb[i], aux_item_emb[i], aux_user_old_emb[i], aux_item_old_emb[i])
            ib_loss += i_ib_loss
        ib_loss = ib_loss * self.beta / self.num_auxG

        bpr_loss = self.bpr_loss(users, pos_items, neg_items, user_emb, item_emb)

        reg_loss = self.reg_loss(users, pos_items, neg_items) * self.l2_reg

        user_view1, user_view2 = target_user_emb[torch.unique(users)], all_aux_user_emb[torch.unique(users)]
        item_view1, item_view2 = target_item_emb[torch.unique(pos_items)], all_aux_item_emb[torch.unique(pos_items)]
        infonce_loss = (self.InfoNce_loss(user_view1, user_view2, self.temperature2) + self.InfoNce_loss(item_view1, item_view2, self.temperature2)) * self.alpha

        loss = bpr_loss + ib_loss  + infonce_loss

        return loss, bpr_loss, ib_loss, infonce_loss, reg_loss



 
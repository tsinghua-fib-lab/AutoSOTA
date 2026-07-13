import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils import *

class LatentMappingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6):
        super(LatentMappingLayer, self).__init__()
        self.num_layers = num_layers
        self.enc = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim)
        ])
        for i in range(1, num_layers):
            if i == num_layers - 1:
                self.enc.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.enc.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x, dropout=0.1):
        z = self.encode(x, dropout)
        return z

    def encode(self, x, dropout=0.1):
        h = x
        for i, layer in enumerate(self.enc):
            if i == self.num_layers - 1:
                if dropout:
                    h = torch.dropout(h, dropout, train=self.training)
                h = layer(h)
            else:
                if dropout:
                    h = torch.dropout(h, dropout, train=self.training)
                h = layer(h)
                h = F.tanh(h)
        return h


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, order, order_h):
        super(GraphEncoder, self).__init__()
        self.LatentMap = LatentMappingLayer(input_dim, hidden_dim, output_dim, num_layers=2)
        self.order = order
        self.order_h = order_h
        self.alpha = nn.Parameter(torch.Tensor(1, ))
        self.alpha.data = torch.tensor(0.99999)

    def forward(self, x, adj):

        adj = F.normalize(adj, p=2, dim=1)
        z_l = self.low_message_passing(x, adj, self.order)
        z_h = self.high_message_passing(x, adj, self.order_h)
        alpha = torch.sigmoid(self.alpha)

        z = alpha * z_l + (1 - alpha) * z_h
        return z, alpha

    def low_message_passing(self, x, adj, order):
        h = x
        for i in range(order):
            h = torch.matmul(adj, h) + (1 * x)
        return h

    def high_message_passing(self, x, adj_norm, order):
        I = torch.eye(adj_norm.size(0)).to(adj_norm.device)
        high_pass_adj = I - adj_norm

        h = x
        for _ in range(order):
            h = torch.matmul(high_pass_adj, h)
        return h

    def normalize_adj(self, x):
        D = x.sum(1).detach().clone()
        r_inv = D.pow(-1).flatten()
        r_inv = r_inv.reshape((x.shape[0], -1))
        r_inv[torch.isinf(r_inv)] = 0.
        x = x * r_inv
        return x


class EnDecoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim):
        super(EnDecoder, self).__init__()

        self.enc = LatentMappingLayer(feat_dim, hidden_dim, latent_dim, num_layers=2)
        self.dec_f = LatentMappingLayer(latent_dim, hidden_dim, feat_dim, num_layers=2)

    def forward(self, x, dropout=0.1):
        z = self.enc(x, dropout)
        z_norm = F.normalize(z, p=2, dim=1)
        x_pred = torch.sigmoid(self.dec_f(z_norm, dropout))
        return x_pred, z_norm

def fuse_adj_with_masked_similarity(A, S_z, normalize_A=False, beta_up=0.8, beta_low=-0.2):
    if normalize_A:
        A = (A - A.min()) / (A.max() - A.min() + 1e-6)

    adj_enhanced = A.clone()

    # Similarity-based add/remove
    adj_enhanced = torch.where(S_z > beta_up, torch.ones_like(adj_enhanced), adj_enhanced)
    adj_enhanced = torch.where(S_z < beta_low, torch.zeros_like(adj_enhanced), adj_enhanced)

    return adj_enhanced

class MVHGC(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim, order, order_h, class_num=None, num_view=None, num_nodes=None, fusion_logit_init=1.0):
        super(MVHGC, self).__init__()
        self.num_view = num_view
        self.num_nodes = num_nodes
        self.fusion_logits = None
        if num_nodes is not None:
            init_value = float(fusion_logit_init)
            self.fusion_logits = nn.Parameter(torch.full((num_view, num_nodes, 1), init_value))

        self.endecs = nn.ModuleList([
            EnDecoder(feat_dim, hidden_dim, latent_dim) for _ in range(num_view)
        ])

        self.graphencs = nn.ModuleList([
            GraphEncoder(feat_dim, hidden_dim, latent_dim, order, order_h) for _ in range(num_view)
        ])

        self.cluster_layer = [Parameter(torch.Tensor(class_num, latent_dim)) for _ in range(num_view)]
        self.cluster_layer.append(Parameter(torch.Tensor(class_num, latent_dim)))
        for v in range(num_view+1):
            self.register_parameter('centroid_{}'.format(v), self.cluster_layer[v])

    def fuse_hs(self, hs, weights_h=None):
        hs_stack = torch.stack(hs, dim=0)
        if self.fusion_logits is None:
            raise ValueError("fusion_logits is not initialized; pass num_nodes to MVHGC.")
        node_weights = torch.softmax(self.fusion_logits, dim=0)
        if weights_h is not None:
            weights_h_t = torch.tensor(weights_h, device=hs_stack.device, dtype=hs_stack.dtype).view(self.num_view, 1, 1)
            node_weights = node_weights * weights_h_t
            node_weights = node_weights / (node_weights.sum(dim=0, keepdim=True) + 1e-12)
        h_all = torch.sum(node_weights * hs_stack, dim=0)
        return h_all

    def forward(self, Xs, adjs, weights_h, dataset, beta_up0, beta_low0):

        x_preds = []
        z_norms = []
        A_recs = []
        A_rec_norms = []
        S_zs = []
        S_zs_dis = []
        hs = []
        qgs = []
        adj_Ss = []
        adj_Ss_rec = []
        adj_Ss_rec_norm = []
        alphas = []

        for v in range(self.num_view):

            x_pred, z_norm = self.endecs[v](Xs[v])
            x_preds.append(x_pred)
            z_norms.append(z_norm)

            S_z = self.compute_similarity_matrix(z_norm)
            S_zs.append(S_z)

            adj_S = fuse_adj_with_masked_similarity(
                adjs[v],
                S_zs[v],
                normalize_A=True,
                beta_up=beta_up0,
                beta_low=beta_low0
            )

            adj_Ss.append(adj_S)
            adj_S_norm = self.process_adj_S(adj_S, dataset)

            h, alpha = self.graphencs[v](z_norm, adj_S_norm)
            h = F.normalize(h, p=2, dim=-1)
            hs.append(h)
            qg = self.predict_distribution(h, v)
            qgs.append(qg)
            alphas.append(alpha.item())

        h_all = self.fuse_hs(hs, weights_h)

        qg = self.predict_distribution(h_all, -1)
        qgs.append(qg)

        return x_preds, z_norms, A_recs, A_rec_norms, S_zs, S_zs_dis, hs, h_all, qgs, adj_Ss, adj_Ss_rec, adj_Ss_rec_norm, alphas


    def predict_distribution(self, z, v, alpha=1.0):
        c = self.cluster_layer[v]
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - c, 2), 2) / alpha)
        q = q.pow((alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def normalize_adj(self, x):
        D = x.sum(1).detach().clone()
        r_inv = D.pow(-1).flatten()
        r_inv = r_inv.reshape((x.shape[0], -1))
        r_inv[torch.isinf(r_inv)] = 0.
        x = x * r_inv
        return x

    def symmetric_normalize_adjacency(self, adj_matrix):
        # Calculate degree matrix
        degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))

        # Calculate degree matrix's inverse square root
        degree_inv_sqrt = torch.diag(torch.pow(torch.sum(adj_matrix, dim=1), -0.5))

        # Symmetrically normalize adjacency matrix
        normalized_adj_matrix = torch.mm(torch.mm(degree_inv_sqrt, adj_matrix), degree_inv_sqrt)

        return normalized_adj_matrix

    def compute_similarity_matrix(self, X):
        S = torch.matmul(X, X.t())
        return S
        
    def process_adj_S(self, adj_S, dataset):
        dataset = dataset.lower()

        if dataset in ['texas']:
            adj_S_norm = adj_S

        elif dataset in ['chameleon', 'cornell', 'wisconsin']:
            adj_S_norm = self.normalize_adj(adj_S)

        elif dataset in ['acm', 'dblp', 'imdb', 'acm00', 'acm01', 'acm02', 'acm03', 'acm04', 'acm05']:
            adj_S_map = self.normalize_and_scale(adj_S)
            adj_S_rec = self.construct_adjacency_matrix(adj_S_map, threshold=0.5)
            adj_S_norm = self.normalize_adj(adj_S_rec)

        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        return adj_S_norm
    
    def construct_adjacency_matrix(self, Score, threshold=0.5):
        # prob_matrix = torch.sigmoid(Scores)
        adjacency_matrix = (Score > threshold).float()
        return adjacency_matrix

    def normalize_matrix(self, matrix):
        min_val = torch.min(matrix)
        max_val = torch.max(matrix)
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
        return normalized_matrix

    def normalize_and_scale(self, matrix, power=2):
        # Data normalization (Min-Max Scaling)
        min_val = torch.min(matrix)
        max_val = torch.max(matrix)
        normalized_matrix = (matrix - min_val) / (max_val - min_val)

        # Power transformation to expand differences
        scaled_matrix = normalized_matrix ** power

        return scaled_matrix





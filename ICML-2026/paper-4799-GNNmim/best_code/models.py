import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge, GATConv
import numpy as np
import torch.nn as nn
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.mixture import GaussianMixture
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.stats import gaussian_kde, norm, entropy
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, degree, k_hop_subgraph
from torch.nn import ModuleList, Linear, BatchNorm1d
# from torch_scatter import scatter_add
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn import SAGEConv, GATConv, GINConv, GCN2Conv
import random
from torch.nn import Sequential, Linear, ReLU
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='torch_geometric.typing')
warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------
# GCN structure + features
# --------------------
class GCNFull(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2, num_layers=3, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        assert self.num_layers >= 1, "GCN must have at least 2 layers"

        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, num_classes))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, num_classes))

        # Residual projection layers for skip connections
        self.residual_proj = torch.nn.ModuleList()
        if num_layers > 1:
            self.residual_proj.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.residual_proj.append(nn.Linear(hidden_channels, hidden_channels))

        self.dropout = dropout

    def forward(self, x, edge_index):
        if self.num_layers == 1:
            x = F.relu(self.convs[-1](x, edge_index))
        else:
            for i, conv in enumerate(self.convs[:-1]):
                h = conv(x, edge_index)
                x = F.relu(h + self.residual_proj[i](x))
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index)
        return x

class GraphSAGEFull(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2, num_layers=3, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        assert self.num_layers >= 1, "GraphSAGE must have at least 1 layer"

        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, num_classes))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, num_classes))

        self.dropout = dropout

    def forward(self, x, edge_index):
        if self.num_layers == 1:
            x = F.relu(self.convs[-1](x, edge_index))
        else:
            for conv in self.convs[:-1]:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index)
        return x

class GATFull(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2, num_layers=3, dropout=0.5, heads=1):
        super().__init__()
        self.num_layers = num_layers
        assert self.num_layers >= 1, "GAT must have at least 1 layer"

        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GATConv(in_channels, num_classes, heads=heads))
        else:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
            self.convs.append(GATConv(hidden_channels * heads, num_classes, heads=1))

        self.dropout = dropout

    def forward(self, x, edge_index):
        if self.num_layers == 1:
            x = F.relu(self.convs[-1](x, edge_index))
        else:
            for conv in self.convs[:-1]:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index)
        return x

class GINFull(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2, num_layers=3, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        assert self.num_layers >= 1, "GIN must have at least 1 layer"

        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            nn = Sequential(Linear(in_channels, num_classes))
            self.convs.append(GINConv(nn))
        else:
            nn = Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
            self.convs.append(GINConv(nn))
            for _ in range(num_layers - 2):
                nn = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
                self.convs.append(GINConv(nn))
            nn = Sequential(Linear(hidden_channels, num_classes))
            self.convs.append(GINConv(nn))

        self.dropout = dropout

    def forward(self, x, edge_index):
        if self.num_layers == 1:
            x = F.relu(self.convs[-1](x, edge_index))
        else:
            for conv in self.convs[:-1]:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index)
        return x        


class GCNII(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2, num_layers=3, alpha=0.1, theta=0.5, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        self.initial_lin = torch.nn.Linear(in_channels, hidden_channels)
        for layer in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha=alpha, theta=theta, layer=layer+1))
        self.final_lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x0 = F.relu(self.initial_lin(x))
        x = x0
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(conv(x, x0, edge_index))
        x = self.final_lin(x)
        return x

    
# --------------------
# GCNmf
# --------------------
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
    def __init__(self, in_features, out_features, data, n_components, dropout, bias=True):
        super(GCNmfConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_components = n_components
        self.dropout = dropout
        self.features = data.x.cpu().numpy()
        self.logp = Parameter(torch.FloatTensor(n_components))
        self.means = Parameter(torch.FloatTensor(n_components, in_features))
        self.logvars = Parameter(torch.FloatTensor(n_components, in_features))
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.adj2 = torch.mul(data.adj, data.adj).to(device)
        self.gmm = None
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.gcn = GCNConv(in_features, out_features)
        self.fusion_strategy = "hard_soft_switch"

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        self.gmm = init_gmm(self.features, self.n_components)
        self.logp.data = torch.FloatTensor(np.log(self.gmm.weights_)).to(device)
        self.means.data = torch.FloatTensor(self.gmm.means_).to(device)
        self.logvars.data = torch.FloatTensor(np.log(self.gmm.covariances_)).to(device)

    def calc_responsibility(self, mean_mat, variances):
        dim = self.in_features
        log_n = (- 1 / 2) *\
            torch.sum(torch.pow(mean_mat - self.means.unsqueeze(1), 2) / variances.unsqueeze(1), 2)\
            - (dim / 2) * np.log(2 * np.pi) - (1 / 2) * torch.sum(self.logvars)
        log_prob = self.logp.unsqueeze(1) + log_n
        return torch.softmax(log_prob, dim=0)

    def forward(self, x, adj, edge_index):
        # Maschere per feature osservate e mancanti
        # observed_mask = ~torch.isnan(x)          # [N, F]
        # missing_mask_any = torch.isnan(x).any(dim=1, keepdim=True)  # [N, 1]

        # # GCN classico sulle feature osservate (riempiamo i NaN con 0 per non influenzare il prodotto)
        x_filled = x.clone()
        x_filled[torch.isnan(x_filled)] = 0.0
        x_gcn = F.relu(self.gcn(x_filled, edge_index))

        # --- Pipeline GMM per le feature mancanti ---
        x_imp = x.repeat(self.n_components, 1, 1)  # [K, N, F]
        x_isnan = torch.isnan(x_imp)

        variances = torch.exp(self.logvars)  # [K, F]
        mean_mat = torch.where(
            x_isnan,
            self.means.repeat((x.size(0), 1, 1)).permute(1, 0, 2),  # [K, N, F]
            x_imp
        )
        var_mat = torch.where(
            x_isnan,
            variances.repeat((x.size(0), 1, 1)).permute(1, 0, 2),
            torch.zeros_like(x_imp)
        )

        dropmat = F.dropout(torch.ones_like(mean_mat), self.dropout, training=self.training)
        mean_mat = mean_mat * dropmat
        var_mat = var_mat * dropmat

        transform_x = torch.matmul(mean_mat, self.weight)  # [K, N, out_features]
        if self.bias is not None:
            transform_x = transform_x + self.bias
        transform_covs = torch.matmul(var_mat, self.weight * self.weight)  # [K, N, out_features]

        conv_x = [torch.spmm(adj, tx) for tx in transform_x]
        conv_covs = [torch.spmm(self.adj2, tc) for tc in transform_covs]
        transform_x = torch.stack(conv_x, dim=0)        # [K, N, out_features]
        transform_covs = torch.stack(conv_covs, dim=0)  # [K, N, out_features]

        expected_x = ex_relu(transform_x, transform_covs)  # [K, N, out_features]

        gamma = self.calc_responsibility(mean_mat, variances)  # [K, N]
        expected_x = torch.sum(expected_x * gamma.unsqueeze(2), dim=0)  # [N, out_features]
        return expected_x
        # --- Output ibrido: usa GCN per nodi osservati, GMM per nodi con missing ---
        # out = torch.where(missing_mask_any, expected_x, x_gcn)  # [N, out_features]
        
        # --- Maschere base ---
        obs_ratio = (~torch.isnan(x)).float().mean(dim=1, keepdim=True)  # [N, 1]
        missing_mask_any = torch.isnan(x).any(dim=1, keepdim=True).float()  # [N, 1]
        obs_ratio = 0.0
        # --- Fusione ---
        if self.fusion_strategy == 'soft_obs_ratio':  # ✅ Quella che già usi
            out = obs_ratio * x_gcn + (1 - obs_ratio) * expected_x

        elif self.fusion_strategy == 'hard_soft_switch':  # 🔁 Alternativa 1
            weight = missing_mask_any * (1 - obs_ratio)  # solo dove c'è almeno un missing
            out = (1 - weight) * x_gcn + weight * expected_x

        elif self.fusion_strategy == 'entropy_gating':  # 🔐 Alternativa 2
            # gamma: [K, N], da calc_responsibility()
            gamma_safe = gamma + 1e-10  # evita log(0)
            entropy = -torch.sum(gamma_safe * torch.log(gamma_safe), dim=0, keepdim=True).T  # [N, 1]
            entropy = entropy / np.log(self.n_components)
            confidence = 1 - entropy  # [N, 1]
            out = confidence * expected_x + (1 - confidence) * x_gcn

        elif self.fusion_strategy == 'residual_correction': 
            delta = expected_x - x_gcn
            out = x_gcn + (1 - obs_ratio) * delta

        elif self.fusion_strategy == 'feature_mask_proj':  # 🧠 Alternativa 4 (più potente)
            # Richiede: self.feature_mask_proj = nn.Linear(in_features, out_features)
            feature_mask = torch.isnan(x).float()  # [N, in_features]
            mask_embedding = torch.sigmoid(self.feature_mask_proj(feature_mask))  # [N, out_features]
            out = mask_embedding * expected_x + (1 - mask_embedding) * x_gcn

        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        

        return out



class GCNConv_(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(GCNConv_, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.fc = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        self.fc.bias.data.fill_(0)

    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        x = torch.spmm(adj, x)
        return x
    
    
class GCNmf(nn.Module):
    def __init__(self, data, nhid=16, dropout=0.5, n_components=5):
        super(GCNmf, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes
        self.gc1 = GCNmfConv(nfeat, nhid, data, n_components, dropout)
        self.gc2 = GCNConv_(nhid, nclass, dropout)
        self.gc1test = GCNConv(nfeat, nhid)
        self.gc2test = GCNConv(nhid, nclass)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, x, adj, edge_index):

        # x = F.relu(self.gc1test(x, data.edge_index))
        x = self.gc1(x, adj, edge_index)
        # x = self.gc2(x, adj)
        
        x = self.gc2test(x, edge_index)
        # return x
        return F.log_softmax(x, dim=1)
    
# --------------------
# FP from 
# --------------------
def get_symmetrically_normalized_adjacency(edge_index, n_nodes):
    """
    Given an edge_index, return the same edge_index and edge weights computed as
    \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}.
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    # deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    device = edge_index.device if isinstance(edge_index, torch.Tensor) else edge_weight.device
    deg = torch.zeros(n_nodes, device=device)
    col = col.to(device)
    edge_weight = edge_weight.to(device)
    deg.index_add_(0, col, edge_weight)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, DAD

class FeaturePropagation(torch.nn.Module):
    def __init__(self, num_iterations: int):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations

    def propagate(self, x, edge_index: Adj, mask, test_mask):
        # out is inizialized to 0 for missing values. However, its initialization does not matter for the final
        # value at convergence
        out = x
        if mask is not None:
            out = torch.zeros_like(x)
            out[mask] = x[mask]
        n_nodes = x.shape[0]
        adj = self.get_propagation_matrix(out, edge_index, n_nodes, no_propagate_mask=test_mask)

        for _ in range(self.num_iterations):
            # Diffuse current features
            adj = adj.to(out.device)
            out = torch.sparse.mm(adj, out)
            # print("out: ", out)
            # Reset original known features
            out[mask] = x[mask]

        return out

    def get_propagation_matrix(self, x, edge_index, n_nodes, no_propagate_mask=None):
        edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, n_nodes=n_nodes)

        if no_propagate_mask is not None:
            # no_propagate_mask: boolean tensor [num_nodes], True = nodo che NON deve propagare
            source_nodes = edge_index[1]  # source side of each edge
            edge_mask = ~no_propagate_mask[source_nodes]  # Keep edge only if source can propagate
            edge_index = edge_index[:, edge_mask]
            edge_weight = edge_weight[edge_mask]

        adj = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([n_nodes, n_nodes])).to(edge_index.device)
        return adj
    
    
def get_conv(conv_type, input_dim, output_dim):
    if conv_type == "sage":
        return SAGEConv(input_dim, output_dim)
    elif conv_type == "gcn":
        return GCNConv(input_dim, output_dim)
    elif conv_type == "gat":
        return GATConv(input_dim, output_dim, heads=1)
    elif conv_type == "cheb":
        return ChebConv(input_dim, output_dim, K=4)
    else:
        raise ValueError(f"Convolution type {conv_type} not supported")
    
    
class GNN(torch.nn.Module):
    def __init__(
        self, num_features, num_classes, hidden_dim, num_layers=2, dropout=0, conv_type="GCN", jumping_knowledge=False,
    ):
        super(GNN, self).__init__()

        self.convs = ModuleList([get_conv(conv_type, num_features, hidden_dim)])
        for _ in range(num_layers - 2):
            self.convs.append(get_conv(conv_type, hidden_dim, hidden_dim))
        output_dim = hidden_dim if jumping_knowledge else num_classes
        self.convs.append(get_conv(conv_type, hidden_dim, output_dim))

        if jumping_knowledge:
            self.lin = Linear(hidden_dim, num_classes)
            self.jump = JumpingKnowledge(mode="max", channels=hidden_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge

    def forward(self, x, edge_index=None, adjs=None, full_batch=True):
        return self.forward_full_batch(x, edge_index) if full_batch else self.forward_sampled(x, adjs)

    def forward_full_batch(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            xs += [x]

        if self.jumping_knowledge:
            x = self.jump(xs)
            x = self.lin(x)

        return torch.nn.functional.log_softmax(x, dim=1)

    def forward_sampled(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x.log_softmax(dim=1)

    def inference(self, x_all, inference_loader, device):
        """Get embeddings for all nodes to be used in evaluation"""

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in inference_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[: size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all
    
    
# --------------------
# PCFI 
# --------------------

def pcfi(edge_index, X, feature_mask, num_iterations=None, mask_type=None, alpha=None, beta=None):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    propagation_model = PCFI(num_iterations=num_iterations, alpha = alpha, beta=beta)
    return propagation_model.propagate(x=X, edge_index=edge_index, mask=feature_mask, mask_type=mask_type)

class PCFI(torch.nn.Module):
    def __init__(self, num_iterations: int, alpha: float, beta: float):
        super(PCFI, self).__init__()
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta

    def propagate(self, x, edge_index, mask, mask_type, edge_weight = None):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        nv = x.shape[0]
        feat_dim = x.shape[1]
        out = x
        if mask_type == 'structural':
            f_n2d = self.compute_f_n2d(edge_index, mask, mask_type)
            adj_c = self.compute_edge_weight_c(edge_index, f_n2d, nv)
            if mask is not None:
                out = torch.zeros_like(x)
                out[mask] = x[mask]

            for _ in range(self.num_iterations):
                # Diffuse current features
                adj_c = adj_c.to(out.device)
                out = torch.sparse.mm(adj_c, out)
                out[mask] = x[mask]
            f_n2d = f_n2d.repeat(feat_dim,1)
        else:
            out = torch.zeros_like(x)
            if mask is not None:
                out[mask] = x[mask]
            f_n2d = self.compute_f_n2d(edge_index, mask, mask_type, feat_dim)
            print('\n ==== propagation on {feat_dim} channels ===='.format(feat_dim=feat_dim))
            for i in range(feat_dim):
                adj_c = self.compute_edge_weight_c(edge_index, f_n2d[i], nv)
                for _ in range(self.num_iterations):
                    out[:,i] = torch.sparse.mm(adj_c, out[:,i].reshape(-1,1)).reshape(-1)
                    out[mask[:,i],i] = x[mask[:,i],i]
        cor = torch.corrcoef(out.T).nan_to_num().fill_diagonal_(0)
        f_n2d = f_n2d.to(out.device)
        a_1 = (self.alpha ** f_n2d.T) * (out - torch.mean(out, dim=0))
        a_2 = torch.matmul(a_1, cor)
        out_1 = self.beta * (1 - (self.alpha ** f_n2d.T)) * a_2
        out = out + out_1
        return out

    def compute_f_n2d(self, edge_index, feature_mask, mask_type, feat_dim: OptTensor = None):
        nv = feature_mask.shape[0]
        if mask_type == 'structural':
            len_v_0tod_list = []
            f_n2d = torch.zeros(nv, dtype = torch.int)
            v_0 = torch.nonzero(feature_mask[:, 0]).view(-1)
            len_v_0tod_list.append(len(v_0))
            v_0_to_now = v_0
            f_n2d[v_0] = 0
            d = 1
            while True:
                v_d_hop_sub = k_hop_subgraph(v_0, d, edge_index, num_nodes=nv)[0]
                v_d = torch.from_numpy(np.setdiff1d(v_d_hop_sub.cpu(), v_0_to_now.cpu())).to(v_0.device)
                if len(v_d) == 0:
                    break
                f_n2d[v_d] = d
                v_0_to_now = torch.cat([v_0_to_now, v_d], dim=0)
                len_v_0tod_list.append(len(v_d))
                d += 1
        else:
            f_n2d = torch.zeros(feat_dim, nv)
            print('\n ==== compute f_n2d for {feat_dim} channels ===='.format(feat_dim=feat_dim))
            # for i in tqdm(range(feat_dim), mininterval=2):
            for i in range(feat_dim):
                v_0 = torch.nonzero(feature_mask[:,i]).view(-1)
                v_0_to_now = v_0
                f_n2d[i, v_0] = 0
                d=1
                while True:
                    v_d_hop_sub = k_hop_subgraph(v_0, d, edge_index, num_nodes=nv)[0]
                    v_d = torch.from_numpy(np.setdiff1d(v_d_hop_sub.cpu(), v_0_to_now.cpu())).to(v_0.device)
                    if len(v_d) == 0:
                        break
                    f_n2d[i, v_d] = d
                    v_0_to_now = torch.cat([v_0_to_now, v_d], dim=0)
                    d += 1
            print('\n ====== f_n2d is computed ======'.format(feat_dim=feat_dim))
        return f_n2d

  
    def compute_edge_weight_c(self, edge_index, f_n2d, n_nodes):
        # edge_weight_c = torch.zeros(edge_index.shape[1], device=edge_index.device)
        row, col = edge_index[0], edge_index[1]
        # row: destination, col: source
        f_n2d = f_n2d.to(edge_index.device)
        d_row = f_n2d[row]
        d_col = f_n2d[col]
        edge_weight_c = (self.alpha ** (d_col - d_row + 1)).to(edge_index.device)
        # deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
        device = edge_weight_c.device
        deg_W = torch.zeros(n_nodes, dtype=edge_weight_c.dtype, device=device)
        row = row.to(device)
        deg_W.index_add_(0, row, edge_weight_c)
        # deg_W = scatter_add(edge_weight_c, row, dim_size= f_n2d.shape[0])
        deg_W_inv = deg_W.pow_(-1.0)
        deg_W_inv.masked_fill_(deg_W_inv == float("inf"), 0)
        A_Dinv = edge_weight_c * deg_W_inv[row]
        adj = torch.sparse.FloatTensor(edge_index, values= A_Dinv, size=[n_nodes, n_nodes]).to(edge_index.device)

        return adj
    
# --------------------
# FairAC 
# --------------------

class GCNFull_AC(torch.nn.Module): ## BEFORE WAS JUST WITHOUT _AC
        def __init__(self, in_channels, hidden_channels, num_classes):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, num_classes) # self.conv2 = GCNConv(hidden_channels, num_classes)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            return x
        
    
def get_model_fairac(nfeat, args):
    if args.model == "GCN":
        model = GCNFull_AC(nfeat,args.num_hidden,args.num_hidden)
    elif args.model == "GAT":
        heads =  ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT_body(args.num_layers,nfeat,args.num_hidden,heads,args.dropout,args.attn_drop,args.negative_slope,args.residual)
    elif args.model == "SAGE":
        model = SAGE_Body(nfeat, args.num_hidden, args.dropout)
    else:
        print("Model not implement")
        return

    return model


# baseAC, used autoencoder to improve performance.
class BaseAC(nn.Module):
    def __init__(self, feature_dim, transformed_feature_dim,  emb_dim, args):
        super(BaseAC, self).__init__()
        self.fc = torch.nn.Linear(feature_dim, transformed_feature_dim)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        self.fcdecoder = torch.nn.Linear(transformed_feature_dim, feature_dim)
        nn.init.xavier_normal_(self.fcdecoder.weight, gain=1.414)
        self.hgnn_ac = HGNN_AC(in_dim=emb_dim, hidden_dim=args.attn_vec_dim, dropout=args.dropout,
                          activation=F.elu, num_heads=args.num_heads, cuda=args.cuda)
        AC_params = list(self.fc.parameters()) + list(self.fcdecoder.parameters()) + list(self.hgnn_ac.parameters())
        self.optimizer_AC = torch.optim.Adam(AC_params, lr=args.lr, weight_decay=args.weight_decay)

    def forward(self, bias, emb_dest, emb_src, feature_src):
        transformed_features = self.fc(feature_src)
        feature_src_re = self.hgnn_ac(bias,
                                 emb_dest, emb_src,
                                 transformed_features)
        feature_hat = self.fcdecoder(transformed_features)
        return feature_src_re, feature_hat

    def feature_transform(self, features):
        return self.fc(features)

    def feature_decoder(self, transformed_features):
        return self.fcdecoder(transformed_features)

    def loss(self, origin_feature, AC_feature):
        return F.pairwise_distance(self.fc(origin_feature), AC_feature, 2).mean()


class AverageAC(nn.Module):
    def __init__(self):
        super(AverageAC, self).__init__()

    def forward(self, adj, feature_src):
        degree = [max(1,adj[i].sum().item()) for i in range(adj.shape[0])]
        mean_adj = torch.stack([adj[i]/degree[i] for i in range(adj.shape[0])])
        feature_src_re = mean_adj.matmul(feature_src)
        return feature_src_re


class HGNN_AC(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, activation, num_heads, cuda=False):
        super(HGNN_AC, self).__init__()
        self.dropout = dropout
        self.attentions = [AttentionLayer(in_dim, hidden_dim, dropout, activation, cuda) for _ in range(num_heads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, bias, emb_dest, emb_src, feature_src):
        adj = F.dropout(bias, self.dropout, training=self.training)
        x = torch.cat([att(adj, emb_dest, emb_src, feature_src).unsqueeze(0) for att in self.attentions], dim=0)

        return torch.mean(x, dim=0, keepdim=False)


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, activation, cuda=False):
        super(AttentionLayer, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.is_cuda = cuda

        self.W = nn.Parameter(nn.init.xavier_normal_(
            torch.Tensor(in_dim, hidden_dim).type(torch.cuda.FloatTensor if cuda else torch.FloatTensor),
            gain=np.sqrt(2.0)), requires_grad=True)
        self.W2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(hidden_dim, hidden_dim).type(
            torch.cuda.FloatTensor if cuda else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, bias, emb_dest, emb_src, feature_src):
        h_1 = torch.mm(emb_src, self.W)
        h_2 = torch.mm(emb_dest, self.W)

        e = self.leakyrelu(torch.mm(torch.mm(h_2, self.W2), h_1.t()))
        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(bias > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, feature_src)

        return self.activation(h_prime)
    
class GNNf(nn.Module):
    def __init__(self, nfeat, args, num_classes):
        super(GNNf, self).__init__()

        nhid = args.num_hidden
        self.GNNf = get_model_fairac(nfeat, args)
        self.classifier = nn.Linear(nhid, num_classes)
        G_params = list(self.GNNf.parameters()) + list(self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr=args.lr, weight_decay=args.weight_decay)

        self.args = args
        self.criterion = nn.CrossEntropyLoss()  # Multi-class compatible

    def forward(self, g, x):
        z = self.GNNf(x, g)
        y = self.classifier(z)
        return z, y

class FairGnn(nn.Module):
    def __init__(self, nfeat, args, num_classes):
        super(FairGnn, self).__init__()

        nhid = args.num_hidden
        self.GNN = get_model_fairac(nfeat, args)
        self.classifier = nn.Linear(nhid, num_classes)           # output multi-class
        self.classifierSen = nn.Linear(nhid, args.num_sen_class) # sensibile multi-class

        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer_S = torch.optim.Adam(self.classifierSen.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.args = args
        self.criterion = nn.CrossEntropyLoss()  # per la classe principale
        self.criterion_sen = nn.CrossEntropyLoss()  # per la classe sensibile

    def forward(self, g, x):
        z = self.GNN(g, x)
        y = self.classifier(z)       # [N, num_classes]
        s = self.classifierSen(z)    # [N, num_sen_class]
        return z, y, s

class FairAC2(nn.Module):
    def __init__(self, feature_dim, transformed_feature_dim, emb_dim, args):
        super(FairAC2, self).__init__()

        # Encoder
        self.fc = nn.Linear(feature_dim, 2 * transformed_feature_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2 * transformed_feature_dim, transformed_feature_dim)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)
        self.encoder = nn.Sequential(self.fc, self.relu, self.fc2)

        # Decoder
        self.fcdecoder = nn.Linear(transformed_feature_dim, transformed_feature_dim*2)
        self.relu2 = nn.ReLU()
        self.fcdecoder2 = nn.Linear(transformed_feature_dim*2, feature_dim)
        nn.init.xavier_normal_(self.fcdecoder.weight, gain=1.414)
        nn.init.xavier_normal_(self.fcdecoder2.weight, gain=1.414)
        self.decoder = nn.Sequential(self.fcdecoder, self.relu2, self.fcdecoder2)

        # HGNN autoencoder
        self.hgnn_ac = HGNN_AC(in_dim=emb_dim, hidden_dim=args.attn_vec_dim, dropout=args.dropout,
                               activation=F.elu, num_heads=args.num_heads, cuda=args.cuda)

        # Ottimizzatori
        AC_params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.hgnn_ac.parameters())
        self.optimizer_AC = torch.optim.Adam(AC_params, lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer_AE = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                             lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer_AConly = torch.optim.Adam(self.hgnn_ac.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Classificatore sensibile (multi-class)
        self.classifierSen = nn.Linear(transformed_feature_dim, args.num_sen_class)
        self.optimizer_S = torch.optim.Adam(self.classifierSen.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def forward(self, bias, emb_dest, emb_src, feature_src):
        transformed_features = self.encoder(feature_src)
        feature_src_re = self.hgnn_ac(bias, emb_dest, emb_src, transformed_features)
        feature_hat = self.decoder(transformed_features)
        return feature_src_re, feature_hat, transformed_features

    def sensitive_pred(self, transformed_features):
        return self.classifierSen(transformed_features)

    def feature_transform(self, features):
        return self.encoder(features)

    def feature_decoder(self, transformed_features):
        return self.decoder(transformed_features)

    def loss(self, origin_feature, AC_feature):
        return F.pairwise_distance(self.encoder(origin_feature).detach(), AC_feature, 2).mean()

    
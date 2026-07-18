import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        """
        :param x: the input features, shape: (batch_size, n, in_features)
        :param adj: the adjacency matrix, shape: (n, n)
        """
        support = torch.einsum("bnc,ck->bnk", x, self.weight)  # torch.bmm(x, self.weight)
        output = torch.einsum("mn,bnk->bmk", adj, support)  # torch.bmm(adj.unsqueeze(0), support)
        return output


class TgrGCN(nn.Module):
    def __init__(self, config, sim_feats, atk_vars, device='cuda'):
        super(TgrGCN, self).__init__()
        self.input_dim = config.bef_tgr_len
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.trigger_len
        self.init_bound = config.epsilon

        self.constant_MLP = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)  # the first element is set to 0
        )
        self.structure_MLP = nn.Sequential(
            nn.Linear(sim_feats.shape[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 64)  # the first element is set to 0
        )

        self.conv1 = GraphConvolutionLayer(self.input_dim, self.hidden_dim)
        self.conv2 = GraphConvolutionLayer(self.hidden_dim, self.output_dim)

        self.sim_feats = torch.from_numpy(sim_feats).float().to(device)[atk_vars]  # (n, c)

        self.device = device
        self.layer_num = 2

        for m in self.constant_MLP:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.uniform_(m.bias, -0.2, 0.2)

        # self.structure_MLP = self.structure_MLP
        # self.constant_MLP = self.constant_MLP

    def forward(self, x, constant_alpha=0.5):
        """
        :param x: the normalized input of the MLP, shape: (batch_size, input_dim)
        """
        n = self.sim_feats.shape[0]
        assert x.shape[0] % n == 0, 'the batch graph size should be a multiple of the number of variables.'
        x = x.view(-1, n, self.input_dim)
        bias = self.constant_MLP(torch.zeros(x.shape[-2], x.shape[-1]).to(self.device))
        bias = torch.tanh(bias) * self.init_bound * constant_alpha

        A = self.cal_structure()
        # symmetric normalization of the adjacency matrix
        D = torch.diag(torch.pow(A.sum(dim=1), -0.5))
        A = torch.matmul(torch.matmul(D, A), D)
        A = A.to(self.device)
        h = self.conv1(x, A)
        h = F.relu(h)
        perturb = self.conv2(h, A)

        perturb = torch.tanh(perturb) * self.init_bound * (1 - constant_alpha)
        # add trigger on x[-1] (last element in history) to ensure the real-time property
        out = perturb + bias + x[..., -1:]
        return out, perturb + bias

    def cal_structure(self):
        """
        calculate the similarity matrix of the variables.
        :return: the similarity matrix, shape: (n, n)
        """
        node_num = self.sim_feats.shape[0]
        node_outs = self.structure_MLP(self.sim_feats.detach())  # (n, c)

        # A = torch.matmul(node_outs, node_outs.T)  # (n, n)
        # # print('A shape', A.shape)
        # A = F.tanh(F.relu(A))
        # cosine similarity
        A = F.cosine_similarity(node_outs.unsqueeze(0), node_outs.unsqueeze(1), dim=-1)
        A[A < 0] *= 0
        # add the self-loop
        identity = torch.eye(node_num).to(self.device)
        A = (1 - identity) * A + identity
        return A

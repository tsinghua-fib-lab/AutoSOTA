import os
import time
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import MeanMetric
from itertools import chain
from functools import partial


from .base import PreTrain
from ..defines import GRAPH_TASKS, NODE_TASKS
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from ..data import load4node, load4graph, NodePretrain
from ..model import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


# loss function: sig
def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss


def mask_edge(graph, mask_prob):
    E = graph.num_edges()
    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


# graph transformation: drop edge
def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = Data(edge_index=torch.concat((nsrc, ndst), 0))
    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


def initialize_gnn_decoder(gnn_type, input_dim, hid_dim, num_layer, device):
    if gnn_type == 'GAT':
        gnn = GAT(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    elif gnn_type == 'GCN':
        gnn = GCN(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    elif gnn_type == 'GraphSAGE':
        gnn = GraphSAGE(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    elif gnn_type == 'GIN':
        gnn = GIN(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    elif gnn_type == 'GCov':
        gnn = GCov(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    elif gnn_type == 'GraphTransformer':
        gnn = GraphTransformer(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    else:
        raise ValueError(f"Unsupported GNN type: {gnn_type}")
    gnn.to(device)
    return gnn


class GraphMAELoss(nn.Module):
    def __init__(self, encoder, decoder, hidden_dim, enc_in_dim, dec_in_dim, mask_rate=0.75, drop_edge_rate=0.0,
                 replace_rate=0.1, loss_fn='sce', alpha_l=2):
        super(GraphMAELoss, self).__init__()
        self._mask_rate = mask_rate
        self._drop_edge_rate = drop_edge_rate
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self.hidden_dim = hidden_dim

        # build encoder
        self.encoder = encoder

        # build decoder
        self.decoder = decoder

        self.enc_mask_token = nn.Parameter(torch.zeros(1, enc_in_dim))
        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        # setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    def forward(self, data):

        loss, x_hidden = self.mask_attr_prediction(data)
        loss_item = {"loss": loss.item()}

        return loss, loss_item, x_hidden

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes
        perm = torch.randperm(num_nodes, device=x.device)
        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        if self._replace_rate > 0:
            num_noise_nodes = max(1, int(self._replace_rate * num_mask_nodes))
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0
        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()
        return use_g, out_x, (mask_nodes, keep_nodes)

    def mask_attr_prediction(self, data, pretrain_method='graphmae'):

        g = data
        x = data.x

        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        # if there are noise nodes before reconstruction, then execture this line
        all_hidden = self.encoder(x=use_x, edge_index=use_g.edge_index)

        # if there are none noise nodes before reconstruction, please execture this line
        # all_hidden = self.encoder(data.x, data.edge_index)

        # ---- attribute reconstruction ----

        node_reps = self.encoder_to_decoder(all_hidden)
        node_reps[mask_nodes] = 0

        recon_graph = Data(x=node_reps, edge_index=pre_use_g.edge_index).to(data.x.device)
        recon_node_reps = self.decoder(recon_graph.x, recon_graph.edge_index)

        x_init = x[mask_nodes]
        x_rec = recon_node_reps[mask_nodes]
        loss = self.criterion(x_rec, x_init)
        return loss, all_hidden

    def embed(self, g, x):
        rep = self.encoder(x=x, edge_index=g.edge_index)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])


class GraphMAE(PreTrain):
    def __init__(self, *args, hid_dim=16, mask_rate=0.75, drop_edge_rate=0.0, replace_rate=0.1, loss_fn='sce',
                 alpha_l=2, **kwargs):  # hid_dim=16
        super().__init__(*args, **kwargs)

        self.graph_dataloader = self.load_graph_data()
        self.graph_n_feat_dim = self.input_dim
        self.initialize_gnn(self.input_dim, hid_dim)
        self.decoder = initialize_gnn_decoder(self.gnn_type, hid_dim, self.input_dim, self.num_layer, self.device)
        self.loss = GraphMAELoss(self.gnn, self.decoder, self.hid_dim, self.graph_n_feat_dim, self.hid_dim, mask_rate,
                                 drop_edge_rate, replace_rate, loss_fn, alpha_l).to(self.device)

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, list(self.gnn.parameters()) + list(self.decoder.parameters())),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def load_graph_data(self):

        if self.dataset_name in NODE_TASKS:
            data, in_node_feat_dim, _ = load4node(self.dataset_name)
            graph_list = NodePretrain(
                data=data,
                num_parts=200,
                split_method='Cluster'
            )  # NodePretrain does not have a dataname parameter; here it should be of the data type in pyG
            # graph_list, in_node_feat_dim = NodePretrain(dataname = self.dataset_name, num_parts=200)
            # data = Batch.from_data_list(graph_list)
        elif self.dataset_name in GRAPH_TASKS:
            in_node_feat_dim, _, graph_list = load4graph(self.dataset_name, pretrained=True)
            # data = Batch.from_data_list()
        self.input_dim = in_node_feat_dim
        if self.dataset_name == 'ogbg-ppa':
            return DataLoader(graph_list, batch_size=256, shuffle=True, num_workers=self.num_workers)
        else:
            return DataLoader(graph_list, batch_size=64, shuffle=True, num_workers=self.num_workers)

    def pretrain(self):
        loss_metric = MeanMetric()

        train_loss_min = np.inf
        patience = 10
        cnt_wait = 0
        for epoch in range(self.epochs):
            st_time = time.time()

            loss_metric.reset()

            for step, batch in enumerate(self.graph_dataloader):
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                loss, loss_item, x_hidden = self.loss.forward(batch)
                loss.backward()
                self.optimizer.step()
                loss_metric.update(loss.item(), batch.size(0))

            print(f"GraphMAE [Pretrain] Epoch {epoch}/{self.epochs} | Train Loss {loss_metric.compute():.5f} | "
                  f"Cost Time {time.time() - st_time:.3}s")

            if train_loss_min > loss_metric.compute():
                train_loss_min = loss_metric.compute()
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait == patience:
                    print('-' * 100)
                    print('Early stopping at ' + str(epoch) + ' eopch!')
                    break

        folder_path = f"./Experiment/pre_trained_model/{self.dataset_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        torch.save(self.gnn.state_dict(),
                   "{}/{}.{}.{}.pth".format(folder_path, 'GraphMAE', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))

        print("+++model saved ! {}/{}.{}.{}.pth".format(self.dataset_name, 'GraphMAE', self.gnn_type,
                                                        str(self.hid_dim) + 'hidden_dim'))

import torch
import torch.nn as nn
import torch_geometric.nn as pyg
from CNDA import CNDA
from CNAP import CNAP
from heteroconv import HeteroConv
import copy
import random
import torch_geometric
import torch.nn.functional as F

'''
code is based on https://pytorch-geometric.readthedocs.io/en/latest/

'''


class HeteroGraphAugmentor:
    def __init__(self, feature_mask_ratios, edge_drop_rates):
        """
        feature_mask_ratios: dict[node_type] = float (0~1)
        edge_drop_rates: dict[edge_type] = float (0~1), edge_type is (src, rel, dst)
        """
        self.feature_mask_ratios = feature_mask_ratios
        self.edge_drop_rates = edge_drop_rates

    def feature_mask(self, x, ratio):
        if not isinstance(x, torch.Tensor):
            return x
        mask = torch.rand_like(x) < ratio
        return x.masked_fill(mask, 0.0)

    def drop_edges(self, edge_index, drop_rate):
        num_edges = edge_index.size(1)
        keep_num = int(num_edges * (1 - drop_rate))
        perm = torch.randperm(num_edges)[:keep_num]
        return edge_index[:, perm]

    def edge_split_complement(self, edge_index, drop_rate):
        num_edges = edge_index.size(1)
        keep_num = int(num_edges * (1 - drop_rate))

        if keep_num == 0:
            return edge_index[:, :0], edge_index[:, :0]

        perm = torch.randperm(num_edges)
        view1_idx = perm[:keep_num]
        view2_idx = perm[keep_num:]

        edge_index1 = edge_index[:, view1_idx]
        edge_index2 = edge_index[:, view2_idx]
        return edge_index1, edge_index2

    def augment(self, data):
        view = copy.deepcopy(data)

        # Feature masking per node type
        for node_type in view.node_types:
            if 'x' in view[node_type] and node_type in self.feature_mask_ratios:
                ratio = self.feature_mask_ratios[node_type]
                view[node_type].x = self.feature_mask(view[node_type].x, ratio)

        # Edge dropping per edge type
        for edge_type in view.edge_types:
            if edge_type in self.edge_drop_rates:
                drop_rate = self.edge_drop_rates[edge_type]
                edge_index = view[edge_type].edge_index
                view[edge_type].edge_index = self.drop_edges(edge_index, drop_rate)

        return view

    def augment_with_complement(self, data):
        data1 = copy.deepcopy(data)
        data2 = copy.deepcopy(data)

        for node_type in data.node_types:
            if 'x' in data[node_type] and node_type in self.feature_mask_ratios:
                x = data[node_type].x
                ratio = self.feature_mask_ratios[node_type]
                mask = torch.rand_like(x) < ratio  # boolean mask for data1
                x1 = x.masked_fill(mask, 0.0)
                x2 = x.masked_fill(~mask, 0.0)
                data1[node_type].x = x1
                data2[node_type].x = x2

        return data1, data2


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, input_channel, out_channels, num_layers):
        super().__init__()

        self.out_channals = out_channels

        self.pretransform_win = pyg.Linear(input_channel, hidden_channels, bias=False)
        self.pretransform_exp = pyg.Linear(input_channel + out_channels, hidden_channels, bias=False)

        self.post_transform = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.2),
            pyg.Linear(hidden_channels, hidden_channels, bias=False),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(0.2, True),
        )
        self.pretransform_ey = pyg.Linear(out_channels, hidden_channels, bias=False)
        self.leaklyrelu = nn.LeakyReLU(0.2)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('target', 'near', 'target'): pyg.SAGEConv(hidden_channels, hidden_channels),
                ('reference', 'close', 'reference'): pyg.SAGEConv(hidden_channels, hidden_channels),
                ('reference', 'refer', 'target'): CNDA((hidden_channels, hidden_channels, hidden_channels),
                                                       hidden_channels)
            }, aggr='mean')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):

        x_dict["reference_y"] = self.pretransform_ey(x_dict["reference"][:, -self.out_channals:])
        x_dict["reference"] = self.post_transform(self.pretransform_exp(x_dict["reference"]))
        x_dict['target'] = self.post_transform(self.pretransform_win(x_dict['target']))

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.leaklyrelu(x) for key, x in x_dict.items()}

        return x_dict


class SpaHGC(torch.nn.Module):
    def __init__(self, num_layers=4, mdim=512, out_channels=217, target=0.1, reference=0.9):
        super().__init__()
        hidden_channels = 256
        input_channel = mdim

        self.encoder = HeteroGNN(hidden_channels, input_channel, out_channels, num_layers)
        self.target_encoder = copy.deepcopy(self.encoder)

        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.augmentor = HeteroGraphAugmentor(
            feature_mask_ratios={"target": target, "reference": reference},
            edge_drop_rates={("target", "near", "target"): 0.5, ("reference", "refer", "target"): 0.4}
        )

        self.pool = CNAP(hidden_channels)
        self.lin = pyg.Linear(hidden_channels, out_channels)


    def forward(self, x_dict, edge_index_dict, val=True):
        data = torch_geometric.data.HeteroData()
        for k, v in x_dict.items():
            data[k].x = v
        for k, v in edge_index_dict.items():
            data[k].edge_index = v

        view1, view2 = self.augmentor.augment_with_complement(data)

        view1_x_dict = self.encoder(view1.x_dict, view1.edge_index_dict)

        with torch.no_grad():
            view2_x_dict = self.target_encoder(view2.x_dict, view2.edge_index_dict)

        z1 = F.normalize(view1_x_dict['target'], dim=-1)
        z2 = F.normalize(view2_x_dict['target'], dim=-1)
        traget_bgrl_loss = 2 - 2 * (z1 * z2).sum(dim=-1).mean()

        z3 = F.normalize(view1_x_dict['reference'], dim=-1)
        z4 = F.normalize(view2_x_dict['reference'], dim=-1)
        reference_bgrl_loss = 2 - 2 * (z3 * z4).sum(dim=-1).mean()

        z1_target = self.pool(view1_x_dict, view1.edge_index_dict)

        if val == True:
            y_pred = self.lin(z1_target)
            return traget_bgrl_loss, reference_bgrl_loss, y_pred
        else:
            x = self.encoder(x_dict, edge_index_dict)
            x = self.pool(x, edge_index_dict)
            x = self.lin(x)
            return x

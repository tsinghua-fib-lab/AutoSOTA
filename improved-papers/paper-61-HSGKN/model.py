import torch
import torch.nn as nn
import torch.nn.init as init
from opt import args


class Model(nn.Module):
    def __init__(self, hiden_path_channel, node_feat_dim, n_classes, hid_dims, norm, dropout):
        super(Model, self).__init__()

        self.hiden_path_channel = hiden_path_channel
        self.hiden_paths_index = list(hiden_path_channel.keys())
        self.node_feat_dim = node_feat_dim

        temp_para = []
        for k, v in hiden_path_channel.items():
            temp_para.append(nn.Linear(k * node_feat_dim, v, bias=False))

        self.hiden_paths_features = nn.ModuleList(temp_para)
        self.hiden_paths_outdim = sum(hiden_path_channel.values())

        # Lightweight attention: single linear + sigmoid (much fewer params)
        self.path_attention = nn.Sequential(
            nn.Linear(self.hiden_paths_outdim, self.hiden_paths_outdim, bias=True),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            torch.nn.Linear(self.hiden_paths_outdim, hid_dims[0], bias=True),
            norm(hid_dims[0]),
            nn.PReLU(),
            nn.Dropout(dropout),
            torch.nn.Linear(hid_dims[0], hid_dims[1], bias=True),
            norm(hid_dims[1]),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dims[1], n_classes, bias=True))
        self.init_weights()

    def init_weights(self):
        for param in self.hiden_paths_features:
            init.xavier_uniform_(param.weight)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)
        for layer in self.path_attention:
            if isinstance(layer, nn.Linear):
                # Initialize to near-identity (small noise)
                init.constant_(layer.weight, 0)
                init.constant_(layer.bias, 0)

    def forward(self, graphs_paths_attrs_tensor):
        split = list(self.hiden_path_channel.keys())
        graphs_paths_attrs_tensor = torch.split(graphs_paths_attrs_tensor, [v * self.node_feat_dim for v in split],
                                                dim=1)
        path_output = torch.empty((graphs_paths_attrs_tensor[0].shape[0], 0), device=args.device)
        for i, linear_layer in enumerate(self.hiden_paths_features):
            a = linear_layer.forward(graphs_paths_attrs_tensor[i])
            path_output = torch.cat((path_output, a), dim=1)
        # Apply lightweight channel attention
        attn = self.path_attention(path_output)
        path_output = path_output * attn
        output = self.fc(path_output)
        return output

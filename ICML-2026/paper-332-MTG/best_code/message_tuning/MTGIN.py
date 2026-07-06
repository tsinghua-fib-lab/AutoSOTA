import torch
import torch.nn as nn
import torch.nn.functional as F
from prompt_graph.utils import act
from prompt_graph.model import GIN


class MTGIN(GIN):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, num_layer=3, JK="last", drop_ratio=0, pool='mean',
                 m_num=10):
        super().__init__(input_dim, hid_dim, out_dim, num_layer, JK, drop_ratio, pool)

        self.MTG_vectors = nn.ParameterList([
            nn.Parameter(torch.Tensor(m_num, conv.nn[0].in_features)) for conv in self.conv_layers
        ])
        self.proj_vectors = nn.ParameterList([
            nn.Linear(conv.nn[0].in_features, m_num) for conv in self.conv_layers
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.MTG_vectors:
            nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
        for proj in self.proj_vectors:
            proj.reset_parameters()

    def freeze_pretrained_params(self):
        for param in super().parameters():
            param.requires_grad = False
        for param in self.MTG_vectors:
            param.requires_grad = True
        for proj in self.proj_vectors:
            for param in proj.parameters():
                param.requires_grad = True

    def forward(self, x, edge_index, batch=None):
        h_list = [x]
        for idx, conv in enumerate(self.conv_layers[0:-1]):
            attention = F.softmax(self.proj_vectors[idx](x), dim=1)
            x = x + attention.mm(self.MTG_vectors[idx])
            x = conv(x, edge_index)
            x = act(x)
            x = F.dropout(x, self.drop_ratio, training=self.training)
            h_list.append(x)

        attention = F.softmax(self.proj_vectors[-1](x), dim=1)
        x = x + attention.mm(self.MTG_vectors[-1])
        x = self.conv_layers[-1](x, edge_index)
        h_list.append(x)

        if self.JK == "last":
            node_emb = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_emb = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)[0]

        if batch is None:
            return node_emb
        else:
            graph_emb = self.pool(node_emb, batch.long())
            return graph_emb
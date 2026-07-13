from torch import nn
import torch

from foundation_model.nn.utils import maybe_cat_emb
from tsl.nn.blocks import RNN, MLPDecoder, LinearReadout
from tsl.nn.layers import DiffConv

from tsl.nn.layers.base import NodeEmbedding
from tsl.nn.models import BaseModel
from tsl.nn.utils import maybe_cat_exog


class STGNNModel(BaseModel):
    return_type = torch.Tensor

    r""""""

    def __init__(self,
                 input_size: int,
                 horizon: int,
                 n_nodes: int,
                 hidden_size: int,
                 emb_size: int,
                 temporal_layers: int,
                 gnn_layers: int,
                 output_size: int = None,
                 exog_size: int = 0,
                 activation: str = 'elu'):
        super(STGNNModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size or input_size
        self.horizon = horizon

        self.emb = NodeEmbedding(n_nodes=n_nodes, emb_size=emb_size)

        self.input_encoder = nn.Linear(
            input_size + exog_size + emb_size,
            hidden_size,
        )

        self.temporal_encoder = RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            return_only_last_state=True,
            n_layers=temporal_layers
        )

        self.gnn_layers = nn.ModuleList([
            DiffConv(in_channels=hidden_size,
                     out_channels=hidden_size,
                     add_backward=True,
                     k=1,
                     activation=activation) for _ in range(gnn_layers)
        ])

        self.decoder = LinearReadout(
            input_size=2 * hidden_size,
            output_size=self.output_size,
            horizon=horizon,
        )

    def message_passing(self, x, edge_index, edge_weight=None):
        for layer in self.gnn_layers:
            x = layer(x=x, edge_index=edge_index, edge_weight=edge_weight, cache_support=True)
        return x

    def forward(self, x, edge_index, edge_weight=None, u=None):
        """"""
        x = maybe_cat_exog(x, u)

        emb = self.emb()
        x = maybe_cat_emb(x, emb)

        # temporal encoding
        # weights are shared across levels
        x = self.input_encoder(x)

        x = self.temporal_encoder(x)

        out = self.message_passing(x,
                                   edge_index=edge_index,
                                   edge_weight=edge_weight)

        out = torch.cat([out, x], dim=-1)

        out = self.decoder(out)
        return out

from typing import Dict
from torch_geometric.typing import Adj, EdgeType, NodeType
from collections import defaultdict
from torch_geometric.nn.conv.hgt_conv import group
import torch_geometric.nn as pyg
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
code is based on https://pytorch-geometric.readthedocs.io/en/latest/

'''

class AttentionAggregator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, tensor_list: list[torch.Tensor]) -> torch.Tensor:
        """
        tensor_list: List of tensors of shape [N, D]
        Returns:
            Aggregated tensor of shape [N, D]
        """
        device = tensor_list[0].device
        stacked = torch.stack(tensor_list, dim=0).to(device)  # [K, N, D]
        K, N, D = stacked.shape

        # reshape for attention computation
        flat = stacked.view(-1, D)  # [K*N, D]
        self.att_mlp = self.att_mlp.to(device)
        scores = self.att_mlp(flat).view(K, N)
        alpha = F.softmax(scores, dim=0)  # across edge types

        # attention-weighted sum
        weighted = (stacked * alpha.unsqueeze(-1)).sum(dim=0)  # [N, D]

        return weighted




class HeteroConv(pyg.HeteroConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_aggr = nn.ModuleDict()
        self.hidden_dim = 256

    def forward(
            self,
            x_dict: Dict[NodeType, torch.Tensor],
            edge_index_dict: Dict[EdgeType, Adj],
            *args_dict,
            **kwargs_dict,
    ) -> Dict[NodeType, torch.Tensor]:
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding node feature
                information for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], Tensor]): A dictionary
                holding graph connectivity information for each individual
                edge type.
            *args_dict (optional): Additional forward arguments of invididual
                :class:`torch_geometric.nn.conv.MessagePassing` layers.
            **kwargs_dict (optional): Additional forward arguments of
                individual :class:`torch_geometric.nn.conv.MessagePassing`
                layers.
                For reference, if a specific GNN layer at edge type
                :obj:`edge_type` expects edge attributes :obj:`edge_attr` as a
                forward argument, then you can pass them to
                :meth:`~torch_geometric.nn.conv.HeteroConv.forward` via
                :obj:`edge_attr_dict = { edge_type: edge_attr }`.
        """
        out_dict = defaultdict(list)
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type

            str_edge_type = '<' + '___'.join(edge_type) + '>'
            if str_edge_type not in self.convs:
                continue

            args = []
            for value_dict in args_dict:
                if edge_type in value_dict:
                    args.append(value_dict[edge_type])
                elif src == dst and src in value_dict:
                    args.append(value_dict[src])
                elif src in value_dict or dst in value_dict:
                    args.append(
                        (value_dict.get(src, None), value_dict.get(dst, None)))

            kwargs = {}
            for arg, value_dict in kwargs_dict.items():
                arg = arg[:-5]
                if edge_type in value_dict:
                    kwargs[arg] = value_dict[edge_type]
                elif src == dst and src in value_dict:
                    kwargs[arg] = value_dict[src]
                elif src in value_dict or dst in value_dict:
                    kwargs[arg] = (value_dict.get(src, None),
                                   value_dict.get(dst, None))

            conv = self.convs[str_edge_type]
            if src == dst:
                out = conv(x_dict[src], edge_index, *args, **kwargs)
            else:
                out_src, out = conv((x_dict[src], x_dict[dst], x_dict["reference_y"]), edge_index, *args,
                                    **kwargs)
                out_dict[src].append(out_src)
            out_dict[dst].append(out)

        for key, value in out_dict.items():
            # if key not in self.attention_aggr:
            #     self.attention_aggr[key] = AttentionAggregator(self.hidden_dim)
            # out_dict[key] = self.attention_aggr[key](value)
            out_dict[key] = group(value, self.aggr)

        out_dict["reference_y"] = x_dict["reference_y"]
        return out_dict

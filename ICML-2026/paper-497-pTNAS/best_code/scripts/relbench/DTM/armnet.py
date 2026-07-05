from einops import rearrange
import torch
import torch.nn as nn
from .entmax import EntmaxBisect
from .layers import MLP

from torch_frame.data import TensorFrame
from torch_frame.data.stats import StatType
from torch_frame import stype
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)

from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder

from typing import Any, Dict, List, Optional




class SparseAttention(nn.Module):
    def __init__(self, nfield: int, d_k: int, nhid: int, nemb: int, alpha: float = 1.5):
        """ Sparse Attention Layer w/ shared bilinear weight -> one-head """
        super(SparseAttention, self).__init__()
        self.sparsemax = nn.Softmax(dim=-1) if alpha == 1. \
            else EntmaxBisect(alpha, dim=-1)

        self.scale = d_k ** -0.5
        self.bilinear_w = nn.Linear(nemb, d_k, bias=False)              # nemb*d_k
        self.query = nn.Parameter(torch.zeros(nhid, d_k))               # nhid*d_k
        self.values = nn.Parameter(torch.zeros(nhid, nfield))           # nhid*nfield
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query, gain=1.414)
        nn.init.xavier_uniform_(self.values, gain=1.414)

    def forward(self, x):
        """
        :param x:       [bsz, nfield, nemb], FloatTensor
        :return:        Att_weights [bsz, nhid, nfield], FloatTensor
        """
        keys = self.bilinear_w(x)                                       # bsz*nfield*d_k
        att_gates = torch.einsum('bfe,oe->bof',
                                 keys, self.query) * self.scale         # bsz*nhid*nfield
        sparse_gates = self.sparsemax(att_gates)                        # bsz*nhid*nfield
        return torch.einsum('bof,of->bof', sparse_gates, self.values)   # bsz*nhid*nfield


class ARMNetModel(nn.Module):
    """
        Model:  Adaptive Relation Modeling Network (Multi-Head)
        Important Hyper-Params: alpha (sparsity), nhead (attention heads), nhid (exponential neurons)
    """
    def __init__(self, nfield: int,  nemb: int,  alpha: float, nhid: int,
                 mlp_nlayer: int, mlp_nhid: int, dropout: float, normalization:str, noutput: int = 1):
        '''
        :param nfield:          Number of Fields
        :param nfeat:           Total Number of Features
        :param nemb:            Feature Embedding size
        :param nhead:           Number of Attention Heads (each with a bilinear attn weight)
        :param alpha:           Sparsity hyper-parameter for ent-max
        :param nhid:            Number of Exponential Neuron
        :param mlp_nlayer:      Number of layers for prediction head
        :param mlp_nhid:        Number of hidden neurons for prediction head
        :param dropout:         Dropout rate
        :param ensemble:        Whether to Ensemble with a DNN
        :param deep_nlayer:     Number of layers for Ensemble DNN
        :param deep_nhid:       Number of hidden neurons for Ensemble DNN
        :param noutput:         Number of prediction output, e.g., 1 for binary cls
        '''
        super().__init__()
        # embedding
        # arm
        self.attn_layer = SparseAttention(nfield, nemb, nhid, nemb, alpha)
        self.arm_bn = nn.BatchNorm1d(nhid)
        # MLP
        self.mlp = MLP(nemb, mlp_nlayer, mlp_nhid, dropout, noutput=noutput, normalization=normalization)

    def forward(self, x):
        """
        :param x:   FloatTensor B*F*E
        :return:    y: [bsz], FloatTensor of size B, for Regression or Classification
        """
        x_arm = x                                                       # bsz*nfield*nemb
        arm_weight = self.attn_layer(x_arm)                             # bsz*nhid*nfield
        x_arm = self.arm_bn(torch.exp(
            torch.einsum('bfe, bof->boe', x_arm, arm_weight)))          # bsz*nhid*nemb
        # average nhid
        x_arm = torch.mean(x_arm, dim = 1)                              # bsz*nemb
        y = self.mlp(x_arm)                                             # bsz*noutput

        return y                                             # bsz*noutput

    def reset_parameters(self):
        self.attn_layer.reset_parameters()
        self.mlp.reset_parameters()
        
class ARMNet(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[stype, List[str]],
        stype_encoder_dict: Optional[Dict[stype, StypeEncoder]] = None,
        dropout_prob: float = 0.2,
        num_layers: int = 2,
        normalization: str = "layer_norm",
        # --- additional parameters for ARMNet
        feat_channels: Optional[int] = None,
        nhid: Optional[int] = 32,
        alpha: float = 1.7,
        mlp_nhid: Optional[int] = None,
    ):
        super().__init__()

        if stype_encoder_dict is None:
            # only support categorical and numerical features
            # numerical -> x*v + b
            # categorical -> unique embedding
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            }

        feat_channels = feat_channels if feat_channels else channels
        nfield = sum(len(col_names) for col_names in col_names_dict.values())
        mlp_nhid = mlp_nhid if mlp_nhid else channels
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=feat_channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )
        self.armnet = ARMNetModel(
            nfield=nfield,
            nemb=feat_channels,
            alpha=alpha,
            nhid=nhid,
            mlp_nlayer=num_layers,
            mlp_nhid=mlp_nhid,
            dropout=dropout_prob,
            normalization=normalization,
            noutput=out_channels
        )
    
    def forward(self, x: TensorFrame) -> torch.Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            x (TensorFrame): Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x, _ = self.encoder(x)  # B*F*E

        y = self.armnet(x)      # B*out_channels

        return y

    def register_parameter(self):
        self.encoder.reset_parameters()
        self.armnet.attn_layer.reset_parameters()
        self.armnet.mlp.reset_parameters()
        
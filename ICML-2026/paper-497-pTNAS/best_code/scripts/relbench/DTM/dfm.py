import torch
from .layers import FactorizationMachine, MLP

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


class DeepFMModel(torch.nn.Module):
    """
    Model:  DeepFM
    Ref:    H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    There is invariant in linear, which we reuse the feature embedding.

    Args:
        nfield (int): Number of feature fields.
        nemb (int): Embedding dimension for each feature.
        mlp_layers (int): Number of MLP layers.
        mlp_hid (int): Hidden dimension for MLP layers.
        dropout (float): Dropout probability for regularization.
        noutput (int): Number of output dimensions (typically 1 for regression/binary classification).
    """

    def __init__(self, nfield, nemb, mlp_layers, mlp_hid, dropout, noutput, normalization: str, reduce_dim: bool = True):
        super().__init__()
        self.linear = torch.nn.Linear(nemb, noutput)
        self.fm = FactorizationMachine(reduce_dim=reduce_dim, normalize=True)
        self.mlp_ninput = nfield*nemb
        self.mlp = MLP(self.mlp_ninput, mlp_layers, mlp_hid,
                       dropout, noutput, normalization)

    def forward(self, x):
        """
        :param x:   FloatTensor B*F*E
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x_emb = x     # B*F*E
        x_linear = torch.mean(self.linear(x_emb), axis=1)  # B * 1
        x_fm = self.fm(x_emb).unsqueeze(-1)               # B * 1
        x_mlp = self.mlp(x_emb.view(-1, self.mlp_ninput))  # B * 1
        # print(x_linear, x_fm, x_mlp)
        y = x_linear + x_mlp + x_fm
        # B * 1
        return y

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        self.mlp.reset_parameters()

class DeepFM(torch.nn.Module):
    """DeepFM model for tabular data with automatic feature encoding.

    Args:
        channels (int): Hidden dimension for MLP layers in the DeepFM model.
        out_channels (int): Number of output dimensions/classes.
        num_layers (int): Number of MLP layers in the deep component.
        col_stats (Dict[str, Dict[StatType, Any]]): Column statistics containing metadata
            for each feature column (e.g., vocabulary size for categorical features).
        col_names_dict (Dict[stype, List[str]]): Dictionary mapping semantic types (stype)
            to lists of column names for that type.
        stype_encoder_dict (Optional[Dict[stype, StypeEncoder]], optional): Dictionary mapping
            semantic types to their respective encoders. If None, defaults to EmbeddingEncoder
            for categorical features and LinearEncoder for numerical features. Defaults to None.
        dropout_prob (float, optional): Dropout probability for regularization in MLP layers.
            Defaults to 0.2.
        feat_channels (Optional[int], optional): Feature embedding dimension. If None, uses
            the same value as `channels`. Defaults to None.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[stype, List[str]],
        stype_encoder_dict: Optional[Dict[stype, StypeEncoder]] = None,
        dropout_prob: float = 0.2,
        normalization: str = "layer_norm",
        # additional parameters for xDeepFM
        feat_channels: Optional[int] = None,
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

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=feat_channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        nfield = sum(len(col_names) for col_names in col_names_dict.values())
        self.deepfm = DeepFMModel(
            nfield=nfield,
            nemb=feat_channels,
            mlp_layers=num_layers,
            mlp_hid=channels,
            dropout=dropout_prob,
            noutput=out_channels,
            normalization=normalization,
        )

        self.reset_parameters()

    def forward(self, x: TensorFrame) -> torch.Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            x (TensorFrame): Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x, _ = self.encoder(x)  # B*F*E

        y = self.deepfm(x)      # B*out_channels

        return y

    def reset_parameters(self):
        """Reset model parameters"""
        self.encoder.reset_parameters()
        self.deepfm.reset_parameters()

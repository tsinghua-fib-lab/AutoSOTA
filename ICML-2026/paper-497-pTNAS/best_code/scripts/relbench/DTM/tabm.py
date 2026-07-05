

# refer to
# TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling
# https://arxiv.org/abs/2410.24210

import torch


from torch_frame.data import TensorFrame
from torch_frame.data.stats import StatType
from torch_frame import stype
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder

from typing import Union, Optional, Any, Dict, List


def _init_rsqrt_uniform_(tensor: torch.Tensor, d: int) -> torch.Tensor:
    assert d > 0, "d must be positive"
    d_rsqrt = d**(-0.5)
    with torch.inference_mode():
        return torch.nn.init.uniform_(tensor, -d_rsqrt, d_rsqrt)


def _init_random_signs_(tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        signs = torch.randint(0, 2, tensor.shape,
                              dtype=tensor.dtype, device=tensor.device)
        signs = 2 * signs - 1  # Convert {0, 1} to {-1, 1}
        tensor.copy_(signs)
    return tensor


class BatchEnsembleLayer(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 k: int,
                 dropout_prob: float = 0.2,
                 activation: str = 'relu'):
        """BatchEnsemble Layer.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            k (int): Number of ensemble members.

        # R (k, in_channels)
        # S (k, out_channels)
        # W (in_channels, out_channels)
        # B (k, out_channels)

        """
        super().__init__()

        self.W = torch.nn.Parameter(
            torch.empty(in_channels, out_channels)
        )

        self.R = torch.nn.Parameter(
            torch.empty(k, in_channels)
        )

        self.S = torch.nn.Parameter(
            torch.empty(k, out_channels)
        )

        self.B = torch.nn.Parameter(
            torch.empty(k, out_channels)
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.normalization = LayerNormEnsemble(
            normalized_shape=out_channels, k=k)

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, k, in_channels)        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, k, out_channels)
        """

        k = x.size(1)
        assert k == self.k, f"Input tensor's k dimension ({k}) does not match layer's k ({self.k})"

        # Apply the BatchEnsemble transformation
        # Step 1: Element-wise multiply with R
        x = x * self.R.unsqueeze(0)  # (batch_size, k, in_channels)

        # step 2: shared linear transformation with W
        x = torch.matmul(x, self.W)  # (batch_size, k, out_channels)

        # step 3. Element-wise multiply with S and add B
        # (batch_size, k, out_channels)
        x = x * self.S.unsqueeze(0) + self.B.unsqueeze(0)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

    def reset_parameters(self) -> None:
        # random signs initialization
        _init_rsqrt_uniform_(self.W, self.in_channels)
        _init_random_signs_(self.R)
        _init_random_signs_(self.S)
        self.normalization.reset_parameters()

        bias_init = torch.empty(
            self.out_channels,
            dtype=self.W.dtype,
            device=self.W.device
        )
        _init_rsqrt_uniform_(bias_init, self.in_channels)

        with torch.inference_mode():
            self.B.copy_(bias_init.expand_as(self.B))


class LayerNormEnsemble(torch.nn.Module):

    def __init__(self,
                 normalized_shape: Union[int, list[int], torch.Size],
                 *,
                 k: int,
                 elementwise_affine: bool = True,
                 eps: float = 1e-5,
                 **kwargs,
                 ):

        super().__init__()

        self.ln = torch.nn.LayerNorm(
            normalized_shape, elementwise_affine=False, eps=eps, **kwargs)
        self.k = k
        self.normalized_shape = tuple([normalized_shape]) if isinstance(
            normalized_shape, int) else normalized_shape
        if elementwise_affine:
            self.gamma = torch.nn.Parameter(
                torch.ones(k, *self.normalized_shape)
            )

            self.bias = torch.nn.Parameter(
                torch.zeros(k, *self.normalized_shape)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, k, *normalized_shape)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, k, *normalzied_shape)
        """
        batch_size, k = x.size(0), x.size(1)
        assert k == self.k, f"Input tensor's k dimension ({k}) does not match layer's k ({self.k})"

        x = self.ln(x)

        if self.gamma is not None:
            x = x * self.gamma.unsqueeze(0) + self.bias.unsqueeze(0)

        return x

    def reset_parameters(self) -> None:
        if self.gamma is not None:
            torch.nn.init.ones_(self.gamma)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
            
class TabMEnsemble(torch.nn.Module):

    def __init__(self,
                 nfields: int,
                 channels: int,
                 num_layers: int,
                 k: int = 8,
                 dropout_prob: float = 0.2,
                 activation: str = 'relu'):
        """TabM Ensemble model.
        There we choose the BatchEnsemble verison of TabM. 
        Specifically: R,S,B are ensembled, while W is shared.

        Args:
            nfields (int): Number of feature fields.
            channels (int): Hidden dimension for layers.
            out_channels (int): Number of output dimensions/classes.
            num_layers (int): Number of layers in the model.
            k (int): Number of ensemble members.
        """

        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.k = k
        in_channels = nfields * channels
        for _ in range(num_layers):
            layer = BatchEnsembleLayer(
                in_channels=in_channels,
                out_channels=channels,
                k=k,
                dropout_prob=dropout_prob,
                activation=activation
            )
            self.layers.append(layer)
            in_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, nfields * channels)        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels)
        """
        # expand x to (batch_size, k, nfields * channels)
        x = x.unsqueeze(1).expand(-1, self.k, -1)
        
        for layer in self.layers:
            x = layer(x)

        x = x.mean(dim=1)

        return x

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()



class TabM(torch.nn.Module):
    """TabM model placeholder.
    There we choose the BatchEnsemble verison of TabM. 
    Specifically: R,S,B are ensembled, while W is shared.

    """

    def __init__(self,
                 channels: int,
                 out_channels: int,
                 num_layers: int,
                 col_stats: Dict[str, Dict[StatType, Any]],
                 col_names_dict: Dict[stype, List[str]],
                 stype_encoder_dict: Optional[Dict[stype,
                                                   StypeEncoder]] = None,
                 dropout_prob: float = 0.2,
                 normalization: str = "layer_norm",
                 # additional parameters for TabM
                 k: int = 8):
        super().__init__()

        if stype_encoder_dict is None:
            # only support categorical and numerical features
            # numerical -> x*v + b
            # categorical -> unique embedding
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            }

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )
        nfield = sum(len(col_names) for col_names in col_names_dict.values())
        self.tabm = TabMEnsemble(
            nfields=nfield,
            channels=channels,
            num_layers=num_layers,
            k=k,
            dropout_prob=dropout_prob,
        )
        self.linear = torch.nn.Linear(channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.tabm.reset_parameters()
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        
    def forward(self, x: TensorFrame) -> torch.Tensor:
        """
        Args:
            x (TensorFrame): Input TensorFrame object.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels)
        """
        x, _ = self.encoder(x) # (B, nfields, channels)
        # flatten to (batch_size, nfields * channels)
        x = x.view(x.size(0), -1)
        x = self.tabm(x)
        x = self.linear(x)
        return x

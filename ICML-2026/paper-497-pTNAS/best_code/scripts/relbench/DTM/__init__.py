from .dfm import DeepFM
from .tabm import TabM
from .armnet import ARMNet
from torch_frame.nn.models import MLP, ResNet, FTTransformer
import copy
from typing import Dict, Any
import torch.nn as nn


def construct_tabular_model(model_name: str, model_args: Dict[str, Any]) -> nn.Module:
    """Construct a tabular model based on the model name and arguments.

    Args:
        model_name (str): Name of the model architecture. Supported models:
            - "MLP": Multi-Layer Perceptron
            - "ResNet": Residual Network
            - "FTTrans": Feature Tokenizer Transformer
            - "DFM": DeepFM (Deep Factorization Machine)
        model_args (Dict[str, Any]): Dictionary containing model arguments. Common arguments:
            - channels (int): Hidden dimension for layers
            - out_channels (int): Number of output dimensions/classes
            - num_layers (int): Number of layers in the model
            - dropout_prob (float): Dropout probability
            - normalization (str): Normalization type (not used for FTTrans and DFM)
            - col_names_dict (Dict): Dictionary mapping semantic types to column names
            - stype_encoder_dict (Dict): Dictionary of semantic type encoders
            - col_stats (Dict): Column statistics

    Returns:
        nn.Module: Constructed model instance.

    Raises:
        ValueError: If model_name is not supported.

    Example:
        >>> model_args = {
        ...     "channels": 64,
        ...     "out_channels": 1,
        ...     "num_layers": 2,
        ...     "dropout_prob": 0.2,
        ...     "normalization": "layer_norm",
        ...     "col_names_dict": col_names_dict,
        ...     "stype_encoder_dict": stype_encoder_dict,
        ...     "col_stats": col_stats,
        ... }
        >>> net = construct_tabular_model("MLP", model_args)
    """
    # Make a copy to avoid modifying the original dictionary
    args = copy.deepcopy(model_args)

    if model_name == "MLP":
        net = MLP(**args)
    elif model_name == "ResNet":
        net = ResNet(**args)
    elif model_name == "FTTrans":
        args.pop("normalization", None)
        args.pop("dropout_prob", None)
        net = FTTransformer(**args)
    elif model_name == "DFM":
        net = DeepFM(**args)
    elif model_name == "TabM":
        net = TabM(**args)
    elif model_name == "ARMNet":
        args.pop("normalization", None)
        net = ARMNet(**args)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: MLP, ResNet, FTTrans, DFM")

    return net
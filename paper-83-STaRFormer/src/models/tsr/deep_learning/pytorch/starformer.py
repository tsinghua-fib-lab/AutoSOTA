import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.models.global_modules import SequenceModel
from src.models import STaRFormer
from src.utils import TaskOptions

from .deep_learning_models import DLRegressorPyTorch

__all__ = ['FCNRegressorPyTorch']


class STaRFormerRegressorPyTorch(DLRegressorPyTorch):
    """
    This is a class implementing the STaRFormer model for time series regression.
    """

    def __init__(
            self,
            output_directory,
            input_shape,
            verbose=False,
            epochs=500,
            batch_size=16,
            loss="mean_squared_error",
            metrics=None,
            config=None,
    ):

        self.name = "STaRFormer"
        super().__init__(
            output_directory=output_directory,
            input_shape=input_shape,
            verbose=verbose,
            epochs=epochs,
            batch_size=batch_size,
            loss=loss,
            metrics=metrics,
            config=config
        )
    
    def build_model(self, **kwargs):
        """
        Build the FCN model

        Inputs:
            input_shape: input shape for the model
        """
        activation = initialize_activation_function(config=self.config)

        model = STaRFormer(
            ### embedding
            d_features=self.config.model.sequence_model.embedding.d_features,
            max_seq_len=self.config.model.sequence_model.embedding.max_seq_len,
            ### transformer layer params
            d_model=self.config.model.sequence_model.d_model,
            n_head=self.config.model.sequence_model.n_head,
            num_encoder_layers=self.config.model.sequence_model.num_encoder_layers,
            dim_feedforward=self.config.model.sequence_model.dim_feedforward,
            dropout=self.config.model.sequence_model.dropout,
            activation=activation,
            layer_norm_eps=self.config.model.sequence_model.layer_norm_eps,
            batch_first=self.config.model.sequence_model.batch_first,
            bias=self.config.model.sequence_model.bias,
            enable_nested_tensor=self.config.model.sequence_model.enable_nested_tensor,
            mask_check=self.config.model.sequence_model.mask_check,
            masking=self.config.model.sequence_model.masking,
            mask_threshold=self.config.model.sequence_model.mask_threshold,
            mask_region_bound=self.config.model.sequence_model.mask_region_bound,
            ratio_highest_attention=self.config.model.sequence_model.ratio_highest_attention,
            aggregate_attn_per_batch=self.config.model.sequence_model.aggregate_attn_per_batch,
            precision=self.config.model.sequence_model.precision,
            reconstruction=self.config.model.sequence_model.reconstruction,
            task=self.config.model.output_head.task,
            batch_size=self.config.model.output_head.batch_size,
            cls_method=self.config.model.output_head.cls_method
        )
        activation_output_head = initialize_activation_function_from_str(
            activation=self.config.model.output_head.activation,
        )
        seq_model_params = {
            'sequence_model': model,
            # Output head
            'task': self.config.model.output_head.task,
            'd_out': self.config.model.output_head.d_out,
            'd_hidden': self.config.model.output_head.d_hidden,
            'activation_output_head': activation_output_head,
            'reduced': self.config.model.output_head.reduced,
            'cls_method': self.config.model.output_head.cls_method,
        }

        return SequenceModel(**seq_model_params)
    

def initialize_activation_function(config: DictConfig):
    if config.model.sequence_model.activation == 'elu':
        activation = F.elu
    elif config.model.sequence_model.activation == 'relu':
        activation = F.relu
    elif config.model.sequence_model.activation == 'selu':
        activation = F.selu
    elif config.model.sequence_model.activation == 'gelu':
        activation = F.gelu
    elif config.model.sequence_model.activation == 'silu':
        activation = F.silu
    elif config.model.sequence_model.activation == 'tanh':
        activation = F.tanh
    elif config.model.sequence_model.activation == 'sigmoid':
        activation = F.sigmoid
    else:
        raise RuntimeError(f'{config.model.sequence_model.activation} not found!')

    return activation

def initialize_activation_function_from_str(activation: str):
    if activation == 'elu':
        activation = nn.ELU()
    elif activation == 'relu':
        activation = nn.ReLU()
    elif activation == 'selu':
        activation = nn.SELU()
    elif activation == 'gelu':
        activation = nn.GELU()
    elif activation == 'silu':
        activation = nn.SiLU()
    elif activation == 'tanh':
        activation = nn.Tanh()
    elif activation == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        raise RuntimeError(f'{activation} not found!')

    return activation
import os.path as osp
import torch
import torch.nn as nn

from omegaconf import DictConfig

from .utils import initialize_activation_function, initialize_activation_function_from_str

try:
    from src.models import (
        GRUNet,
        LSTMNet, 
        RNNNet,
        FCN,
        STaRFormer,
        SequenceModel,
        SequenceTextDualModel
    )
    from src.utils import TrainingMethodOptions, ModelOptions, TaskOptions
    from src.runtime import CentralizedModel

except:
    import sys
    file_path = osp.abspath(__file__)
    dir_path = "/".join(file_path.split("/")[:-4])
    sys.path.append(dir_path)
    from src.models import (
        GRUNet,
        LSTMNet, 
        RNNNet,
        FCN,
        STaRFormer,
        SequenceModel,
        SequenceTextDualModel
    )
    from src.utils import TrainingMethodOptions, ModelOptions, TaskOptions
    from src.runtime import CentralizedModel


def initialize_lightning_model(config: DictConfig, **kwargs):
    """ initialize lightning module"""
    model = initialize_model(config=config, **kwargs)
    print(model)
    if config.datamodule.training_method == TrainingMethodOptions.centralized:
        lightning_model = CentralizedModel(config=config, model=model)
    elif config.datamodule.training_method == TrainingMethodOptions.federated:
        pass
    else:
        raise RuntimeError(f'{config.model.sequence_model.name} not found!')

    return lightning_model


def initialize_model(config: DictConfig, max_seq_len: int=None, dataset: object=None, **kwargs):
    """ Initialize Torch Model"""
    if config.model.sequence_model.name == ModelOptions.gru:
        model = gru(config)
    elif config.model.sequence_model.name == ModelOptions.lstm:
        model = lstm(config)
    elif config.model.sequence_model.name == ModelOptions.rnn:
        model = rnn(config)
    elif config.model.sequence_model.name == ModelOptions.starformer: 
        assert max_seq_len is not None, f'max_seq_len has to be given!'
        model = starformer(config, max_seq_len, dataset=dataset)
    elif config.model.sequence_model.name == 'fcn': 
        model = fcn(config)
    else:
        raise RuntimeError(f'{config.model.sequence_model.name} not found!')
    
    if config.dataset in ['p12', 'p19']:
        model = sequence_text_dual_model(config=config, sequence_model=model)
    else:    
        model = sequence_model(config=config, sequence_model=model)

    return model

def sequence_model(config: DictConfig, sequence_model: nn.Module):
    activation = initialize_activation_function_from_str(config.model.output_head.activation)
    seq_model_params = {
        'sequence_model': sequence_model,
        # Output head
        'task': config.model.output_head.task,
        'd_out': config.model.output_head.d_out,
        'd_hidden': config.model.output_head.d_hidden,
        'activation_output_head': activation,
        'reduced': config.model.output_head.reduced,
        'cls_method': config.model.output_head.cls_method,
    }
    
    #if config.model.output_head.task == TaskOptions.anomaly_detection:
        #seq_model_params['norm_dim'] = config.model.output_head.norm_dim
    
    return SequenceModel(**seq_model_params)

def sequence_text_dual_model(config: DictConfig, sequence_model: nn.Module):
    from transformers import AutoTokenizer, RobertaModel, RobertaConfig
    roberta_config = RobertaConfig(
        hidden_size=config.model.text_model.hidden_size
    ) # irrelevant when loading pretrained weights

    llm = RobertaModel(config=roberta_config).from_pretrained("FacebookAI/roberta-base")
    for param in llm.parameters():
        # freeze roberta parameters, not updated in training :)
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base") # FacebookAI/roberta-base

    activation = initialize_activation_function_from_str(config.model.output_head.activation)
    activation_aligner = initialize_activation_function_from_str(config.model.text_model.aligner.activation)
    return SequenceTextDualModel(
        # sequence_model
        sequence_model=sequence_model,
        # text model
        tokenizer=tokenizer,
        text_model=llm,
        #
        kernel_size=config.model.text_model.aligner.kernel_size,
        activation_aligner=activation_aligner,
        # Output head
        task=config.model.output_head.task,
        d_out=config.model.output_head.d_out,
        d_hidden=config.model.output_head.d_hidden,
        activation_output_head=activation,
        reduced=config.model.output_head.reduced,
        cls_method=config.model.output_head.cls_method,
    )


def gru(config: DictConfig):
    return GRUNet(
        # LSTM
        input_size=config.model.sequence_model.input_size,
        hidden_size=config.model.sequence_model.hidden_size,
        num_layers=config.model.sequence_model.num_layers,
        bias=config.model.sequence_model.bias,
        batch_first=config.model.sequence_model.batch_first,
        dropout=config.model.sequence_model.dropout,
        bidirectional=config.model.sequence_model.bidirectional
    )

def lstm(config: DictConfig):
    return LSTMNet(
        # LSTM
        input_size=config.model.sequence_model.input_size,
        hidden_size=config.model.sequence_model.hidden_size,
        num_layers=config.model.sequence_model.num_layers,
        bias=config.model.sequence_model.bias,
        batch_first=config.model.sequence_model.batch_first,
        dropout=config.model.sequence_model.dropout,
        bidirectional=config.model.sequence_model.bidirectional
    )

def rnn(config: DictConfig):
    return RNNNet(
        # LSTM
        input_size=config.model.sequence_model.input_size,
        hidden_size=config.model.sequence_model.hidden_size,
        num_layers=config.model.sequence_model.num_layers,
        nonlinearity=config.model.sequence_model.nonlinearity,
        bias=config.model.sequence_model.bias,
        batch_first=config.model.sequence_model.batch_first,
        dropout=config.model.sequence_model.dropout,
        bidirectional=config.model.sequence_model.bidirectional,
    )

def fcn(config: DictConfig):
    return FCN(
        # LSTM
        in_channels=config.model.sequence_model.in_channels,
    )

def starformer(config: DictConfig, max_seq_len: int, dataset=None):
    activation = initialize_activation_function(config)
    if config.model.sequence_model.embedding.max_seq_len is None:
        config.model.sequence_model.embedding.max_seq_len = max_seq_len
    
    if max_seq_len > config.model.sequence_model.embedding.max_seq_len:
        config.model.sequence_model.embedding.max_seq_len = max_seq_len
    
    return STaRFormer(
        ### embedding
        d_features=config.model.sequence_model.embedding.d_features,
        max_seq_len=config.model.sequence_model.embedding.max_seq_len,
        ### transformer layer params
        d_model=config.model.sequence_model.d_model,
        n_head=config.model.sequence_model.n_head,
        num_encoder_layers=config.model.sequence_model.num_encoder_layers,
        dim_feedforward=config.model.sequence_model.dim_feedforward,
        dropout=config.model.sequence_model.dropout,
        activation=activation,
        layer_norm_eps=config.model.sequence_model.layer_norm_eps,
        batch_first=config.model.sequence_model.batch_first,
        bias=config.model.sequence_model.bias,
        #norm=,
        enable_nested_tensor=config.model.sequence_model.enable_nested_tensor,
        mask_check=config.model.sequence_model.mask_check,
        masking=config.model.sequence_model.masking,
        mask_threshold=config.model.sequence_model.mask_threshold,
        mask_region_bound=config.model.sequence_model.mask_region_bound,
        ratio_highest_attention=config.model.sequence_model.ratio_highest_attention,
        aggregate_attn_per_batch=config.model.sequence_model.aggregate_attn_per_batch,
        precision=config.model.sequence_model.precision,
        reconstruction=config.model.sequence_model.reconstruction,
        task=config.model.output_head.task,
        batch_size=config.model.output_head.batch_size,
        cls_method=config.model.output_head.cls_method
    )
"""
import os.path as osp
import torch
import torch.nn as nn

from omegaconf import DictConfig

from .utils import initialize_activation_function, initialize_activation_function_from_str

try:
    from src.models import (
        GRUNet,
        LSTMNet, 
        RNNNet,
        STaRFormer,
        SequenceModel,    
        SequenceTextDualModel,
    )
    from src.utils import TrainingMethodOptions, ModelOptions, DatasetOptions
    from src.runtime import CentralizedModel

except:
    import sys
    file_path = osp.abspath(__file__)
    dir_path = "/".join(file_path.split("/")[:-4])
    sys.path.append(dir_path)
    from src.models import (
        GRUNet,
        LSTMNet, 
        RNNNet,
        STaRFormer,
        SequenceModel,    
        SequenceTextDualModel,
    )
    from src.utils import TrainingMethodOptions, ModelOptions, DatasetOptions
    from src.runtime import CentralizedModel


def initialize_lightning_model(config: DictConfig, **kwargs):
    #initialize lightning module
    model = initialize_model(config=config, **kwargs)
    print(model)
    if config.datamodule.training_method == TrainingMethodOptions.centralized:
        lightning_model = CentralizedModel(config=config, model=model)
    elif config.datamodule.training_method == TrainingMethodOptions.federated:
        pass
    else:
        raise RuntimeError(f'{config.model.name} not found!')

    return lightning_model


def initialize_model(config: DictConfig, max_seq_len: int=None, dataset: object=None, **kwargs):
    # Initialize Torch Model
    if config.model.sequence_model.name == ModelOptions.gru:
        model = gru(config)
    elif config.model.sequence_model.name == ModelOptions.lstm:
        model = lstm(config)
    elif config.model.sequence_model.name == ModelOptions.rnn:
        model = rnn(config)
    elif config.model.sequence_model.name == ModelOptions.starformer: 
        assert max_seq_len is not None, f'max_seq_len has to be given!'
        model = starformer(config, max_seq_len, dataset=dataset)
    else:
        raise RuntimeError(f'{config.model.sequence_model.name} not found!')

    return sequence_model(sequence_model=model, config=config)

def sequence_model(sequence_model: nn.Module, config: DictConfig):
    activation = initialize_activation_function_from_str(config.model.output_head.activation)
    activation_text_model = initialize_activation_function_from_str(config.model.text_model.aligner.activation)

    if config.dataset in [DatasetOptions.p12, DatasetOptions.p19]:
        from transformers import AutoTokenizer, RobertaModel, RobertaConfig

        if config.model.text_model.name == 'roberta':
            #rconfig = RobertaConfig(hidden_size=config.model.text_model.hidden_size)
            text_model = RobertaModel.from_pretrained("FacebookAI/roberta-base")
            tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        else:
            raise NotImplementedError(f'{config.model.text_model.name} is not implemented!')
        # Freeze text model params
        for param in text_model.parameters():
            param.requires_grad = False

        return SequenceTextDualModel(
            # sequence
            sequence_model=sequence_model,
            # text
            tokenizer=tokenizer,
            text_model=text_model,
            # feature aligment
            kernel_size=config.model.text_model.aligner.kernel_size,
            activation_aligner=activation_text_model,
            #
            task=config.model.output_head.task,
            d_out=config.model.output_head.d_out,
            d_hidden=config.model.output_head.d_hidden,
            activation_output_head=activation,
            reduced=config.model.output_head.reduced,
            cls_method=config.model.output_head.cls_method,
        )
    else:
        return SequenceModel(
            sequence_model=sequence_model,
            task=config.model.output_head.task,
            d_out=config.model.output_head.d_out,
            d_hidden=config.model.output_head.d_hidden,
            activation_output_head=activation,
            reduced=config.model.output_head.reduced,
            cls_method=config.model.output_head.cls_method,
        )


def gru(config: DictConfig):
    activation = initialize_activation_function_from_str(config.model.output_head.activation)
    return GRUNet(
        # LSTM
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        bias=config.model.bias,
        batch_first=config.model.batch_first,
        dropout=config.model.dropout,
        bidirectional=config.model.bidirectional,
        # Output head
        #task=config.model.output_head.task,
        #d_out=config.model.output_head.d_out,
        #d_hidden=config.model.output_head.d_hidden,
        #activation=activation,
        #reduced=config.model.output_head.reduced,
        #cls_method=config.model.output_head.cls_method,
    )

def lstm(config: DictConfig):
    activation = initialize_activation_function_from_str(config.model.output_head.activation)
    return LSTMNet(
        # LSTM
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        bias=config.model.bias,
        batch_first=config.model.batch_first,
        dropout=config.model.dropout,
        bidirectional=config.model.bidirectional,
        # Output head
        #task=config.model.output_head.task,
        #d_out=config.model.output_head.d_out,
        #d_hidden=config.model.output_head.d_hidden,
        #activation=activation,
        #reduced=config.model.output_head.reduced,
        #cls_method=config.model.output_head.cls_method,
    )

def rnn(config: DictConfig):
    #nonlinearity = initialize_activation_function_from_str(config.model.nonlinearity)
    activation = initialize_activation_function_from_str(config.model.output_head.activation)
    return RNNNet(
        # LSTM
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        nonlinearity=config.model.nonlinearity,
        bias=config.model.bias,
        batch_first=config.model.batch_first,
        dropout=config.model.dropout,
        bidirectional=config.model.bidirectional,
        # Output head
        #task=config.model.output_head.task,
        #d_out=config.model.output_head.d_out,
        #d_hidden=config.model.output_head.d_hidden,
        #activation=activation,
        #reduced=config.model.output_head.reduced,
        #cls_method=config.model.output_head.cls_method,
    )

def starformer(config: DictConfig, max_seq_len: int, dataset=None):
    activation = initialize_activation_function(config)

    if max_seq_len > config.model.sequence_model.embedding.max_seq_len:
        config.model.sequence_model.embedding.max_seq_len = max_seq_len
    
    return STaRFormer(
        ### embedding
        d_features=config.model.sequence_model.embedding.d_features,
        max_seq_len=config.model.sequence_model.embedding.max_seq_len,
        ### transformer layer params
        d_model=config.model.sequence_model.d_model,
        n_head=config.model.sequence_model.n_head,
        num_encoder_layers=config.model.sequence_model.num_encoder_layers,
        dim_feedforward=config.model.sequence_model.dim_feedforward,
        dropout=config.model.sequence_model.dropout,
        activation=activation,
        layer_norm_eps=config.model.sequence_model.layer_norm_eps,
        batch_first=config.model.sequence_model.batch_first,
        bias=config.model.sequence_model.bias,
        enable_nested_tensor=config.model.sequence_model.enable_nested_tensor,
        mask_check=config.model.sequence_model.mask_check,
        masking=config.model.sequence_model.masking,
        mask_threshold=config.model.sequence_model.mask_threshold,
        mask_region_bound=config.model.sequence_model.mask_region_bound,
        ratio_highest_attention=config.model.sequence_model.ratio_highest_attention,
        aggregate_attn_per_batch=config.model.sequence_model.aggregate_attn_per_batch,
        precision=config.model.sequence_model.precision,
        reconstruction=config.model.sequence_model.reconstruction,
        #device=None,
        #dtype=None,
        # Output head
        task=config.model.output_head.task,
        batch_size=config.model.output_head.batch_size,
        #d_out=config.model.output_head.d_out,
        #d_hidden=config.model.output_head.d_hidden,
        #activation_output_head=activation_output_head,
        #reduced=config.model.output_head.reduced,
        cls_method=config.model.output_head.cls_method,
    )

        """
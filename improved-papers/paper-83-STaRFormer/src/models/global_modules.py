import torch
import torch.nn as nn 

from torch import Tensor
from transformers import AutoTokenizer 
from typing import Literal, Union, Callable, Dict, Any, List

from src.nn.functional import FeatureAligner
from .output_heads import OutputHead, OutputHeadOptions


__all__ = ["SequenceModel", "SequenceTextDualModel"]


class SequenceModel(nn.Module):
    """
    Sequence Wrapper Module.

    This module wraps a sequence model (e.g., RNN/LSTM/GRU/STaRFormer/FCN) and
    attaches a configurable output head suitable for classification, anomaly detection,
    regression, or forecasting tasks.

    Attributes:
        sequence_model (nn.Module): The wrapped sequence-processing model.
        _task (str): Target task for the OutputHead (e.g., 'classification',
            'regression', or 'forecasting').
        _cls_method (str): Prediction/method used by the OutputHead (e.g., 'autoregressive',
            'cls_token'); defaults to the configuration provided at initialization.
        output_head (OutputHead): The output head configured for the selected task and
            model dimension.
    """
    def __init__(self, 
                 sequence_model: nn.Module=None,
                 # Output head
                 task: Literal['classification', 'regression', 'forecasting']='classification',
                 d_out: int=1, 
                 d_hidden: int=None, 
                 activation_output_head: Union[str, Callable[[Tensor], Tensor]] = nn.ReLU(),
                 reduced: bool=True, 
                 cls_method: Literal['cls_token', 'regr_token', 'autoregressive', 'elementwise', 'pooling']='autoregressive',
                 # cls synonym for prediction method
                 norm_dim: int=None, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        Initialize a SequenceModel wrapper.

        Args:
            sequence_model (nn.Module, optional): A PyTorch module representing the
                underlying sequence model. Must be an instance of nn.Module and of a
                supported type (RNN/LSTM/GRU, STaRFormer, or FCN). If None, an assertion
                error is raised during initialization.
            task (Literal['classification', 'regression', 'forecasting'], optional): The type
                of output head to use. Defaults to 'classification'.
            d_out (int, optional): Dimensionality of the output. Defaults to 1.
            d_hidden (int, optional): Hidden dimensionality for the output head (if used).
                Defaults to None.
            activation_output_head (Union[str, Callable[[Tensor], Tensor]], optional):
                Activation to apply at the output head. Can be a string or a callable.
                Defaults to nn.ReLU().
            reduced (bool, optional): If True, use a reduced form of the output head.
                Defaults to True.
            cls_method (Literal['cls_token', 'regr_token', 'autoregressive', 'elementwise', 'pooling'], optional):
                Method used by the output head for combining sequence features. Defaults to
                'autoregressive'.
            norm_dim (int, optional): Normalization dimension used by the output head.
                Defaults to None.
            *args, **kwargs: Additional positional and keyword arguments passed to
                the base class (nn.Module).
        """
        assert sequence_model is not None and isinstance(sequence_model, nn.Module)
        # currently accepted models 
        assert sequence_model.__class__.__name__.startswith('RNN') or \
        sequence_model.__class__.__name__.startswith('LSTM') or \
        sequence_model.__class__.__name__.startswith('GRU') or \
        sequence_model.__class__.__name__.startswith('STaRFormer') or \
        sequence_model.__class__.__name__.startswith('FCN'), f'{sequence_model.__class__.__name__} not accepted!'
        self.sequence_model = sequence_model
        
        # output head
        self._task = task
        self._cls_method = cls_method # can also be a regression method
        
        if hasattr(self.sequence_model, '_d_model'):
            d_model= sequence_model._d_model
        elif hasattr(self.sequence_model, '_hidden_size'):
            d_model = sequence_model._hidden_size
        elif self.sequence_model.__class__.__name__ == 'FCN':
            d_model = 128
        else:
            raise RuntimeError(f"Cannot find 'd_model' or 'hidden_size' attribute!")

        self.output_head = OutputHead(task=task, d_model=d_model, d_out=d_out,
                                      d_hidden=d_hidden, activation=activation_output_head, reduced=reduced, 
                                      method=cls_method, **{'norm_dim': norm_dim})
    
    def forward(self, x: Tensor, N: Tensor=None, batch_size: int=None, **kwargs) -> Dict[str, Tensor | Any]:
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor for the sequence_model. Shape depends on the underlying
                sequence model (e.g., [N, B, D] for many RNN-like backends).
            N (Tensor, optional): Auxiliary sequence length or related information passed to certain models
                (e.g., STaRFormer). If not required by the model, may be left as None.
            batch_size (int, optional): Batch size. Required for RNN/LSTM/GRU backends. Must be provided
                when using recurrent models.
            **kwargs: Additional keyword arguments forwarded to the underlying sequence_model. May include
                mode, dataset, padding_mask, or other model-specific flags.

        Returns:
            dict: A dictionary containing at least:
                - 'logits' (Tensor): The final predictions produced by the OutputHead.
                - Other keys produced by the underlying sequence_model (e.g., 'hidden_state',
                'embedding_cls', 'features'), preserved from the model outputs.
        """
        if self.sequence_model.__class__.__name__.startswith('RNN') or \
            self.sequence_model.__class__.__name__.startswith('LSTM') or \
            self.sequence_model.__class__.__name__.startswith('GRU'):
            #assert kwargs.get('batch_size', None) is not None, f'batch_size was not given!'
            assert batch_size is not None, f'batch_size was not given!'
            outputs = self.sequence_model(x, batch_size)
            latent_embedding = outputs['hidden_state']
        
        elif self.sequence_model.__class__.__name__.startswith('STaRFormer'):
            #assert kwargs.get('padding_mask', None) is not None, f'padding_mask not given!'
            #assert isinstance(kwargs.get('padding_mask'), Tensor), f"padding_mask must be a Tensor and not {type(kwargs.get('padding_mask'))}"
            assert kwargs.get('mode', None) is not None, f'mode not given!'
            assert kwargs.get('mode') in ['train', 'val', 'test'], f"mode has to be one of the following ['train', 'val', 'test'], not {kwargs.get('mode')}"
            assert kwargs.get('dataset', None) is not None, f'dataset is not given!'
            assert isinstance(kwargs.get('dataset'), str) is not None, f"dataset must be a str, not {type(kwargs.get('dataset'))}!"
            outputs = self.sequence_model(x, N, **kwargs)
            latent_embedding = outputs['embedding_cls']
        
        elif self.sequence_model.__class__.__name__.startswith('FCN'):
            outputs = self.sequence_model(x)
            latent_embedding = outputs['features']
            
        else:
            raise NotImplementedError

        # output head 
        if self._task in [OutputHeadOptions.classification, OutputHeadOptions.anomaly_detection]:
            logits = self.output_head(x=latent_embedding, N=N, batch_size=batch_size)

        elif self._task == OutputHeadOptions.regression:
            logits = self.output_head(x=latent_embedding, N=N, batch_size=batch_size)

        else:
            raise NotImplementedError(f'{self._task}')

        outputs['logits'] = logits
        return outputs


class SequenceTextDualModel(nn.Module):
    """
    Sequence Text Dual Wrapper Module.

    This module combines a sequence model (e.g., RNN/LSTM/GRU, STaRFormer) with a
    text (RoBERTa-like) model to produce a fused representation that is then passed
    through an OutputHead to generate task-specific predictions (e.g., classification).

    Architecture overview:
        - Process the input sequence with the provided sequence_model to obtain a latent
        embedding (e.g., hidden state or embedding_cls).
        - Tokenize and pass the text input through the provided tokenizer and text_model to
        obtain text representations.
        - Align text representations to the sequence latent space via a FeatureAligner.
        - Fuse the sequence and text representations (currently by concatenating the sequence
        and aligned text embeddings, using the cls_token pathway for classification).
        - Pass the fused representation to an OutputHead to produce logits for the specified task.

    The supported sequence_model types are determined by class name prefixes, i.e., 
    RNN, LSTM, GRU, STaRFormer.

    The text_model must be a Roberta-like model (RobertaModel) and the tokenizer must be
    provided to convert raw text into model-ready inputs.

    Attributes:
        sequence_model (nn.Module): The sequence-processing module.
        tokenizer (AutoTokenizer): Tokenizer for text input.
        text_model (nn.Module): Text encoding model (Roberta-like).
        _align_features (FeatureAligner): Aligns text features to the sequence latent space.
        _task (str): Target task for the OutputHead.
        _cls_method (str): Prediction method (e.g., 'cls_token').
        output_head (OutputHead): Learnable head mapping fused features to logits.
    """
    def __init__(self, 
                 # sequence model
                 sequence_model: nn.Module=None,
                 # text model
                 tokenizer: AutoTokenizer=None,
                 text_model: nn.Module=None,
                 # feature 
                 kernel_size: int=3,
                 activation_aligner: Union[str, Callable[[Tensor], Tensor]] = nn.ReLU(),
                 # output head 
                 task: Literal['classification', 'regression', 'forecasting']='classification',
                 d_out: int=1, 
                 d_hidden: int=None, 
                 activation_output_head: Union[str, Callable[[Tensor], Tensor]] = nn.ReLU(),
                 reduced: bool=True, 
                 cls_method: Literal['cls_token', 'regr_token', 'autoregressive', 'elementwise']='autoregressive',
                 # cls synonym for prediction method
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        Initializes the Sequence Text Dual Model 

        Args:
            sequence_model (nn.Module): A PyTorch module that processes sequences and yields
                latent representations for fusion. Must be an instance of nn.Module.
            tokenizer (AutoTokenizer): Tokenizer used to convert raw text strings into input
                tensors for the text_model.
            text_model (nn.Module): Text encoding model (must be a RobertaModel in this setup).
            kernel_size (int): Kernel size used by the FeatureAligner to align text features
                to the sequence latent dimension.
            activation_aligner (Union[str, Callable[[Tensor], Tensor]]): Activation applied
                inside the FeatureAligner after alignment.
            task (Literal['classification', 'regression', 'forecasting']): The target task for the
                OutputHead. Determines how logits are interpreted.
            d_out (int): Output dimensionality of the final head (e.g., number of classes
                for classification).
            d_hidden (int, optional): Hidden dimensionality for internal representations within
                the OutputHead. If None, a sensible default based on d_model will be used.
            activation_output_head (Union[str, Callable[[Tensor], Tensor]]): Activation applied
                to the final outputs from the OutputHead.
            reduced (bool): Whether to use a reduced form in the OutputHead (implementation-specific).
            cls_method (Literal['cls_token', 'regr_token', 'autoregressive', 'elementwise']): Method
                used by the OutputHead or prediction strategy when combining modalities. Currently
                supports 'cls_token' for classification fusion.
            *args, **kwargs: Additional positional and keyword arguments passed to the base class
                constructor.
        """
        assert sequence_model is not None and isinstance(sequence_model, nn.Module)
        # currently accepted models 
        assert sequence_model.__class__.__name__.startswith('RNN') or \
        sequence_model.__class__.__name__.startswith('LSTM') or \
        sequence_model.__class__.__name__.startswith('GRU') or \
        sequence_model.__class__.__name__.startswith('STaRFormer'), f'{sequence_model.__class__.__name__} not accepted!'
        self.sequence_model = sequence_model
        
        # tokenizer
        assert tokenizer is not None, f'tokenizer must be given!'
        self.tokenizer = tokenizer
        # text model
        assert text_model is not None, f'text_model must be given!'
        assert text_model.__class__.__name__ == 'RobertaModel', f'{text_model.__class__.__name__} currently not accepted'
        self.text_model = text_model

        # get hidden dimension
        if hasattr(self.sequence_model, '_hidden_size'):
            d_hidden = sequence_model._hidden_size
        elif hasattr(self.sequence_model, '_d_model'):
            d_hidden = sequence_model._d_model
        assert d_hidden is not None, f'd_hidden cannot be None'

        self._align_features = FeatureAligner(input_size=int(self.text_model.config.hidden_size),
                                              output_size=d_hidden,
                                              kernel_size=kernel_size,
                                              activation=activation_aligner)
        
        # output head
        self._task = task
        self._cls_method = cls_method

        if hasattr(self.sequence_model, '_d_model'):
            d_model = sequence_model._d_model
        elif hasattr(self.sequence_model, '_input_size'):
            d_model = sequence_model._input_size

        self.output_head = OutputHead(task=task, d_model=int(2*d_model), d_out=d_out,
                                      d_hidden=d_hidden, activation=activation_output_head, reduced=reduced, 
                                      method=cls_method)


    def forward(self, x: Tensor, text: str | List[str], N: Tensor, batch_size: int, **kwargs):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor for the sequence_model. Shape depends on the underlying
                sequence model (e.g., [N, B, D] for many RNN-like backends).
            text (Union[str, List[str]]): Raw text input(s) to be tokenized by the tokenizer.
                Can be a single string or a list of strings.
            N (Tensor): Optional sequence length or related information required by certain
                sequence models (e.g., STaRFormer). Passed through to downstream components as needed.
            batch_size (int, optional): Batch size. Required for RNN/LSTM/GRU backends. Must be provided
                when using recurrent models.
            kwargs: Additional keyword arguments forwarded to the underlying sequence_model. May
                include model-specific flags such as mode, padding_mask, etc.

        Returns:
            dict: A dictionary containing at least:
                - 'logits' (Tensor): The final predictions produced by the OutputHead.
                - Other keys produced by the underlying sequence_model (e.g., 'hidden_state',
                'embedding_cls', 'features'), preserved from the model outputs.
        """
        if self.sequence_model.__class__.__name__.startswith('RNN') or \
            self.sequence_model.__class__.__name__.startswith('LSTM') or \
            self.sequence_model.__class__.__name__.startswith('GRU'):
            assert kwargs.get('batch_size', None) is not None, f'batch_size was not given!'
            outputs = self.sequence_model(x, kwargs['batch_size'])
            latent_embedding = outputs['hidden_state'] # [N, B, D]
        
        elif self.sequence_model.__class__.__name__.startswith('STaRFormer'):
            #assert kwargs.get('padding_mask', None) is not None, f'padding_mask not given!'
            #assert isinstance(kwargs.get('padding_mask'), Tensor), f"padding_mask must be a Tensor and not {type(kwargs.get('padding_mask'))}"
            assert kwargs.get('mode', None) is not None, f'mode not given!'
            assert kwargs.get('mode') in ['train', 'val', 'test'], f"mode has to be one of the following ['train', 'val', 'test'], not {kwargs.get('mode')}"
            assert kwargs.get('dataset', None) is not None, f'dataset is not given!'
            assert isinstance(kwargs.get('dataset'), str) is not None, f"dataset must be a str, not {type(kwargs.get('dataset'))}!"
            outputs = self.sequence_model(x, N, **kwargs)
            latent_embedding = outputs['embedding_cls'] # [N, B, D]
    
        else:
            raise NotImplementedError
        
        text_inputs = self.tokenizer(text, return_tensors='pt', padding=True, 
                                     truncation=True, max_length=512)
        text_inputs = {k: v.to(latent_embedding.device) for k, v, in text_inputs.items()}
        text_outputs = self.text_model(**text_inputs) # [B, text_length, F] (F for RoBERTa = 768)
        #text_latent_embedding = text_outputs[:, 0, :] # select token 
        text_latent_embedding_aligned = self._align_features(text_outputs.last_hidden_state) # [B, text_length, F] --> [B, text_length, D]

        if self._cls_method == 'cls_token':
            # select token 
            sequence_outputs_cls_token = latent_embedding[0, ...] # [B, D]
            text_latent_embedding_aligned_cls_token = text_latent_embedding_aligned[:, 0, :] # [B, D]
            latent_embedding_cls_token = torch.concat([
                sequence_outputs_cls_token, text_latent_embedding_aligned_cls_token
            ], dim=1) # [B, 2*D]
        else:
            raise NotImplementedError(f'Only cls_token based classification is currently implemented!')
        
        # output head 
        if self._task == OutputHeadOptions.classification:
            logits = self.output_head(x=latent_embedding_cls_token, N=N, batch_size=batch_size)
        else:
            raise NotImplementedError
        outputs['logits'] = logits
        
        return outputs
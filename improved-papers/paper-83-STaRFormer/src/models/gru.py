import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from typing import Dict


__all__ = ['GRUNet']


class GRUNet(nn.Module):
    """A simple GRU-based neural network module.

    This module wraps a PyTorch GRU and exposes a lightweight interface
    for initializing and running a GRU over input sequences.

    The class is designed to be instantiated with standard GRU parameters
    (input size, hidden size, number of layers, etc.) and to produce a
    representative hidden state for the input sequence, suitable for
    downstream tasks such as sequence classification or encoding.

    Attributes:
        _num_layers (int): Number of recurrent layers in the GRU.
        _input_size (int): Dimensionality of the input features.
        _hidden_size (int): Dimensionality of the GRU hidden state per layer.
        _batch_first (bool): If True, inputs are provided as (batch, seq, feature).
        _hidden_activation (torch.nn.Module): Non-linearity applied to the final
            hidden state before returning it (default: ReLU).
        rnn (nn.GRU): The underlying GRU module.
    """
    def __init__(self, 
                 input_size: int,
                 hidden_size: int, # num of cell states
                 num_layers: int=1,
                 bias: bool=True,
                 batch_first: bool=False,
                 dropout: float=0.0,
                 bidirectional: bool=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        Initializes the GRUNet module with the given GRU parameters.

        Args:
            input_size (int): Number of expected features in the input.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int, optional): Number of recurrent layers. Default is 1.
            bias (bool, optional): If False, the GRU layer does not use bias weights. Default is True.
            batch_first (bool, optional): If True, then the input and output tensors are provided as (batch, seq, feature). Default is False.
            dropout (float, optional): If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer. Default is 0.0.
            bidirectional (bool, optional): If True, becomes a bidirectional GRU. Default is False.
            *args: Additional positional arguments for the base nn.Module.
            **kwargs: Additional keyword arguments for the base nn.Module.
        """
        self._num_layers = num_layers
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._batch_first = batch_first
        self._hidden_activation = nn.ReLU()

        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                          bias=bias, batch_first=batch_first, 
                          dropout=dropout, bidirectional=bidirectional)
    
    def forward(self, x: Tensor | PackedSequence, batch_size: int=None, **kwargs) -> Dict[str, Tensor]:
        """Forward pass through GRU.

        Args:
            x (Tensor or PackedSequence): Input sequence tensor or packed sequence.
                If ``batch_first`` was set to True during initialization, ``x`` should
                have shape (batch, seq_len, input_size); otherwise, (seq_len, batch, input_size).
            batch_size (int, optional): Explicit batch size to use when constructing the
                initial hidden state. If not provided, the size will be inferred from ``x``.
            **kwargs: Additional keyword arguments are accepted for forward compatibility
                but are not used in the current implementation.

        Returns:
            dict: A dictionary containing the processed outputs:
                - 'hidden_state' (Tensor): The final hidden state after processing the
                  entire input sequence. This value is passed through a ReLU activation
                  to introduce non-linearity.

        Notes - RNN Hidden state versus RNN output:
            Hidden State:
              - Represents the final state of the RNN after processing the entire sequence.
              - Used in tasks requiring a summary representation of the entire sequence.
              - Commonly applied in classification tasks where the entire sequence is classified based on the final hidden state.
            
            RNN Output:
              - Contains the output at each time step of the sequence.
              - Useful for tasks requiring information at each time step.
              - Applied in sequence-to-sequence tasks or when classifying each element in the sequence.
        """
        #
        if batch_size is None:
            batch_size = x.data.size(0) if self._batch_first else x.data.size(1)
        
        device = x.data.device
        dtype = x.data.dtype if x.data.dtype == torch.float32 else torch.float32
        if torch.backends.mps.is_available() and x.data.dtype == torch.double:
            x = x.to(dtype)
            self.rnn = self.rnn.to(device)
        dtype = x.data.dtype
        device = x.data.device
        # init hidden state
        self.hidden = torch.zeros(self._num_layers, batch_size, self._hidden_size, dtype=dtype).to(device)
        # rnn 
        out, hn = self.rnn(x, self.hidden)
        hn = self._hidden_activation(hn[-1])
        return {'hidden_state': hn}
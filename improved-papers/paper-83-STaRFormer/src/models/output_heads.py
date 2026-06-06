import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from typing import Union, Callable, Literal

from src.utils import TaskOptions
from src.utils.parameters import OptionsBaseClass

__all__ = ["OutputHead"]


OutputHeadOptions = TaskOptions

class ClassificationOptions(OptionsBaseClass):
    """Options for classification head methods.

    Attributes:
        cls_token (str): Option for classification using a special token ('cls_token').
        autoregressive (str): Option for autoregressive classification ('autoregressive').
        elementwise (str): Option for elementwise classification along the sequence ('elementwise').
    """
    cls_token: str='cls_token'
    autoregressive: str='autoregressive'
    elementwise: str='elementwise' # per element in sequence


class RegressionOptions(OptionsBaseClass):
    """Options for regression head methods.

    Attributes:
        regr_token (str): Option for regression using a special token ('regr_token').
        autoregressive (str): Option for autoregressive regression ('autoregressive').
        elementwise (str): Option for elementwise regression along the sequence ('elementwise').
    """
    regr_token: str='regr_token'
    autoregressive: str='autoregressive'
    elementwise: str='elementwise' # per element in sequence    


class OutputHead(nn.Module):
    """A modular output head for classification, regression, or forecasting tasks.

    This class serves as a wrapper to dynamically instantiate and use the appropriate
    output head (e.g., classification or regression) based on the specified task.

    Attributes:
        net (nn.Module): The instantiated output head module (e.g., ClassificationHead or RegressionHead)
            corresponding to the given task and configuration.
        _options (OutputHeadOptions): Helper object containing supported task options.

    """
    def __init__(self, 
                 task: Literal['classification', 'regression', 'forecasting']='classification',
                 d_model: int=None, 
                 d_out: int=1, 
                 d_hidden: int=None, 
                 activation: Union[str, Callable[[Tensor], Tensor]] = nn.ReLU(),
                 reduced: bool=True, 
                 method: Literal['cls_token', 'regr_token', 'autoregressive', 'elementwise']='autoregressive',
                 norm_dim: int=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initializes the OutputHead module for the specified task.

        Args:
            task (str, optional): The type of task ('classification', 'regression', or 'forecasting').
                Determines which head is used. Default is 'classification'.
            d_model (int, optional): The input feature dimension for the output head.
            d_out (int, optional): The output dimension (e.g., number of classes for classification).
                Default is 1.
            d_hidden (int, optional): The hidden dimension for the output head, if applicable.
            activation (str or callable, optional): Activation function to use in the output head.
                Default is nn.ReLU().
            reduced (bool, optional): Whether to apply reduction in the output head.
                Default is True.
            method (str, optional): The pooling or output method to use
                ('cls_token', 'regr_token', 'autoregressive', or 'elementwise').
                Default is 'autoregressive'.
            norm_dim (int, optional): Normalization dimension, used for anomaly detection.
            *args: Additional positional arguments for the base nn.Module.
            **kwargs: Additional keyword arguments for the base nn.Module.

        Raises:
            NotImplementedError: If the 'forecasting' task is specified.
            ValueError: If an unsupported task is provided.
            AssertionError: If required arguments are missing for the selected task.
        """
        self._options = OutputHeadOptions()
        if task == self._options.classification:
            assert d_model is not None and isinstance(d_model, int)
            assert d_out is not None and isinstance(d_out, int)
            self.net = ClassificationHead(d_model=d_model, d_out=d_out, d_hidden=d_hidden,
                                          activation=activation, reduced=reduced, cls_method=method)
        elif task == self._options.regression:
            # Implement regression head initialization
            assert d_model is not None and isinstance(d_model, int)
            assert d_out is not None and isinstance(d_out, int)
            self.net = RegressionHead(d_model=d_model, d_out=d_out, d_hidden=d_hidden,
                                      activation=activation, reduced=reduced, regr_method=method
            )
        elif task == self._options.forecasting:
            # Implement forecasting head initialization
            raise NotImplementedError
        elif task == self._options.anomaly_detection:
            assert method == 'elementwise'
            self.net = ClassificationHead(d_model=d_model, d_out=d_out, d_hidden=d_hidden,
                                          activation=activation, reduced=reduced, cls_method=method,
                                          norm_dim=norm_dim)
            
        else:
            raise ValueError(f"Unsupported task: {task}")

    
    def forward(self, x: Union[Tensor, PackedSequence], N: Tensor=None, batch_size: int=None) -> Tensor:
        """Forward pass through the output head.

        Args:
            x (Tensor or PackedSequence): The input features to the output head.
            N (Tensor, optional): Optional parameter, e.g., sequence lengths or indices.
            batch_size (int, optional): Optional batch size, if required by the output head.

        Returns:
            Tensor: The output tensor from the corresponding output head.
        """
        return self.net(x=x, N=N, batch_size=batch_size)


class ClassificationHead(nn.Module):
    """A flexible classification head for sequence and elementwise classification tasks.

    Supports multiple pooling and output strategies, including classification by a special
    token, autoregressive pooling, and elementwise classification. The output is always logits.

    Attributes:
        _cls_options (ClassificationOptions): Helper object containing supported classification options.
        _cls_method (str): Selected classification method, e.g., 'cls_token', 'autoregressive', or 'elementwise'.
        net (nn.Sequential): The sequential neural network module used to compute logits.
    """
    def __init__(self, d_model: int, d_out: int=1, d_hidden: int=None, 
        activation: Union[str, Callable[[Tensor], Tensor]] = nn.ReLU(),
        reduced: bool=True, cls_method: Literal['cls_token', 'autoregressive', 'elementwise']='autoregressive',
        norm_dim: int=None,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """Initializes the ClassificationHead module.

        Args:
            d_model (int): Input feature dimension.
            d_out (int, optional): Output dimension (number of classes). Default is 1.
            d_hidden (int, optional): Hidden dimension for the intermediate layer. If None and required,
                it is set to half of d_model.
            activation (str or callable, optional): Activation function for the hidden layer. Default is nn.ReLU().
            reduced (bool, optional): If True, uses a single linear layer for reduced mode.
                If False, uses a deeper network with non-linearity and batch normalization. Default is True.
            cls_method (str, optional): Classification method to use:
                - 'cls_token': Use the special classification token as input.
                - 'autoregressive': Use the last element of each sequence.
                - 'elementwise': Perform elementwise classification along the sequence.
                Default is 'autoregressive'.
            norm_dim (int, optional): Normalization dimension, used with 'elementwise' method.
            *args: Additional positional arguments for the base nn.Module.
            **kwargs: Additional keyword arguments for the base nn.Module.
        """
        self._cls_options = ClassificationOptions()
        self._cls_method = cls_method

        # always return logits and not probas
        if reduced:
            self.net = nn.Sequential(
                nn.Linear(d_model, d_out),
                #nn.Sigmoid()
            )
        else:
            if d_hidden is None:
                d_hidden = int(d_model / 2)
            
            if cls_method == self._cls_options.elementwise:
                self.net = nn.Sequential(
                    nn.Linear(d_model, d_hidden),
                    activation,
                    nn.LazyBatchNorm1d(),
                    nn.Linear(d_hidden, d_out),
                )
            else:
                self.net = nn.Sequential(
                    nn.Linear(d_model, d_hidden),
                    activation,
                    nn.BatchNorm1d(d_hidden),
                    nn.Linear(d_hidden, d_out),
                )
    
    def forward(self, x: Union[Tensor, PackedSequence], N: Tensor=None, batch_size: int=None) -> Tensor:
        """
        Forward pass facilitating different classification logics.

        Args:
            x (Tensor or PackedSequence): Input tensor of shape depending on context,
                e.g., (seq_len, batch_size, d_model) or (batch_size, d_model).
            N (Tensor, optional): Tensor containing sequence lengths or indices for pooling,
                required for 'autoregressive' and 'elementwise' methods.
            batch_size (int, optional): Batch size, required for 'autoregressive' and 'elementwise' methods.

        Returns:
            Tensor: Output logits, shape depends on classification method and input.
        """

        if self._cls_method == self._cls_options.cls_token:
            return self._forward_cls_token(x=x)
        elif self._cls_method == self._cls_options.autoregressive:
            return self._forward_autoregessive(x=x, N=N, batch_size=batch_size)
        elif self._cls_method == self._cls_options.elementwise:
            assert N is not None, f'N cannot be `None`'
            return self._forward_elementwise(x=x, N=N, batch_size=batch_size)
        else:
            raise ValueError(f'{self._cls_method} is not known, possible options are {self._cls_options.cls_token}')

    def _forward_cls_token(self, x: Union[Tensor, PackedSequence]) -> Tensor:
        """Forward pass for token based classification.

        Args:
            x (Tensor or PackedSequence): Input tensor, first sequence position is treated as the cls token.

        Returns:
            Tensor: Output logits for the cls token.
        """

        if len(x.size()) >= 3:
            x = x[0, ...]
        else:
            x = x
        return self.net(x)

    def _forward_autoregessive(self, x: Union[Tensor, PackedSequence], N: Tensor=None, batch_size: int=None) -> Tensor:
        """Forward pass for sequence-level classification using autoregressive pooling (last element of each sequence).

        Args:
            x (Tensor or PackedSequence): Input tensor, shape (seq_len, batch_size, d_model) or (batch_size, d_model).
            N (Tensor, optional): Tensor indicating the sequence lengths.
            batch_size (int, optional): Batch size.

        Returns:
            Tensor: Output logits for the last element in each sequence.
        
        Notes:
            - N selects the last element of the sequence
            - the batch size selects the correct element in the batch matching the seq_len selected
        """
        # x [N, batch_size, d_model]
        # N selects the last element of the sequence, 
        # and the batch size select the correct element in the batch 
        # matching the seq_len selected
        #x = x[N.flatten()-1, [i for i in range(batch_size)],...] # [batch_size, d_model]
        if len(x.size()) == 3:
            assert N is not None, f'N cannot be `None`'
            assert batch_size is not None, f'batch_size cannot be `None`'
            x = x[N.flatten()-1, list(range(batch_size)),...] # [batch_size, d_model] (ensure correct last element is selected, even for padded sequences)
        elif len(x.size()) == 2: # for rnns (use last hidden state)
            x = x # [batch_size, d_model]
        return self.net(x)

    def _forward_elementwise(self, x: Union[Tensor, PackedSequence], N: Tensor, batch_size: int=None, **kwargs) -> Tensor:
        """Forward pass for sequence element-level classification along the sequence.

        Args:
            x (Tensor or PackedSequence): Input tensor, usually (seq_len, batch_size, d_model).
            N (Tensor): Sequence lengths or mask for valid positions.
            batch_size (int, optional): Batch size.

        Returns:
            Tensor: Output logits for each element in each sequence.
        """
        if batch_size != x.shape[0]:
            # [seq_len, batch_size, d_out] --> [batch_size, seq_len, d_out] 
            x = x.permute(1,0,2)

        logits = self.net(x) # shape [batch_size, seq_len, d_out] 

        #if not check_tensor_elements_are_equal_to_each_other(N):
        #    logits = torch.concat(
        #        # logit [seq_len, d_out]
        #        [logit[:N[idx].item(), :] for idx, logit in enumerate(logits)],
        #        dim=1
        #    )
        #    print('OUtputhead 2', logits.shape)
        return logits


def check_tensor_elements_are_equal_to_each_other(tensor):
    repeated = tensor[0].repeat(len(tensor))
    return torch.all(repeated == tensor)


class RegressionHead(nn.Module):
    """A flexible regression head for sequence and elementwise regression tasks.

    Supports multiple pooling and output strategies, including regression by a special
    token, autoregressive pooling, elementwise regression, and global pooling. The output
    is always raw regression values (logits).

    Attributes:
        _regr_options (RegressionOptions): Helper object containing supported regression options.
        _regr_method (str): Selected regression method, e.g., 'regr_token', 'autoregressive', 'elementwise', or 'pooling'.
        net (nn.Sequential): The sequential neural network module used to compute regression outputs.
    """
    def __init__(self, d_model: int, d_out: int=1, d_hidden: int=None, 
        activation: Union[str, Callable[[Tensor], Tensor]] = nn.ReLU(),
        reduced: bool=True, regr_method: Literal['regr_token', 'autoregressive', 'elementwise', 'pooling']='autoregressive',
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """Initializes the RegressionHead module.

        Args:
            d_model (int): Input feature dimension.
            d_out (int, optional): Output dimension (number of regression targets). Default is 1.
            d_hidden (int, optional): Hidden dimension for intermediate layers. If None and required,
                it is set to half of d_model.
            activation (str or callable, optional): Activation function for the hidden layers. Default is nn.ReLU().
            reduced (bool, optional): If True, uses a single linear layer for reduced mode. If False, uses a deeper
                network with non-linearity and batch normalization. Default is True.
            regr_method (str, optional): Regression method to use:
                - 'regr_token': Use a special regression token as input.
                - 'autoregressive': Use the last element of each sequence.
                - 'elementwise': Perform elementwise regression along the sequence.
                - 'pooling': Perform global pooling (mean) before regression.
                Default is 'autoregressive'.
            *args: Additional positional arguments for the base nn.Module.
            **kwargs: Additional keyword arguments for the base nn.Module.

        """
        self._regr_options = RegressionOptions()
        self._regr_method = regr_method

        # always return logits and not probas
        if reduced:
            self.net = nn.Sequential(
                nn.Linear(d_model, d_out),
            )
        else:
            if d_hidden is None:
                d_hidden = int(d_model / 2)
            
            self.net = nn.Sequential(
                nn.Linear(d_model, d_hidden),
                activation,
                nn.BatchNorm1d(d_hidden),
                nn.Linear(d_hidden, d_hidden),
                activation,
                nn.BatchNorm1d(d_hidden),
                nn.Linear(d_hidden, d_out),
            )
    
    def forward(self, x: Union[Tensor, PackedSequence], N: Tensor=None, batch_size: int=None) -> Tensor:
        """Forward pass facilitating different regssion logics.

        Args:
            x (Tensor or PackedSequence): Input tensor of shape depending on context,
                e.g., (seq_len, batch_size, d_model) or (batch_size, d_model).
            N (Tensor, optional): Tensor containing sequence lengths or indices for pooling,
                required for 'autoregressive' and 'elementwise' methods.
            batch_size (int, optional): Batch size, required for 'autoregressive' and 'elementwise' methods.

        Returns:
            Tensor: Output regression values (logits), shape depends on regression method and input.

        Raises:
            ValueError: If an unknown regression method is specified.
        """

        if self._regr_method == self._regr_options.regr_token:
            return self._forward_regr_token(x=x)
        elif self._regr_method == self._regr_options.autoregressive:
            return self._forward_autoregessive(x=x, N=N, batch_size=batch_size)
        elif self._regr_method == self._regr_options.elementwise:
            assert N is not None, f'N cannot be `None`'
            return self._forward_elementwise(x=x, N=N)
        elif self._regr_method == 'pooling':
            return self._forward_pooling(x=x)
        else:
            raise ValueError(f'{self._regr_method} is not known, possible options are {self._cls_options.cls_token}')

    def _forward_pooling(self, x: Union[Tensor, PackedSequence]) -> Tensor:
        """Perform global mean pooling over the sequence dimension before regression.

        Args:
            x (Tensor or PackedSequence): Input tensor of shape (seq_len, batch_size, d_model).

        Returns:
            Tensor: Output regression values after pooling, shape (batch_size, d_out).
        """
        # perform global pooling
        # [N, B, D] --> [B, D]
        # average pooling
        x_pooled = x.mean(dim=0)
        #x_pooled = x.sum(dim=0)
        return self.net(x_pooled)
    
    def _forward_regr_token(self, x: Union[Tensor, PackedSequence]) -> Tensor:
        """Regression using a special `regression' token.

        Args:
            x (Tensor or PackedSequence): Input tensor, first sequence position is treated as the regression token.

        Returns:
            Tensor: Output regression values for the regression token.
        """
        if len(x.size()) >= 3:
            x = x[0, ...]
        else:
            x = x
        return self.net(x)

    def _forward_autoregessive(self, x: Union[Tensor, PackedSequence], N: Tensor=None, batch_size: int=None) -> Tensor:
        """Regression using the last element of each sequence (autoregressive mode).

        Args:
            x (Tensor or PackedSequence): Input tensor, shape (seq_len, batch_size, d_model) or (batch_size, d_model).
            N (Tensor, optional): Sequence lengths for each batch.
            batch_size (int, optional): Batch size.

        Returns:
            Tensor: Output regression values for the last element in each sequence.
        
        Notes:
            - N selects the last element of the sequence
            - the batch size selects the correct element in the batch matching the seq_len selected
        """
        if len(x.size()) == 3:
            assert N is not None, f'N cannot be `None`'
            assert batch_size is not None, f'batch_size cannot be `None`'
            x = x[N.flatten()-1, list(range(batch_size)),...] # [batch_size, d_model] (ensure correct last element is selected, even for padded sequences)
        elif len(x.size()) == 2: # for rnns (use last hidden state)
            x = x # [batch_size, d_model]
        return self.net(x)

    def _forward_elementwise(self, x: Union[Tensor, PackedSequence], N: Tensor, **kwargs) -> Tensor:
        """Elementwise regression along the sequence.

        Args:
            x (Tensor or PackedSequence): Input tensor, usually (seq_len, batch_size, d_model).
            N (Tensor): Sequence lengths or mask for valid positions.

        Returns:
            Tensor: Output regression values for each element in each sequence, concatenated along the batch dimension.
        """
        # check this implementation'
        # not production ready
        logits = self.net(x) # shape [batch_size, seq_len, d_out] 
        logits = [logit[:N[idx].item(), :] for idx, logit in enumerate(logits)]    
        return torch.concat(logits, dim=0)

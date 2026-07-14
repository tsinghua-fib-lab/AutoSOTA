"""
This file contains:

(1) Class definition for 'VanillaNN',
an extension of 'BaseModule' that
implements a vanilla neural network 
(aka multi-layer perceptron, or MLP)
programmatically with the desired width
and depth of layers.

(2) Function 'load_vanilla_nn_with_accelerate',
which re-loads a VanillaNN following HuggingFace's
'Accelerate' library protocols (note that the
train_fn.train_model function uses Accelerate).
"""

import models.base_module as bm
import models.nn_utilities as nnu

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import (
    Dataset, 
    DataLoader
)
from accelerate import Accelerator
from typing import (
    Tuple,
    List,
    Dict,
    Optional,
    Callable,
    Any
)

class VanillaNN(bm.BaseModule):
    """
    Extension of BaseModule that programmatically
    constructs a vanilla neural network of arbitrary
    size.
    
    __init__ args:
         input_dim: input feature dimension to the first layer.
         output_dim: final output dimension of the network
             predictions.
         hidden_dims_list: list of integers specifying the
             dimensions of the hidden linear layers.
         bias_in_hidden_layers: bool whether to include a 
             bias term in the linear layers.
         nonlin_fn: nonlinear activation function to apply
             after the linear layers (excluding final output).
         nonlin_fn_kwargs: kwargs to pass to the nonlin_fn 
             torch module.
         wt_init_fn: weight initialization function for 
             model weights.
         wt_init_fn_kwargs: kwargs to pass wt_init_fn.
         batch_normalization_kwargs: kwargs to pass to pytorch's
             batch normalization function.
         use_dropout: bool whether to use dropout in between
             hidden layers.
         dropout_p: the probability of zeroing out a perceptron,
             if using dropout.
         base_module_kwargs: kwargs to pass to the parent
             BaseModule class.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims_list: List[int],
        bias_in_hidden_layers: bool = True,
        nonlin_fn: Callable = nn.ReLU,
        nonlin_fn_kwargs: Dict[str, Any] = {},
        wt_init_fn: Callable = nn.init.kaiming_uniform_,
        wt_init_fn_kwargs: Dict[str, Any] = {'nonlinearity': 'relu'},
        use_batch_normalization: bool = False,
        batch_normalization_kwargs: Dict[str, Any] = {'affine': True},
        use_dropout: bool = False,
        dropout_p: float = 0.5,
        base_module_kwargs: Dict[str, Any] = {}
    ):
        super(VanillaNN, self).__init__(**base_module_kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims_list = hidden_dims_list
        self.bias_in_hidden_layers = bias_in_hidden_layers
        self.nonlin_fn = nonlin_fn
        self.nonlin_fn_kwargs = nonlin_fn_kwargs
        self.fc_out_layer_activ_fn = None
        self.wt_init_fn = wt_init_fn
        self.wt_init_fn_kwargs = wt_init_fn_kwargs
        self.use_dropout = use_dropout
        self.use_batch_normalization = use_batch_normalization
        self.dropout_p = dropout_p
        
        self._build_fc_network()


    def _build_fc_network(self):

        # set up fully-connected layer modules
        self.fc_lin_fns, self.fc_nonlin_fn, self.fc_lin_out = nnu.build_ffnn(
            input_dim=self.input_dim,
            output_dim=self.output_dim, 
            hidden_dims_list=self.hidden_dims_list, 
            bias_in_hidden_layers=self.bias_in_hidden_layers,
            nonlin_fn=self.nonlin_fn,
            nonlin_fn_kwargs=self.nonlin_fn_kwargs
         )

        # initialize layer weights
        for lin_fn in self.fc_lin_fns:
            self._init_linear_layer_weights(lin_fn)
        self._init_linear_layer_weights(self.fc_lin_out)

        # optional: set batch normalization modules
        if self.use_batch_normalization:
            self.batch_norms = nn.ModuleList(
                nn.BatchNorm1d(num_features) \
                for num_features in self.hidden_dims_list
            )
            
        # optional: set dropout module
        if self.use_dropout:
            # dropout module must be attribute of model
            # for model.eval() to turn it off during inference!
            self.dropout = nn.Dropout(self.dropout_p)
        
    
    def _init_linear_layer_weights(self, m):
        # random weights initialization for linear layers
        if isinstance(m, nn.Linear):
            self.wt_init_fn(
                m.weight.data, 
                **self.wt_init_fn_kwargs
            )
            if self.bias_in_hidden_layers:
                nn.init.zeros_(m.bias.data)

    
    def _fc_forward(
        self, 
        x: torch.Tensor | Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # print('vnn device:', self.get_device())
        if isinstance(x, dict):
            x = x['x']
        # x = self._prep_data(input)
        # print(x.get_device())

        # this (silently) crashes it all on Apple silicon/MPS!
        # if self.device is not None:
        #     x = x.to(self.device)

        # proceed through layers: linear -> activation [-> dropout]
        for i, lin_fn in enumerate(self.fc_lin_fns):
            x = lin_fn(x)
            x = self.fc_nonlin_fn(x)
            if self.use_batch_normalization:
                x = self.batch_norms[i](x)
            if self.use_dropout:
                x = self.dropout(x)

        # final layer (has unique, non-ReLU activation, or no activation)
        x = self.fc_lin_out(x)
        if self.fc_out_layer_activ_fn is not None:
            x = self.fc_out_layer_activ_fn(x)

        # contain model output in dict
        model_output_dict = {
            'preds': x
        }
        return model_output_dict
        
    
    def forward(
        self, 
        x: torch.Tensor | Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self._fc_forward(x)

    
    


def load_vanilla_nn_with_accelerate(
    args, 
    task,
    loss_fn,
    target_name,
    input_dim,
    output_dim,
    dropout_p,
    model_path, 
    dataloaders,
    sets
) -> Tuple:
    """
    Loads a (trained) VanillaNN state using accelerate.
    Returns the accelerate object, vae, and dataloaders
    for the specified sets (train/valid/test).
    Must get 'args' class with needed values.
    """
    # init new model
    model = VanillaNN(
        task=task,
        input_dim=input_dim,
        hidden_dims_list=args.MLP_DIM_ARR,
        wt_init_fn=nn.init.kaiming_uniform_,
        nonlin_fn=nn.ReLU,
        nonlin_fn_kwargs={},
        loss_fn=loss_fn, 
        loss_fn_kwargs=None,
        use_dropout=True,
        dropout_p=dropout_p,
        output_dim=output_dim,
        target_name=target_name
    )
    
    # must be the same optimizer type used to train the vae!
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.LEARN_RATE,
        betas=args.ADAM_BETAS
    )

    # a saved/trained model is wrapped by accelerate, so
    # we have to use accelerate's `load_state` to access
    accelerator = Accelerator(cpu=args.ON_CPU)
    (model, optimizer) = accelerator.prepare(model, optimizer)
    dataloaders = {
        set: accelerator.prepare(dataloaders[set]) \
        for set in sets
    }
    epoch_ctr = nnu.EpochCounter(0, args.MAIN_METRIC)
    accelerator.register_for_checkpointing(epoch_ctr)
    
    # finally, load trained model from save path and return objects
    accelerator.load_state(model_path)
    return (accelerator, model, dataloaders)
    

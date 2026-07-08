import torch.nn as nn
import torch
import json
import numpy as np
from torch.nn import ReLU, ELU, LeakyReLU
# Try newer torch.func API (PyTorch 2.0+), fall back to functorch for older versions
try:
    from torch.func import vmap, jacrev
except ImportError:
    from functorch import vmap, jacrev

from pathlib import Path
def setup_latent_network(args_architecture):
    """
    Initialize the latent projector network based on the provided architecture parameters.

    Args:
        args_architecture (str, Path, or dict): Path to a JSON file or a dictionary
            containing architecture parameters. If a file is provided, it should
            include an "architecture" key with the network settings 'latent_dim', 
            'intermediate_dim', 'depth', and optionally 'activation' and 'normalization'.
    """
    if isinstance(args_architecture, (str, Path)) and Path(args_architecture).is_file():
        with open(args_architecture, 'r') as f:
            args_architecture = json.load(f)
            args_architecture = args_architecture['architecture']

    if args_architecture.get('activation') is not None:
        if args_architecture['activation'] == 'elu':
            activation = ELU
        if args_architecture['activation'] == 'relu':
            activation = ReLU
        if args_architecture['activation'] == 'leaky_relu':
            activation = LeakyReLU
    else:
        activation = ELU
        print("ELU activation used as default")

    latent_projector = LatentProjector(args_architecture['latent_dim'],
                                       intermediate_dim=args_architecture['intermediate_dim'],
                                       num_layers=args_architecture['depth'],
                                       activation=activation,
                                       normalization=args_architecture.get('normalization', False)
                                       )
    return latent_projector

def _initialize(layers):
    """
    Initialize layers for ELU activations using Kaiming normal initialization.
    """
    for layer in layers:
        if layer.__class__.__name__ not in ["PReLU", "BatchNorm2d"]:
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias.data.zero_()


class LatentProjector(nn.Sequential):
    def __init__(self, in_out_dim, intermediate_dim=None, num_layers=4, activation=LeakyReLU, normalization =True):
        super().__init__()
        
        self.add_module(f'linear_0', nn.Linear(in_out_dim, intermediate_dim))
        if normalization:
            self.add_module(f'norm_{0}', nn.LayerNorm(intermediate_dim))

        self.add_module(f'activation_0', activation())
        
        for i in range(1, num_layers - 1):
            self.add_module(f'linear_{i}', nn.Linear(intermediate_dim, intermediate_dim))

            self.add_module(f'activation_{i}', activation())
        
        self.add_module(f'linear_{num_layers - 1}', nn.Linear(intermediate_dim, in_out_dim))
        
        #kaiming initialization for elu or relu activations
        _initialize(self.modules())

        self.in_out_dim = in_out_dim

    def latent_space_deviation(self, latent_code):
        return self(latent_code) - latent_code

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def device(self):
        return next(self.parameters()).device
    
    def phi(self, use_torch=True): #coord in batch x la_dim
        in_out_dim = self.in_out_dim
        latent_space_deviation = self.latent_space_deviation
        device = self.device()
        if use_torch:
            def _phi(coord):
                latent_coord = coord.reshape(-1,in_out_dim)
                return (latent_space_deviation(latent_coord).reshape(-1,in_out_dim)).T
        else:
            def _phi(coord):
                latent_coord = torch.tensor(coord.reshape(-1,in_out_dim), device = device, dtype = torch.float32, requires_grad = True)
                return np.transpose(latent_space_deviation(latent_coord).detach().cpu().numpy().reshape(-1,in_out_dim)) #returns in n xm 

        return _phi

    def dphi(self, use_torch=True): #coord in batch x la_dim
        in_out_dim = self.in_out_dim
        latent_space_deviation = self.latent_space_deviation
        device = self.device()
        if use_torch:
            def _dphi(coord):
                latent_coord = coord.reshape(-1,in_out_dim)
                def f(coord):
                    return latent_space_deviation(coord).reshape(in_out_dim)
                jac = vmap(jacrev(f))(latent_coord)
                return jac.swapaxes(0,1)
        else:
            def _dphi(coord):
                latent_coord = torch.tensor(coord.reshape(-1,in_out_dim), device = device, dtype = torch.float32, requires_grad = True)
                def f(coord):
                    return latent_space_deviation(coord).reshape(in_out_dim)
                jac = vmap(jacrev(f))(latent_coord).detach().cpu().numpy()
                return jac.swapaxes(0,1) #reshape for solver
        
        return _dphi
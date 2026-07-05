"""mlp.py: MLP Approximators compatible to TrainableMountainCarMRPWrapper class.
"""
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import create_mlp
from algorithms import rl_approximator_interfaces as rl_approximator


__author__ = "Hannes Unger"
__version__ = "0.3.0"
__email__ = "hannes.unger@fh-salzburg.ac.at"


INIT_WEIGHT = 1e-6


class SingleHiddenLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size, bias=True)
        )

    def forward(self, x):
        return self.layers(x)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def get_weights(self):
        ret = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                ret.append([layer.weight, layer.bias])
        return ret
    
    # Initialize weights and biases such that the output is approximately a constant value
    def initialize_to_constant(self, constant_value, only_bias=True):
        np.random.seed(0)
        with torch.no_grad():
            # Set all weights to zero (to ignore the inputs)
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    if only_bias:
                        layer.weight.fill_(np.random.uniform(0, INIT_WEIGHT))  # Set all weights to a small random value
                        if layer.out_features == 1:  # Output layer (last Linear)
                            layer.bias.fill_(constant_value)  # Set bias to constant value
                        else:
                            layer.bias.fill_(np.random.uniform(0, INIT_WEIGHT))  # Set all weights to a small random value
                    else:
                        layer.weight.fill_(np.random.uniform(0, INIT_WEIGHT))  # Set all weights to a small random value
                        layer.bias.fill_(constant_value)  # Set bias to constant value           


class SingleHiddenLayerMLPApproximator(rl_approximator.Approximator):
    def __init__(self, n_input=2, n_hidden_nodes=4, n_output=1, initialization='constant', flat_init_offset=0.2, params=None):
        self.model = SingleHiddenLayerMLP(n_input, n_hidden_nodes, n_output)

        if params is not None:
            self.model.load_state_dict(params)
        else:
            if initialization == 'flat':
                self.model.initialize_to_constant(flat_init_offset, only_bias=True)
            # if initialization == 'constant': # Not a good idea: If all neurons are initialized the same way, learning something different is effectively prohibited
            #     self.model.initialize_to_constant(1e-6, only_bias=False)
            # Neurons are initialized by default anyway: https://stackoverflow.com/questions/48529625/in-pytorch-how-are-layer-weights-and-biases-initialized-by-default

    def evaluate_point(self, x):
        return self.model(torch.tensor(x, dtype=torch.float32))

    def evaluate(self, *data):
        #return np.array([[self.evaluate_point([x,y]) for x in data_x] for y in data_y]) # for 2d data
        grid = np.meshgrid(*data, indexing='ij')
        points = np.stack(grid, axis=-1)
        return torch.from_numpy(np.apply_along_axis(self.evaluate_point, -1, points)) 


class MLPApproximator(rl_approximator.Approximator):
    def __init__(self, n_input=2, n_output=1, net_arch=[64, 64], initialization='constant', flat_init_offset=0.2, params=None):
        self.model = nn.Sequential(*create_mlp(input_dim=n_input, output_dim=n_output, net_arch=net_arch))

        if params is not None:
            self.model.load_state_dict(params)
        else:
            if initialization == 'flat':
                self.initialize_to_constant(target_value=flat_init_offset)

    # Initialize weights and biases such that the output is approximately a constant value
    def initialize_to_constant(self, target_value=0.0):
        """Initialize with tiny random weights."""
        for i, module in enumerate(self.model):
            if isinstance(module, nn.Linear):
                # Very small random weights
                nn.init.uniform_(module.weight, -INIT_WEIGHT, INIT_WEIGHT)
                
                if i < len(self.model) - 1:
                    nn.init.zeros_(module.bias)
                else:
                    nn.init.constant_(module.bias, target_value)

    def evaluate_point(self, x):
        return self.model(torch.tensor(x, dtype=torch.float32))[0]

    def flatten_innermost(self, data):
        if isinstance(data, list):
            if len(data) == 1 and isinstance(data[0], (int, float)):
                return data[0]
            return [self.flatten_innermost(item) for item in data]
        return data

    def evaluate(self, *data):
        grid = np.meshgrid(*data, indexing='ij')
        points = np.stack(grid, axis=-1)
        return torch.from_numpy(np.apply_along_axis(self.evaluate_point, -1, points))
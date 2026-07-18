'''
Taken from (Zhang et al. 2024): https://github.com/rsinghlab/scNODE
Description:
    Basic neural networks.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''

import torch.nn as nn
import torch
# ===========================================
# Dictionary of activation functions
ACT_FUNC_MAP = {
    "none": nn.Identity(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "softplus": nn.Softplus(),
    "relu": nn.ReLU(),
    "leakyrelu": nn.LeakyReLU(),
}

# ===========================================

class LinearNet(nn.Module):
    '''
    Fully-connected neural network.
    '''
    def __init__(self, input_dim, latent_size_list, output_dim, act_name):
        '''
        Initialize the neural network.
        :param input_dim (int): Input layer size.
        :param latent_size_list (None or list): Either None indicates no hidden layers or a list of integers
                                 representing size of every hidden layers.
        :param output_dim (int): Output layer size.
        :param act_name (str): The name of activation function, should be one of "none" (identity transformation),
                         "sigmoid", "tanh", "softplus", and "relu".
        '''
        super(LinearNet, self).__init__()
        layer_list = []
        if act_name not in ACT_FUNC_MAP:
            raise ValueError("The activation function should be one of {}.".format(ACT_FUNC_MAP.keys()))
        act_func = ACT_FUNC_MAP[act_name]
        if latent_size_list is not None:
            layer_list.extend([nn.Linear(input_dim, latent_size_list[0]), act_func, nn.LayerNorm(latent_size_list[0])])
            for i in range(len(latent_size_list) - 1):
                layer_list.extend([nn.Linear(latent_size_list[i], latent_size_list[i + 1]), act_func, nn.LayerNorm(latent_size_list[i + 1])])
            layer_list.extend([nn.Linear(latent_size_list[-1], output_dim), act_func])
        else:
            layer_list.extend([nn.Linear(input_dim, output_dim)])
        self.net = nn.Sequential(*layer_list)
        self.input_dim = input_dim
        self.output_dim = output_dim


    def forward(self, data):
        out = self.net(data)
        return out


    def forwardWithTime(self, t, data):
        '''
        This is equivalent to `forward` and is used for `differential equation solver.
        :param t: Placeholder.
        :param data (torch.FloatTensor): Neural network input matrix with the shape of (# cells, # genes).
        :return: Neural network output.
        '''
        out = self.net(data)
        return out


class LinearVAENet(nn.Module):
    '''
    Fully-connected neural network used for variational autoencoder (VAE) encoder.
    '''
    def __init__(self, input_dim, latent_size_list, output_dim, act_name):
        '''
        Initialize the neural network.
        :param input_dim (int): Input layer size.
        :param latent_size_list (None or list): Either None indicates no hidden layers or a list of integers
                                                representing size of every hidden layers.
        :param output_dim (int): Output layer size.
        :param act_name (str): The name of activation function, should be one of "none" (identity transformation),
                               "sigmoid", "tanh", "softplus", and "relu".
        '''
        super(LinearVAENet, self).__init__()
        layer_list = []
        if act_name not in ACT_FUNC_MAP:
            raise ValueError("The activation function should be one of {}.".format(ACT_FUNC_MAP.keys()))
        act_func = ACT_FUNC_MAP[act_name]
        if latent_size_list is not None:
            layer_list.extend([nn.Linear(input_dim, latent_size_list[0]), act_func, nn.LayerNorm(latent_size_list[0])])
            for i in range(len(latent_size_list) - 1):
                layer_list.extend([nn.Linear(latent_size_list[i], latent_size_list[i + 1]), act_func, nn.LayerNorm(latent_size_list[i + 1])])
            layer_list.extend([nn.Linear(latent_size_list[-1], output_dim), act_func])
        else:
            layer_list.extend([nn.Linear(input_dim, output_dim)])
        self.net = nn.Sequential(*layer_list)
        self.mu_layer = nn.Linear(output_dim, output_dim)
        self.var_layer = nn.Linear(output_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim


    def forward(self, data):
        out = self.net(data)
        mu = self.mu_layer(out)
        std = torch.abs(self.var_layer(out)) # avoid incompatible std values
        return mu, std



if __name__ == '__main__':
    linear_net = LinearNet(input_dim=100, latent_size_list=[64, 32, 16], output_dim=8, act_name="sigmoid")
    print(linear_net)

    linear_net = LinearNet(input_dim=100, latent_size_list=None, output_dim=8, act_name="sigmoid")
    print(linear_net)
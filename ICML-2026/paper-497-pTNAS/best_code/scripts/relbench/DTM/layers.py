import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, ninput, nlayers, nhid, dropout, noutput=1, normalization='batch_norm'):
        """
        MLP with configurable normalization.

        Args:
            ninput: Input dimension
            nlayers: Number of hidden layers
            nhid: Hidden dimension
            dropout: Dropout probability
            noutput: Output dimension
            normalization: Type of normalization ('batch', 'layer', or None)
        """
        super().__init__()
        layers = list()
        for i in range(nlayers):
            layers.append(nn.Linear(ninput, nhid))

            # Add normalization layer based on the argument
            if normalization == 'batch_norm':
                layers.append(nn.BatchNorm1d(nhid))
            elif normalization == 'layer_norm':
                layers.append(nn.LayerNorm(nhid))
            elif normalization is not None:
                raise ValueError(f"normalization must be 'batch', 'layer', or None, got {normalization}")

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            ninput = nhid
        if nlayers == 0:
            nhid = ninput
        layers.append(nn.Linear(nhid, noutput))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x:   FloatTensor B*ninput
        :return:    FloatTensor B*nouput
        """
        return self.mlp(x)

    def reset_parameters(self):
        """Reset model parameters"""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)


class FactorizationMachine(nn.Module):

    def __init__(self, reduce_dim=True, normalize=False):
        super().__init__()
        self.reduce_dim = reduce_dim
        self.normalize = normalize

    def forward(self, x):
        """
        :param x:   FloatTensor B*F*E
        """
        square_of_sum = torch.sum(x, dim=1)**2                  # B*E
        sum_of_square = torch.sum(x**2, dim=1)                  # B*E
        fm = square_of_sum - sum_of_square                      # B*E
        
        F = x.size(1)
        
        if self.reduce_dim:
            fm = torch.sum(fm, dim=1)
            # [B, 1]
            
        if self.normalize:
            fm = fm / F

        return 0.5 * fm                                        # B*E/B

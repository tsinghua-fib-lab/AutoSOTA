import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import importlib
import torch.nn.functional as F

class IMPACTNet(torch.nn.Module):
    def __init__(self, n_features, n_hidden=64, n_output=64, score_dim=3,
                 kernel_size=2, bias=True, maxpool_out_channels=8, normalize_embedding=True,
                 ref_size=5, dropout=0.2, activation='ReLU'):
        super(IMPACTNet, self).__init__()
        """
        Initializes the IMPACTNet with specified parameters.
        """
        network_params = {
            'n_features': n_features,
            'n_hidden': n_hidden,
            'n_output': n_output,
            'kernel_size': kernel_size,
            'activation': activation,
            'bias': bias,
            'maxpool_out_channels': maxpool_out_channels,
            'normalize_embedding': normalize_embedding,
            'dropout': dropout
        }

        self.feature_extractor = TCNEncoder(**network_params)
        self.seen_head = HolisticHead(n_output, score_dim)
        self.pseudo_head = HolisticHead(n_output, score_dim)
        self.ref_size = ref_size
        pooled_dim = int(n_hidden) * maxpool_out_channels if isinstance(n_hidden, int) else int(n_hidden.split(",")[-1]) * maxpool_out_channels
        self.recon_decoder = nn.Linear(n_output, pooled_dim)

    def forward(self, x, label):
        x_pyramid = list()
        for i in range(2):
            x_pyramid.append(list())
        feature = self.feature_extractor(x)
        abnormal_scores = self.seen_head(feature[label != 2])
        dummy_scores = self.pseudo_head(feature[label != 1])
        for i, scores in enumerate([abnormal_scores, dummy_scores]):
            x_pyramid[i].append(scores)
        for i in range(2):
            x_pyramid[i] = torch.cat(x_pyramid[i], dim=1)
        return x_pyramid

class HolisticHead(nn.Module):
    def __init__(self, in_dim, score_dim, dropout=0):
        super(HolisticHead, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_dim, in_dim),
                                   nn.GroupNorm(num_groups=4, num_channels=in_dim),
                                   nn.ReLU())
        self.fc2 = nn.Linear(in_dim, score_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.fc1(x))
        x = self.fc2(x)
        return x

def _instantiate_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_()

class TCNEncoder(torch.nn.Module):
    def __init__(self, n_features, n_hidden='500,100', n_output=20,
                 kernel_size=2, bias=False,
                 dropout=0.2, activation='ReLU', maxpool_out_channels: int = 1,
                 normalize_embedding: bool = True):
        super(TCNEncoder, self).__init__()
        self.layers = []
        self.num_inputs = n_features
        self.normalize_embedding = normalize_embedding

        if type(n_hidden) == int:
            n_hidden = [n_hidden]
        if type(n_hidden) == str:
            n_hidden = n_hidden.split(',')
            n_hidden = [int(a) for a in n_hidden]
        num_layers = len(n_hidden)

        if dropout is None:
            dropout = 0.0

        for i in range(num_layers):
            dilation_size = 2 ** i
            padding_size = (kernel_size-1) * dilation_size
            in_channels = n_features if i == 0 else n_hidden[i-1]
            out_channels = n_hidden[i]
            self.layers += [TcnResidualBlock(in_channels, out_channels, kernel_size,
                                             stride=1, dilation=dilation_size,
                                             padding=padding_size, dropout=dropout,
                                             bias=bias, activation=activation)]
        self.network = torch.nn.Sequential(*self.layers)
        maxpool_out_channels = int(maxpool_out_channels)
        self.maxpooltime = torch.nn.AdaptiveMaxPool1d(maxpool_out_channels)
        self.flatten = torch.nn.Flatten()  # Flatten two and third dimensions (tcn_out_channels and time)
        self.l1 = torch.nn.Linear(n_hidden[-1] * maxpool_out_channels, n_output, bias=bias)

    def get_pooled(self, x):
        """Return pooled TCN features before final linear projection."""
        out = self.network(x.transpose(2, 1))
        return self.flatten(self.maxpooltime(out))
    def forward(self, x, style='en'):
        if style == 'de':
            out = self.network(x.transpose(2, 1))
            return out
        out = self.network(x.transpose(2, 1))
        out = self.flatten(self.maxpooltime(out))
        rep = self.l1(out)
        if self.normalize_embedding:
            return F.normalize(rep, p=2, dim=1)
        else:
            return rep

class Chomp1d(torch.nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Clipped module, clipped the extra padding
        """
        return x[:, :, :-self.chomp_size].contiguous()

class TcnResidualBlock(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 dropout=0.2, activation='ReLU', bias=True):
        super(TcnResidualBlock, self).__init__()

        self.conv1 = weight_norm(torch.nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                 stride=stride, padding=padding, bias=bias,
                                                 dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.act1 = _instantiate_class("torch.nn.modules.activation", activation)
        self.dropout1 = torch.nn.Dropout(dropout)

        self.conv2 = weight_norm(torch.nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                 stride=stride, padding=padding, bias=bias,
                                                 dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.act2 = _instantiate_class("torch.nn.modules.activation", activation)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.net = torch.nn.Sequential(self.conv1, self.chomp1, self.act1, self.dropout1,
                                       self.conv2, self.chomp2, self.act2, self.dropout2)
        self.downsample = torch.nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.out_act = _instantiate_class("torch.nn.modules.activation", activation)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x shape:(bs, embed, seq_len)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.out_act(out + res)
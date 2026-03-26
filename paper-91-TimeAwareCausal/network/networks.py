# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import numpy as np

# from domainbed.lib import wide_resnet
from engine.configs import Embeddings, Classifers


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SqueezeLastTwo(nn.Module):
    """
    A module which squeezes the last two dimensions,
    ordinary squeeze can be a problem for batch size 1
    """

    def __init__(self):
        super(SqueezeLastTwo, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1])


class MLP(nn.Module):
    """Just  an MLP"""

    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
                for _ in range(hparams['mlp_depth'] - 2)
            ]
        )
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class CMNIST_MLP(nn.Module):
    "For fairly comparison with IRM & BIRM"
    def __init__(self, input_shape, hparams):
        super(CMNIST_MLP, self).__init__()
        self.input_shape = input_shape
        self.hparams = hparams
        self.hidden_dim = hparams["hidden_dim"]
        self.grayscale_model = hparams["grayscale_model"]
        if self.grayscale_model:
          lin1 = nn.Linear(14 * 14, self.hidden_dim)
        else:
          lin1 = nn.Linear(2 * 14 * 14, self.hidden_dim)
        lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        lin3 = nn.Linear(self.hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
          nn.init.xavier_uniform_(lin.weight)
          nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True)) #, lin3)
        self.n_outputs = self.hidden_dim

    def forward(self, input):
        if self.grayscale_model:
          out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
        else:
          out = input.view(input.shape[0], 2 * 14 * 14)
        out = self._main(out)
        return out


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, hparams, network=None):
        super(ResNet, self).__init__()
        if hparams["resnet18"]:
            if network is None:
                network = torchvision.models.resnet18(pretrained=hparams["pretrained"])
            self.network = network
            self.n_outputs = 512
        else:
            if network is None:
                network = torchvision.models.resnet50(pretrained=hparams["pretrained"])
            self.network = network
            self.n_outputs = 2048

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams["resnet_dropout"])
        self.freeze_bn()

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        if self.hparams["freeze_bn"] is False:
            return

        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """

    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.squeezeLastTwo = SqueezeLastTwo()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = self.squeezeLastTwo(x)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 2:
        return MLP(input_shape[1], 32, hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (84, 84):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError(f"Input shape {input_shape} is not supported")


class LocallyConnected(nn.Module):
    """
    Local linear layer, i.e., Conv1dLocal() with filter size 1.

    Args:
        num_linear: num of local linear layers
        in_features: m1
        out_features: m2
        bias: whether to include bias

    Shape:
        - Input: [n, d, m1]
        - Output: [n, d, m2]

    Attributes:
        weight: [d, m1, m2]
        bias: [d, m2]
    """
    def __init__(self, num_linear, in_features, out_features, bias=True):
        super(LocallyConnected, self).__init__()
        self.num_linear = num_linear
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(num_linear,
                                                in_features,
                                                out_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.in_features
        bound = np.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = torch.matmul(inputs.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        return 'num_linear={}, in_features={}, out_features={}, bias={}'.format(
            self.num_linear, self.in_features, self.out_features,
            self.bias is not None
        )


class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # detach so we can cast to NumPy
        E = torch.matrix_exp(input.detach().numpy())
        f = np.trace(E)
        E = torch.from_numpy(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        E, = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input


class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >=2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        self.d = dims[0]
        self.register_buffer("_identity", torch.eye(d))
        # fc1: variable spliting for l1 ref: <http://arxiv.org/abs/1909.13189>
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        # specific bounds for customize optimizer
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(nn.Sigmoid())
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.Sequential(*layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):
        # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)
        x = x.view(-1, self.dims[0], self.dims[1])
        x = self.fc2(x)
        x = x.squeeze(dim=2)
        return x

    def h_func(self):
        """
        Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG
        """
        d = self.dims[0]
        # [j * m1, i]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight
        fc1_weight = fc1_weight.view(d, -1, d)
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()
        # h = torch.trace(torch.matrix_exp(A)) - d
        # A different formulation, slightly faster at the cost of numerical stability
        M = self._identity + A / self.d
        E = torch.matrix_power(M, self.d - 1)
        h = (E.t() * M).sum() - self.d
        return h

    def l2_reg(self):
        """
        Take 2-norm-squared of all parameters
        """
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight
        reg += torch.sum(fc1_weight ** 2)

        for fc in self.fc2:
            if hasattr(fc, 'weight'):
                reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """
        Take l1 norm of fc1 weight
        """
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:
        """
        Get W from fc1 weight, take 2-norm over m1 dim
        """
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight
        fc1_weight = fc1_weight.view(d, -1, d)
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()
        return W

    @torch.no_grad()
    def fc1_to_p_sub(self) -> torch.Tensor:
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight
        fc1_weight = fc1_weight.view(d, -1, d)
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()
        P_sub = torch.inverse(self._identity - A)
        return P_sub


class LinearNotears(nn.Module):
    def __init__(self, dims, loss_type='l2'):
        super(LinearNotears, self).__init__()
        self.dims = dims
        self.loss_type = loss_type
        self.register_buffer("_I", torch.eye(dims))
        self.weight_pos = nn.Parameter(torch.zeros(dims, dims))
        self.weight_neg = nn.Parameter(torch.zeros(dims, dims))

    def _adj(self):
        return self.weight_pos - self.weight_neg

    def h_func(self):
        W = self._adj()
        E = torch.matrix_exp(W * W)
        h = torch.trace(E) - self.dims
        # G_h = E.T * W * 2
        return h

    def _h_faster(self):
        W = self._adj()
        M = self._I + W * W / self.dims
        E = torch.matrix_power(M, self.dims - 1)
        h = (E.T * M).sum() - self.dims
        return h

    def w_l1_reg(self):
        reg = torch.sum(self.weight_pos + self.weight_neg)
        return reg

    def forward(self, x):
        W = self._adj()
        M = x @ W
        return M

    def w_to_p_sub(self):
        W = self._adj()
        P_sub = torch.inverse(self._I - W)
        return P_sub


class NotearsClassifier(nn.Module):
    def __init__(self, dims, num_classes):
        super(NotearsClassifier, self).__init__()
        self.dims = dims
        self.num_classes = num_classes
        self.weight_pos = nn.Parameter(torch.zeros(dims + 1, dims + 1))
        self.weight_neg = nn.Parameter(torch.zeros(dims + 1, dims + 1))
        self.register_buffer("_I", torch.eye(dims + 1))
        self.register_buffer("_repeats", torch.ones(dims + 1).long())
        self._repeats[-1] *= num_classes

    def _adj(self):
        return self.weight_pos - self.weight_neg

    def _adj_sub(self):
        W = self._adj()
        return torch.matrix_exp(W * W)

    def h_func(self):
        W = self._adj()
        E = torch.matrix_exp(W * W)
        h = torch.trace(E) - self.dims - 1
        return h

    def w_l1_reg(self):
        reg = torch.mean(self.weight_pos + self.weight_neg)
        return reg

    def forward(self, x, y=None):
        W = self._adj()
        W_sub = self._adj_sub()
        if y is not None:
            x_aug = torch.cat((x, y.unsqueeze(1)), dim=1)
            M = x_aug @ W
            masked_x = x * W_sub[:self.dims, -1].unsqueeze(0)
            # reconstruct variables, classification logits
            return M[:, :self.dims], masked_x
        else:
            masked_x = x * W_sub[:self.dims, -1].unsqueeze(0).detach()
            return masked_x

    def mask_feature(self, x):
        W_sub = self._adj_sub()
        mask = W_sub[:self.dims, -1].unsqueeze(0).detach()
        return x * mask

    @torch.no_grad()
    def projection(self):
        self.weight_pos.data.clamp_(0, None)
        self.weight_neg.data.clamp_(0, None)
        self.weight_pos.data.fill_diagonal_(0)
        self.weight_neg.data.fill_diagonal_(0)

    @torch.no_grad()
    def masked_ratio(self):
        W = self._adj()
        return torch.norm(W[:self.dims, -1], p=0)


class LightEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, num_hidden_layers=0) -> None:
        super(LightEncoder, self).__init__()
        self.dropout = nn.Dropout(0.25)
        layers = [
            nn.Linear(in_channels, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            self.dropout,
        ]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(self.dropout)
        layers.append(nn.Linear(hidden_size, out_channels))
        self.encoder = nn.Sequential(*layers)
        self._initialize_weights(self.encoder)

    def _initialize_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        feat = self.encoder(x)
        return feat


import torch
import torch.nn as nn
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F
from typing import List
from torch.utils.hooks import RemovableHandle


def mul(t: Iterable[int]):
    result = 1
    for item in t:
        result *= item

    return result


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        # nn.init.kaiming_normal_(m.bias)

        m.bias.data.fill_(0.01)


class Time2Vec(nn.Module):
    '''
    Time encoding inspired by the Time2Vec paper
    '''

    def __init__(self, in_shape, out_shape):

        super(Time2Vec, self).__init__()
        linear_shape = out_shape // 4
        dirac_shape = 0
        sine_shape = out_shape - linear_shape - dirac_shape
        self.model_0 = nn.Linear(in_shape, linear_shape)
        self.model_1 = nn.Linear(in_shape, sine_shape)

    def forward(self, X):

        te_lin = self.model_0(X)
        te_sin = torch.sin(self.model_1(X))
        if len(te_lin.shape) == 3:
            te_lin = te_lin.squeeze(1)
        if len(te_sin.shape) == 3:
            te_sin = te_sin.squeeze(1)
        te = torch.cat((te_lin, te_sin), dim=1)
        return te


class TimeReLUCompatible(nn.Module):
    '''
    A ReLU with threshold and alpha as a function of domain indices.
    '''

    def __init__(self, time_shape, leaky=False, use_time=True, deep=False):

        super(TimeReLUCompatible, self).__init__()
        self.deep = deep  # Option for a deeper version of TReLU
        if deep:
            trelu_shape = 16
        else:
            trelu_shape = 1

        self.trelu_shape = trelu_shape
        self.leaky = leaky

        self.use_time = use_time  # Whether TReLU is active or not
        self.model_0 = nn.Linear(time_shape, trelu_shape)

        # self.model_1 = nn.Linear(trelu_shape, data_shape)
        self.model_1 = None

        self.time_dim = time_shape

        if self.leaky:
            self.model_alpha_0 = nn.Linear(time_shape, trelu_shape)

            # self.model_alpha_1 = nn.Linear(trelu_shape, data_shape)
            self.model_alpha_1 = None

        self.sigmoid = nn.Sigmoid()

        if self.leaky:
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()

        self.times = None

    def forward(self, X):

        assert self.times is not None, "please call self.set_times(times) for this forward process"

        if not self.use_time:
            return self.relu(X)
        if len(self.times.size()) == 3:
            self.times = self.times.squeeze(2)
        thresholds = self.model_1(self.relu(self.model_0(self.times)))

        # reshape to (B, ...)
        thresholds = torch.reshape(thresholds, (-1, *self.data_shape))

        if self.leaky:
            alphas = self.model_alpha_1(self.relu(self.model_alpha_0(self.times)))
            alphas = torch.reshape(alphas, (-1, *self.data_shape))
        else:
            alphas = 0.0
        if self.deep:
            X = torch.where(X > thresholds, X, alphas * (X - thresholds) + thresholds)
        else:
            X = torch.where(X > thresholds, X - thresholds, alphas * (X - thresholds))

        # set times to None
        self.times = None

        return X

    def set_times(self, times):
        self.times = times


def hook_for_init_timerelu(module: TimeReLUCompatible, input_):
    X, = input_
    if module.model_1 is not None:
        # print("please delete this hook after the first run!")
        return
    shape = X.shape
    assert len(shape) >= 2, "error input size"  # (B, ...)
    data_shape = shape[1:]
    setattr(module, "data_shape", data_shape)
    model_1 = nn.Linear(module.trelu_shape, mul(data_shape))
    model_alpha_1 = nn.Linear(module.trelu_shape, mul(data_shape))

    device = module.model_0.weight.device
    model_1.to(device)
    model_alpha_1.to(device)

    setattr(module, "model_1", model_1)
    setattr(module, "model_alpha_1", model_alpha_1)


@Classifers.register('time_cla')
class TimeReluClassifier(nn.Module):
    def __init__(self, input_dim, out_dim, time_dim=8):
        super(TimeReluClassifier, self).__init__()
        self.fc_1 = nn.Linear(input_dim, 256)

        self.time_relu = TimeReLUCompatible(time_dim, leaky=True, use_time=True)
        self.handle = self.time_relu.register_forward_pre_hook(hook_for_init_timerelu)

        self.fc_2 = nn.Linear(256, out_dim)
        self.first_forward = True

    def forward(self, x, time_v):
        x = self.fc_1(x)

        self.time_relu.set_times(time_v)
        x = self.time_relu(x)

        out = self.fc_2(x)

        if self.first_forward:
            self.handle.remove()
            self.first_forward = False

        return out


class WrappedGIFeature(nn.Module):
    def __init__(self, feature: nn.Module, time_dim=8, num_replace=None):
        # network already in gup
        super(WrappedGIFeature, self).__init__()
        self.time_dim = time_dim
        self.timerelu_names, self.timerelu_handles = self.replace_relu_with_timerelu(feature, time_dim, num_replace)
        assert len(self.timerelu_names) >= 0, "no relu in the model!"
        self.feature = feature
        self.first_forward = True

    @staticmethod
    def replace_relu_with_timerelu(network: nn.Module, time_dim, num_replace=None):
        timerelu_names: List[str] = []
        timerelu_handles: List[RemovableHandle] = []
        for name, sub_module in network.named_modules(remove_duplicate=False):
            if isinstance(sub_module, nn.ReLU):
                timerelu_names.append(name)

        if num_replace is not None:
            timerelu_names.reverse()
            timerelu_names = timerelu_names[0:num_replace]

        for name in timerelu_names:
            parent_name, _, relu_name = name.rpartition(".")
            parent_module = network.get_submodule(parent_name)
            time_relu = TimeReLUCompatible(time_dim, leaky=True, use_time=True)
            handle = time_relu.register_forward_pre_hook(hook_for_init_timerelu)
            timerelu_handles.append(handle)
            setattr(parent_module, relu_name, time_relu)

        return timerelu_names, timerelu_handles

    def forward(self, x, time_v):
        self.set_time_for_timerelu(time_v)

        out = self.feature(x)

        if self.first_forward:
            self.remove_hooks()
            self.first_forward = False

        return out

    def remove_hooks(self):
        for handle in self.timerelu_handles:
            handle.remove()

    def set_time_for_timerelu(self, time):
        for name in self.timerelu_names:
            timerelu: TimeReLUCompatible = self.feature.get_submodule(name)
            timerelu.set_times(time)


@Embeddings.register('toy_linear_fe_two')
class LinearFeatExtractorTwo(nn.Module):
    def __init__(self, input_shape, output_dim, hidden_dim=512, depth=4, drop_rate=0.):
        super(LinearFeatExtractorTwo, self).__init__()
        self.input = nn.Linear(input_shape[-1], hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.relu_input = nn.ReLU()
        self.hiddens = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(depth - 2)])
        self.relu_list = nn.ModuleList([
            nn.ReLU() for _ in range(depth - 2)
        ])
        self.output = nn.Linear(hidden_dim, output_dim)
        self.n_outputs = output_dim

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = self.relu_input(x)
        for hidden, relu in zip(self.hiddens, self.relu_list):
            x = hidden(x)
            x = self.dropout(x)
            x = relu(x)
        x = self.output(x)
        return x


class WrappedGINetwork(nn.Module):
    def __init__(self, feature: WrappedGIFeature, classifier: TimeReluClassifier, time_dim=8, time_append_dim=256):
        super(WrappedGINetwork, self).__init__()

        self.time_dim = time_dim
        self.time_append_dim = time_append_dim

        self.t2v = Time2Vec(1, self.time_dim)

        self.time_fc = nn.Linear(self.time_dim, self.time_append_dim)

        self.feature = feature

        self.classifier = classifier

    def forward(self, x, t):
        time_v = self.t2v(t)

        feature = self.feature(x, time_v)

        time_append = self.time_fc(time_v)

        feature = torch.cat((feature, time_append), dim=-1)

        out = self.classifier(feature, time_v)

        return out

    def foward_encoder(self, x, t):
        time_v = self.t2v(t)

        feature = self.feature(x, time_v)

        time_append = self.time_fc(time_v)

        feature = torch.cat((feature, time_append), dim=-1)

        return feature


import torch
import torch.nn as nn
from typing import Iterable
import torch.nn.functional as F


class SlidingWindow:
    def __init__(self, size):
        assert size > 0
        self.size = size
        self._data = []

    def __len__(self):
        return len(self._data)

    def push(self, _data: torch.Tensor):
        _data = _data.detach()
        if len(self) == self.size:
            self.pop()
        self._data.append(_data)

    def pop(self):
        if len(self) > 0:
            self._data.pop()

    @property
    def data(self):
        return self._data

    def reset(self):
        del self._data
        self._data = []


class WrappedDrainNetwork(nn.Module):

    def __init__(self, network: nn.Module,
                 hidden_dim: int,
                 latent_dim: int,
                 num_rnn_layers=1,
                 num_layer_to_replace=-1,
                 window_size=-1,
                 lambda_forgetting=0.):
        super(WrappedDrainNetwork, self).__init__()
        self.num_layer_to_replace = num_layer_to_replace  # < 0 means all
        self.window_size = window_size
        self.lambda_forgetting = lambda_forgetting

        self.sliding_window = SlidingWindow(self.window_size) if self.window_size > 0 else None

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_rnn_layer = num_rnn_layers

        offset = self.process_network(network)

        self.network = network  # without any parameters

        self.code_dim = offset

        self.rnn = nn.LSTM(self.latent_dim, self.latent_dim, self.num_rnn_layer)

        # Transforming LSTM output to vector shape
        self.param_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.code_dim))
        # Transforming vector to LSTM input shape
        self.param_encoder = nn.Sequential(
            nn.Linear(self.code_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim))

        # self.init_e_hidden()

        init_c, init_h = [], []
        for _ in range(self.num_rnn_layer):
            init_c.append(torch.tanh(torch.randn(1, self.latent_dim)))
            init_h.append(torch.tanh(torch.randn(1, self.latent_dim)))

        # self.hidden_0 = (torch.stack(init_c, dim=0).cuda(), torch.stack(init_h, dim=0).cuda())
        self.register_buffer("hidden_c0", torch.stack(init_c, dim=0))
        self.register_buffer("hidden_h0", torch.stack(init_h, dim=0))
        # E0 is directly learnt from the first domain
        self.E0 = nn.Parameter(torch.randn((1, self.code_dim)))

        self.previous_E = None  # none means the first domain
        self.previous_hidden_c_h = None

    def reset_e_hidden(self):
        self.previous_E = None
        self.previous_hidden_c_h = None

        if self.sliding_window is not None:
            self.sliding_window.reset()

    @torch.no_grad()
    def next_e_hidden(self):
        if self.previous_E is None:
            # current is the first domain
            # then next domain will use the domain0's e and h
            self.previous_E = self.E0.detach()
            self.previous_hidden_c_h = (self.hidden_c0, self.hidden_h0)
        else:
            previous_E = self.previous_E
            previous_hidden = self.previous_hidden_c_h
            # only update E, hidden
            lstm_input = self.param_encoder(previous_E)
            lstm_out, new_hidden = self.rnn(lstm_input.unsqueeze(0), previous_hidden)
            new_E = self.param_decoder(lstm_out.squeeze(0))

            self.previous_E = new_E
            self.previous_hidden_c_h = new_hidden

        if self.sliding_window is not None:
            self.sliding_window.push(self.previous_E.view(-1))

    def process_network(self, network):
        offset = 0
        block_names = self.get_block_names(network)
        for index, name in enumerate(block_names):
            if 0 < self.num_layer_to_replace <= index:
                break
            block = network.get_submodule(name)
            for sub_module in block.modules():
                offset += self.trans_param_to_buffer(sub_module, offset)

        return offset

    @staticmethod
    def get_block_names(network):
        if isinstance(network, nn.Module):
            block_names = [""]
        else:
            raise NotImplementedError
        block_names.reverse()
        return block_names

    @staticmethod
    def trans_param_to_buffer(module: nn.Module, offset: int):
        module.has_been_transformed = True
        names = []
        shapes = []
        # get the name and shape of params of the current module
        for name, param in module.named_parameters(recurse=False):
            names.append(name)
            shapes.append(param.shape)

        for name, shape in zip(names, shapes):
            # delete all parameters and register these name as buffer (maybe simple attribute)
            delattr(module, name)
            module.register_buffer(name, torch.randn(shape))

        module.transformed_names = names
        module.shapes_for_names = shapes
        module.offset = offset

        if len(names) == 0:
            return 0
        else:
            num_params = 0
            for shape in shapes:
                num_params += mul(shape)
            return num_params

    def reconstruct(self, decoded_params: torch.Tensor):
        decoded_params = self.skip_connection(decoded_params)
        for sub_module in self.network.modules():
            self.reconstruct_module(sub_module, decoded_params)

    def skip_connection(self, decoded_params: torch.Tensor):
        if self.sliding_window is not None:
            history = self.lambda_forgetting * sum(self.sliding_window.data)
            # history.squeeze()
            return decoded_params + history
        else:
            return decoded_params

    @staticmethod
    def reconstruct_module(module: nn.Module, decoded_params: torch.Tensor):
        if not hasattr(module, "has_been_transformed"):
            return

        offset = module.offset
        all_names = module.transformed_names
        all_shapes = module.shapes_for_names

        local_offset = 0

        for name, shape in zip(all_names, all_shapes):
            value = torch.reshape(decoded_params[offset + local_offset:offset + local_offset + mul(shape)], shape)
            setattr(module, name, value)
            local_offset += mul(shape)

    def forward(self, x):
        if self.previous_E is not None and self.previous_hidden_c_h is not None:
            # which might be slow for inference
            lstm_input = self.param_encoder(self.previous_E)
            lstm_out, _ = self.rnn(lstm_input.unsqueeze(0), self.previous_hidden_c_h)
            new_E = self.param_decoder(lstm_out.squeeze(0))
            self.reconstruct(new_E.view(-1))
        elif self.previous_E is None and self.previous_hidden_c_h is None:
            # the first domain
            self.reconstruct(self.E0.view(-1))
        else:
            raise RuntimeError("self.previous_E and self.previous_hidden_c_h should be all None or all not None!")

        prediction = self.network(x)

        return prediction


class DomainAdaptor(nn.Module):
    def __init__(self, hdim, hparams, n_heads=16, MLP=None):
        super(DomainAdaptor, self).__init__()
        self.attentive_module = nn.ModuleList([nn.MultiheadAttention(embed_dim=hdim, num_heads=n_heads) for _ in range(hparams['attn_depth'])])
        self.bn = nn.BatchNorm1d(hdim)
        self.layers = hparams['attn_depth']
        self.batch_size = hparams['batch_size']
        self.n_heads = hparams['attn_head'] if 'attn_head' in hparams else n_heads
        self.env_number = hparams['env_number']
        self.MLP = MLP

    def forward(self, x, x_kv, attn_mask=None, attend_to_domain_embs=False):
        x_0 = x
        x = x.unsqueeze(1)
        x_kv = x_kv.unsqueeze(1)
        if attend_to_domain_embs:
            for i in range(self.layers):
                residual = x
                x = self.attentive_module[i](x, x_kv, x_kv, attn_mask=attn_mask)[0]
            if self.MLP:
                return self.bn(x.squeeze(1) + self.MLP(x_0))
            else:
                return x.squeeze(1)
        else:
            for i in range(self.layers):
                residual = x
                x = self.attentive_module[i](x, x, x, attn_mask=attn_mask)[0]
                x = x + residual
            return self.bn(x.squeeze(1) + self.MLP(x_0))


class Masker(nn.Module):
    def __init__(self, in_dim=2048, num_classes=2048, middle=8192, k=1024):
        super(Masker, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.k = k

        self.layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(middle, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(middle, num_classes))

        self.bn = nn.BatchNorm1d(num_classes, affine=False)

    def forward(self, f):
       # mask = self.bn(self.layers(f))
       mask = self.layers(f)
       z = torch.zeros_like(mask)
       remaining_mask = mask.clone()

       # for _ in range(self.k):
       #     sample = F.gumbel_softmax(mask, dim=1, tau=0.5, hard=False)
       #     z = torch.maximum(sample, z)

       for _ in range(self.k):
           sample = F.gumbel_softmax(remaining_mask, dim=1, tau=0.5, hard=False)
           z = torch.maximum(sample, z)
           _, max_indices = sample.max(dim=1)
           remaining_mask[torch.arange(mask.size(0)), max_indices] = -float('inf')

       return z


class Masker_inputfree(nn.Module):
    def __init__(self, in_dim=2048, num_classes=2048, middle=8192, k=1024):
        super(Masker_inputfree, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.k = k

        self.fix_input = torch.randn(1, num_classes).cuda()

        self.layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(middle, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(middle, num_classes))

        self.bn = nn.BatchNorm1d(num_classes, affine=False)

    def forward(self, _=None):
       mask = self.bn(self.layers(self.fix_input))
       z = torch.zeros_like(mask)
       for _ in range(self.k):
           sample = F.gumbel_softmax(mask, dim=1, tau=0.5, hard=True)
           z = torch.maximum(sample, z)
       return z
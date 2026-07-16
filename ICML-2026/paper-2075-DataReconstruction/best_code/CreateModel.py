import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

import common_utils


def get_activation(activation, model_relu_alpha):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'modifiedrelu':
        return ModifiedRelu(model_relu_alpha)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU()


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ModifiedReluFunc(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.set_materialize_grads(False)
        ctx.x = x
        ctx.alpha = alpha
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None
        return grad_output * ctx.x.mul(ctx.alpha).sigmoid(), None


class ModifiedRelu(nn.Module):
    def __init__(self, alpha):
        super(ModifiedRelu, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return ModifiedReluFunc.apply(x, self.alpha) # here you call the function!


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, output_dim, activation, use_bias=False):
        super().__init__()
        self.activation = activation
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim_list[0])])
        for i in range(1, len(hidden_dim_list)):
            self.layers.append(nn.Linear(hidden_dim_list[i-1], hidden_dim_list[i], bias=use_bias))
        self.layers.append(nn.Linear(hidden_dim_list[-1], output_dim, bias=False))  # output layer

    def forward(self, data):
        feats = Flatten()(data)
        for layer in self.layers[:-1]:
            feats = layer(feats)
            feats = self.activation(feats)
        feats = self.layers[-1](feats)
        return feats


class ConvFCNetwork(nn.Module):
    def __init__(
        self,
        input_channels,
        input_height,
        input_width,
        activation,
        use_bias=False
    ):
        super().__init__()
        self.activation = activation

        # -------- First layer: Conv --------
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,     # Keep H and W unchanged
            bias=False
        )

        # -------- Fully connected layers --------
        conv_out_dim = 32 * input_height * input_width  # 32 channels, H, W unchanged

        self.layers = nn.ModuleList([
            nn.Linear(conv_out_dim, 1000, bias=False),
            nn.Linear(1000, 1, bias=False)   # output layer
        ])

    def forward(self, data):
        """
        data: (B, C, H, W)
        """
        feats = self.conv(data)
        feats = self.activation(feats)

        feats = feats.view(feats.size(0), -1)  # Flatten

        feats = self.layers[0](feats)
        feats = self.activation(feats)

        feats = self.layers[1](feats)
        return feats

def create_model(args, extraction):
    if not extraction:
        activation = get_activation(args.model_train_activation, args.extraction_model_relu_alpha)
    else:
        activation = get_activation(args.extraction_model_activation, args.extraction_model_relu_alpha)

    if args.model_type == 'mlp':
        model = NeuralNetwork(
            input_dim=args.input_dim, hidden_dim_list=args.model_hidden_list, output_dim=args.output_dim,
            activation=activation, use_bias=args.model_use_bias
        )
    elif args.model_type == 'convfc':
        model = ConvFCNetwork(
            input_channels=args.input_channels,
            input_height=args.input_height,
            input_width=args.input_width,
            activation=activation,
            use_bias=args.model_use_bias
        )
    else:
        raise ValueError(f'No such args.model_type={args.model_type}')

    model = model.to(args.device)

    # initialize
    if args.use_init_scale and not extraction:

        if not args.use_init_scale_only_first:
            assert len(args.model_init_list) == 1 + len(args.model_hidden_list), "use_init_scale_only_first=False but you didn't specify suitable model_init_list"

        # intialize bias of first layer
        if hasattr(model.layers[0], 'bias') and model.layers[0].bias is not None:
            model.layers[0].bias.data.normal_().mul_(args.model_init_list[0])

        if args.use_init_scale_only_first:
            print('Initializing model weights - Only First Layer')
            model.layers[0].weight.data.normal_().mul_(args.model_init_list[0])
        else:
            print('Initializing model weights - All Layers')
            j = 0
            for i in range(len(model.layers)):
                name = model.layers[i].__class__.__name__.lower()
                if 'conv' in name or 'linear' in name:
                    model.layers[i].weight.data.normal_().mul_(args.model_init_list[j])
                    print(i, name, 'scale', args.model_init_list[j])
                    j += 1

    else:
        print('NO INITIALIZATION OF WEIGHTS')

    common_utils.common.calc_model_parameters(model)

    return model

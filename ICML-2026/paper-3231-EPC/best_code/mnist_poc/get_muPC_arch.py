from typing import Callable

from torch import nn


# Make one muP-ResNet block
def make_resnet_block(  # Paper notation
    in_features: int,  # = N_{l-1}
    out_features: int,  # = N_l
    act_fn: Callable,  # = phi_l
    scaling: float,  # = a_l
    skip_connection: bool,  # = tau_l
    w_init_var: float,  # = b_l
):

    class ResNet_Block(nn.Linear):
        def reset_parameters(self):
            nn.init.normal_(self.weight, std=w_init_var)
            if self.bias is not None:  # typically, muPC uses no  biases
                nn.init.zeros_(self.bias)

        def forward(self, x):
            output = scaling * super().forward(act_fn(x))

            if skip_connection:
                output += x

            return output

    return ResNet_Block(in_features, out_features, bias=False)


# Make full muP-ResNet architecture
def make_resnet(
    input_dim: int,
    output_dim: int,
    width: int,  # = N
    depth: int,  # = L = H+1
):
    act_fn = nn.Tanh()

    first_layer = make_resnet_block(
        input_dim,
        width,
        nn.Identity(),  # no act_fn on given input!
        scaling=input_dim ** (-0.5),
        skip_connection=False,  # Can't do skip when dim changes
        w_init_var=1.0,
    )

    hidden_layers = [
        make_resnet_block(
            width,
            width,
            act_fn,
            scaling=(width * depth) ** (-0.5),
            skip_connection=True,
            w_init_var=1.0,
        )
        for _ in range(depth - 2)
    ]

    final_layer = make_resnet_block(
        width,
        output_dim,
        act_fn,
        scaling=width ** (-1.0),
        skip_connection=False,  # Can't do skip when dim changes
        w_init_var=1.0,
    )
    # Very strange: muPC does NOT use an output activation (like sigmoid)
    # It uses MSELoss directly on the outputs, or a CELoss (which already includes a softmax)

    return [first_layer] + hidden_layers + [final_layer]


def get_architecture(dataset: str):
    if dataset == "EMNIST" or dataset == "FashionMNIST":
        architecture = make_resnet(input_dim=28 * 28, output_dim=10, width=128, depth=32)

    elif dataset == "EMNIST-deep" or dataset == "FashionMNIST-deep":
        architecture = make_resnet(input_dim=28 * 28, output_dim=10, width=64, depth=64)

    return architecture

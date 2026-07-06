import torch
import torch.nn as nn
import math


class LocallyConnected(nn.Module):
    """Local linear layer, i.e. Conv1dLocal() with filter size 1.

    Args:
        num_linear: num of local linear layers, i.e.
        in_features: m1
        out_features: m2
        bias: whether to include bias or not

    Shape:
        - Input: [n, d, m1]
        - Output: [n, d, m2]

    Attributes:
        weight: [d, m1, m2]
        bias: [d, m2]
    """

    def __init__(self, num_linear, input_features, output_features, bias=True):
        super(LocallyConnected, self).__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(num_linear,
                                                input_features,
                                                output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = torch.matmul(input.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'num_linear={}, in_features={}, out_features={}, bias={}'.format(
            self.num_linear, self.in_features, self.out_features,
            self.bias is not None
        )


def main():
    n, d, m1, m2 = 2, 3, 5, 7

    # numpy
    import numpy as np
    input_numpy = np.random.randn(n, d, m1)
    weight = np.random.randn(d, m1, m2)
    output_numpy = np.zeros([n, d, m2])
    for j in range(d):
        # [n, m2] = [n, m1] @ [m1, m2]
        output_numpy[:, j, :] = input_numpy[:, j, :] @ weight[j, :, :]

    # torch
    torch.set_default_dtype(torch.double)
    input_torch = torch.from_numpy(input_numpy)
    locally_connected = LocallyConnected(d, m1, m2, bias=False)
    locally_connected.weight.data[:] = torch.from_numpy(weight)
    output_torch = locally_connected(input_torch)

    # compare
    print(torch.allclose(output_torch, torch.from_numpy(output_numpy)))


if __name__ == '__main__':
    main()

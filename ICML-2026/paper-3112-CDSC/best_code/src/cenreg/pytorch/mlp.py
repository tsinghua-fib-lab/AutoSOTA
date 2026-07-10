import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """
    Single Layer Perceptron
    """

    def __init__(self, output_len: int):
        super().__init__()
        self.fc = nn.Linear(1, output_len, bias=False)

    def forward(self, x: torch.Tensor):
        return F.softmax(self.fc(x))


class MLP(nn.Module):
    """
    Multi Layer Perceptron
    """

    def __init__(self, input_len: int, output_len: int, num_neuron: int):
        super().__init__()
        self.fc1 = nn.Linear(input_len, num_neuron)
        self.fc2 = nn.Linear(num_neuron, num_neuron)
        self.fc3 = nn.Linear(num_neuron, num_neuron)
        self.fc4 = nn.Linear(num_neuron, output_len)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.softmax(self.fc4(x), dim=1)
        return x


class MLPMultiHead(nn.Module):
    """
    Multi Layer Perceptron with Multiple Outputs
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        output_num: int,
        num_neuron: int,
        use_softmax: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_len, num_neuron)
        list_fc2 = []
        list_fc3 = []
        list_fc4 = []
        for _ in range(output_num):
            list_fc2.append(nn.Linear(num_neuron, num_neuron))
            list_fc3.append(nn.Linear(num_neuron, num_neuron))
            list_fc4.append(nn.Linear(num_neuron, output_len))
        self.list_fc2 = nn.ModuleList(list_fc2)
        self.list_fc3 = nn.ModuleList(list_fc3)
        self.list_fc4 = nn.ModuleList(list_fc4)
        self.dropout = nn.Dropout(0.5)
        self.use_softmax = use_softmax

    def forward(self, x: torch.Tensor):
        x = self.dropout(F.relu(self.fc1(x)))
        list_out = []
        for i in range(len(self.list_fc2)):
            x2 = x
            x2 = self.dropout(F.relu(self.list_fc2[i](x2)))
            x2 = self.dropout(F.relu(self.list_fc3[i](x2)))
            x2 = self.list_fc4[i](x2)
            if self.use_softmax:
                x2 = F.softmax(x2, dim=1)
            list_out.append(x2)
        return torch.stack(list_out)


class SMM(nn.Module):
    """
    Fully monotonic neural network.
    The output y is a function of input x, and the function is monotonic with respect to all dimensions of x.
    """

    def __init__(self, embed_size: int):
        super().__init__()
        self.K = 6

        # init parameters
        self.beta = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        list_init_w = []
        for _ in range(self.K):
            init_w = np.random.uniform(low=0.0, high=1.0, size=(embed_size, self.K))
            list_init_w.append(torch.Tensor(init_w))
        self.w = nn.ParameterList([nn.Parameter(list_init_w[i], requires_grad=True) for i in range(self.K)])
        list_init_b = []
        for _ in range(self.K):
            init_b = np.random.uniform(low=0.0, high=1.0, size=(self.K,))
            list_init_b.append(torch.Tensor(init_b))
        self.b = nn.ParameterList([nn.Parameter(list_init_b[i], requires_grad=True) for i in range(self.K)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        list_h = []
        for i in range(self.K):
            w2 = self.w[i] * self.w[i]
            h = torch.matmul(x, w2) + self.b[i]
            h = torch.logsumexp(h, dim=1)
            list_h.append(h.view(h.shape[0], 1))
        h = torch.cat(list_h, dim=1)
        y = -torch.logsumexp(-h, dim=1) / torch.exp(self.beta) + self.gamma
        return y


class SMMMultiHead(nn.Module):
    def __init__(self, input_len: int, input_monotone_len: int, output_num: int, num_neuron: int):
        """
        Initializes the SMMMultiHead class.

        Parameters
        ----------
            input_len: int
                number of features
            input_monotone_len: int
                number of columns of monotone input
            output_num: int
                number of output heads (not the number of columns of output)
            num_neuron: int
                number of neurons in the first layer
        """

        super().__init__()
        self.fc1 = nn.Linear(input_len, num_neuron)
        list_smm = []
        for _ in range(output_num):
            list_smm.append(SMM(num_neuron + input_monotone_len))
        self.list_smm = nn.ModuleList(list_smm)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 2:
            raise ValueError("x must be of shape [batch_size, num_features]")
        if len(t.shape) != 2:
            raise ValueError("t must be of shape [batch_size, num_monotone_features]")

        x = F.relu(self.fc1(x))
        list_out = []
        for i in range(len(self.list_smm)):
            xt = torch.cat([t, x], dim=1)
            xt = self.list_smm[i](xt)
            list_out.append(torch.sigmoid(xt.view(-1, 1)))
        return torch.concat(list_out, dim=1)

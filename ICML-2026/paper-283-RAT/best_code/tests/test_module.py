import torch
import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Polynomial3(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

class PopArt(torch.nn.Module):
    
    def __init__(self, input_shape, output_shape, norm_axes=1, beta=0.99999, epsilon=1e-5, device=torch.device("cuda")):
        super().__init__()

        # self.beta = beta
        # self.epsilon = epsilon
        # self.norm_axes = norm_axes

        # self.input_shape = input_shape
        # self.output_shape = output_shape

        self.weight = nn.Parameter(torch.randn(output_shape, input_shape))
        self.bias = nn.Parameter(torch.randn(output_shape))
        
        # self.stddev = nn.Parameter(torch.ones(output_shape), requires_grad=False).to(**self.tpdv)
        # self.mean = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
        # self.mean_sq = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
        # self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

        # self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    # def forward(self, input_vector):
    #     if type(input_vector) == np.ndarray:
    #         input_vector = torch.from_numpy(input_vector)
    #     input_vector = input_vector.to(**self.tpdv)

    #     return F.linear(input_vector, self.weight, self.bias)
    
    # @torch.no_grad()
    # def update(self, input_vector):
    #     if type(input_vector) == np.ndarray:
    #         input_vector = torch.from_numpy(input_vector)
    #     input_vector = input_vector.to(**self.tpdv)
    #     
    #     old_mean, old_var = self.debiased_mean_var()
    #     old_stddev = torch.sqrt(old_var)

    #     batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
    #     batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

    #     self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
    #     self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
    #     self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

    #     self.stddev = (self.mean_sq - self.mean ** 2).sqrt().clamp(min=1e-4)
    #     
    #     new_mean, new_var = self.debiased_mean_var()
    #     new_stddev = torch.sqrt(new_var)
    #     
    #     self.weight = self.weight * old_stddev / new_stddev
    #     self.bias = (old_stddev * self.bias + old_mean - new_mean) / new_stddev

    # def debiased_mean_var(self):
    #     debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
    #     debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
    #     debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
    #     return debiased_mean, debiased_var

    # def normalize(self, input_vector):
    #     if type(input_vector) == np.ndarray:
    #         input_vector = torch.from_numpy(input_vector)
    #     input_vector = input_vector.to(**self.tpdv)

    #     mean, var = self.debiased_mean_var()
    #     out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
    #     
    #     return out

    # def unnormalize(self, input_vector):
    #     if type(input_vector) == np.ndarray:
    #         input_vector = torch.from_numpy(input_vector)
    #     input_vector = input_vector.to(**self.tpdv)

    #     mean, var = self.debiased_mean_var()
    #     out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
    #     
    #     # out = out.cpu().numpy()

    #     return out


# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Construct our model by instantiating the class defined above
model = Polynomial3()
model2 = PopArt(16, 4)
breakpoint()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters (defined
# with torch.nn.Parameter) which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(2000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')
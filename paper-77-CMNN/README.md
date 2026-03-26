# Advancing Constrained Monotonic Neural Networks
This repository offers the code to reproduce and use the approach proposed in _"Advancing Constrained Monotonic Neural Networks: Achieving Universal Approximation Beyond Bounded Activations"_ published at ICML 2025

## How to use it
The following class implements th whole proposed approach. Keep in mind the first layer **should have `nn.Identity()` as activation** , while for the rest you can use your favorite monotonic activation (ReLU, SiLU, CELU, etc. etc.).
```python
class MonotonicLinear(nn.Linear):
    def __init__(
        self,
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        device=None, 
        dtype=None,
        pre_activation=nn.Identity(),
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.act = pre_activation
        
    def forward(self, x):
        w_pos = self.weight.clamp(min=0.0)
        w_neg = self.weight.clamp(max=0.0)
        x_pos = F.linear(self.act(x), w_pos, self.bias)
        x_neg = F.linear(self.act(-x), w_neg, self.bias)  
        return x_pos + x_neg
```

An example of monotonic MLP (you **must use at least 4 layers** to have theoretical guarantees of universal approximation):
```python
monotonic_mlp = nn.Sequential([
    MonotonicLinear(N, 16, pre_activation=nn.Identity()),
    MonotonicLinear(16, 16, pre_activation=nn.SELU()),
    MonotonicLinear(16, 16, pre_activation=nn.SELU()),
    MonotonicLinear(16, 1, pre_activation=nn.SELU()),
])
```

An example of **partially** monotonic MLP:
```python
non_monotonic_mlp = nn.Sequential([
    nn.LazyLinear(16),
    nn.SiLU(),
    nn.LazyLinear(16),
    nn.SiLU(),
])

monotonic_mlp = nn.Sequential([
    MonotonicLinear(16+N, 16, pre_activation=nn.Identity()),
    MonotonicLinear(16, 16, pre_activation=nn.SELU()),
    MonotonicLinear(16, 16, pre_activation=nn.SELU()),
    MonotonicLinear(16, 1, pre_activation=nn.SELU()),
])

x_non_monotonic = non_monotonic_mlp(x_non_monotonic)
x_monotonic = torch.cat((x, x_monotonic), dim=-1)
x_monotonic = monotonic_mlp(x_monotonic)
```

## Repository structure
Inside `experiments`, the `data` folder already contains the CSV for the datasets. Instead, `exp_{dataset}.ipynb` contains the training loop and the evaluation for the corresponding `dataset`

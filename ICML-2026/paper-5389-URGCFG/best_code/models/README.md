# Models Module

This module contains the core neural network models for representing and computing with finitely convex/concave functions.

## Components

### FiniteModel (`finite_model.py`)

A unified representation for finitely convex or finitely concave functions using max/min over affine combinations:

- **Convex mode**: `f(x) = max_j [k(x, y_j) - b_j]`
- **Concave mode**: `f(x) = min_j [k(x, y_j) - b_j]`

**Features:**
- Soft/hard/STE selection modes for max/min
- Numerical sup/inf transforms with multiple optimizers (LBFGS, Adam, GD)
- Full batching support with per-sample warm-start storage
- Bounded parameter initialization
- Support for fixed or learnable candidates

**Usage:**
```python
from models import FiniteModel
from tools.utils import L22

model = FiniteModel(
    num_candidates=50,
    num_dims=2,
    kernel=lambda X, Y: L22(X, Y),
    mode="convex",
    temp=50.0
)

# Forward pass
f_values = model(x_batch)

# Sup/inf transform (returns 3-tuple)
X_opt, g_values, converged = model.sup_transform(
    y_batch, 
    optimizer="lbfgs",
    steps=50,
    lr=1e-3,
    tol=1e-6
)

# Check convergence
if not converged:
    print("Warning: optimization did not converge")
```

### InfConvolution (`inf_convolution.py`)

Differentiable infimal convolution operation using implicit differentiation via the envelope theorem:

`g(y) = inf_x [K(x,y) - f(x)]`

**Features:**
- Implicit differentiation (no backprop through optimization loop)
- Multiple optimizers: LBFGS, Adam, GD
- Early stopping based on relative change
- L2 regularization support
- Returns convergence flag
- Comprehensive test coverage (12 tests)

**Key Insight:**
Gradients are computed efficiently using the envelope theorem:
```
dg/dθ = -df(x*,θ)/dθ
```
where `x*` is the minimizer. No need to compute `dx*/dθ`!

**Usage:**
```python
from models import InfConvolution
import torch.nn as nn

# Define a network f(x)
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)
    
    def forward(self, x):
        return self.fc(x).squeeze()

# Define kernel K(x,y)
def kernel(x, y):
    return 0.5 * ((x - y)**2).sum()

# Compute infimal convolution
f_net = MyNetwork()
y = torch.randn(3)
x_init = torch.zeros(3)

g, converged = InfConvolution.apply(
    y, f_net, kernel, x_init,
    solver_steps=100,
    lr=1.0,
    optimizer="lbfgs",
    lam=0.0,
    tol=1e-6,
    *list(f_net.parameters())
)

# Gradients computed via implicit differentiation
g.backward()
print(f"Gradients: {[p.grad for p in f_net.parameters()]}")
```

## Migration from Old Structure

### Before:
```python
from model import FiniteModel
from tools.inf_convolution import InfConvolution
```

### After:
```python
from models import FiniteModel, InfConvolution
# or
from models import FiniteModel
from models import InfConvolution
```

## Testing

Both components have comprehensive test suites:

```bash
# Test FiniteModel
python -m unittest tests.test_model -v

# Test InfConvolution
python -m unittest tests.test_infconv -v
python -m unittest tests.test_inf_convolution -v

# Test FCOT (uses FiniteModel)
python -m unittest tests.test_fcot -v
```

## Documentation

- **FiniteModel**: See docstrings in `finite_model.py`
- **InfConvolution**: See extensive documentation in `inf_convolution.py` including:
  - Module-level overview with mathematical background
  - Class-level description
  - Detailed forward/backward method documentation with mathematical derivations
  - Complete usage examples

## References

- Envelope Theorem: https://en.wikipedia.org/wiki/Envelope_theorem
- Implicit Differentiation in Optimization: Amos & Kolter (2017), OptNet

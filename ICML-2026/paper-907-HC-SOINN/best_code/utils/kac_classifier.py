"""Core component."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class SplineLinear(nn.Linear):
    """Handle init."""
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        """Handle reset parameters."""
        if self.init_scale == 0:
            nn.init.zeros_(self.weight)
        else:
            nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    """Handle init."""
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: Optional[float] = None,
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        """Handle forward."""
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class KACLayer(nn.Module):
    """Handle init."""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 16,
        use_base_update: bool = False,
        base_activation = None,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_grids = num_grids
        
        self.layernorm = nn.LayerNorm(input_dim)
        
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        
        self.basis_linear = nn.Parameter(torch.zeros([input_dim, num_grids]))
        nn.init.trunc_normal_(self.basis_linear, mean=0, std=spline_weight_init_scale)
        
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, 0)
        
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation or F.silu
            self.base_linear = nn.Linear(input_dim, output_dim)
        else:
            self.base_linear = None

    def forward(self, x, time_benchmark: bool = False):
        """Handle forward."""
        if not time_benchmark:
            normalized_x = self.layernorm(x)
        else:
            normalized_x = x
        
        spline_basis = self.rbf(normalized_x)
        
        flattened = spline_basis.view(*spline_basis.shape[:-2], -1)
        ret = self.spline_linear(flattened)
        
        if self.use_base_update and self.base_linear is not None:
            base_out = self.base_linear(self.base_activation(normalized_x))
            ret = ret + base_out
        
        return ret

    def update(self, new_output_dim: int):
        """Handle update."""
        if new_output_dim < self.output_dim:
            raise ValueError(f"new_output_dim ({new_output_dim}) must be >= current output_dim ({self.output_dim})")
        
        if new_output_dim == self.output_dim:
            return
        
        old_weight = self.spline_linear.weight.data.clone()
        
        new_spline_linear = SplineLinear(
            self.input_dim * self.num_grids, 
            new_output_dim, 
            self.spline_linear.init_scale
        )
        
        new_spline_linear.weight.data[:self.output_dim, :] = old_weight
        
        if new_output_dim > self.output_dim:
            nn.init.trunc_normal_(
                new_spline_linear.weight.data[self.output_dim:, :],
                mean=0,
                std=self.spline_linear.init_scale
            )
        
        self.spline_linear = new_spline_linear
        self.output_dim = new_output_dim

    def extra_repr(self) -> str:
        """Handle extra repr."""
        return f'input_dim={self.input_dim}, output_dim={self.output_dim}, num_grids={self.num_grids}'


class KACClassifier(nn.Module):
    """Handle init."""
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 10,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 16,
        spline_weight_init_scale: float = 0.1,
    ):
        super().__init__()
        self.kac_layer = KACLayer(
            input_dim=input_dim,
            output_dim=output_dim,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            spline_weight_init_scale=spline_weight_init_scale,
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        """Handle forward."""
        return self.kac_layer(x)

    def update(self, new_output_dim: int):
        """Handle update."""
        self.kac_layer.update(new_output_dim)
        self.output_dim = new_output_dim

    @property
    def out_features(self):
        """Handle out features."""
        return self.output_dim

    @property
    def weight(self):
        """Handle weight."""
        return self.kac_layer.spline_linear.weight







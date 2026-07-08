"""Search space definitions for mixed variable optimization."""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Union
from enum import Enum


class VariableType(Enum):
    """Types of variables in the search space."""
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"


@dataclass(frozen=True)
class Dimension:
    """Immutable dimension specification for mixed variables.
    
    Attributes:
        name: Name of the parameter
        type: Type of dimension (continuous, categorical, ordinal)
        bounds: For continuous dimensions, (min, max) tuple
        choices: For categorical/ordinal dimensions, list of valid values
        log_scale: For continuous dimensions, whether to use log scale
        normalize: Whether this dimension should be normalized
    """
    name: str
    type: Union[str, VariableType] = 'continuous'
    bounds: Optional[Tuple[float, float]] = None
    choices: Optional[Union[List[str], List[int], List[float]]] = None
    log_scale: bool = False
    normalize: bool = True
    
    def __post_init__(self):
        """Validate dimension specification."""
        # Convert string type to enum if needed
        if isinstance(self.type, str):
            type_map = {
                'continuous': VariableType.CONTINUOUS,
                'categorical': VariableType.CATEGORICAL,
                'ordinal': VariableType.ORDINAL
            }
            if self.type not in type_map:
                raise ValueError(f"Unknown dimension type: {self.type}")
            object.__setattr__(self, 'type', type_map[self.type])
        
        if self.type == VariableType.CONTINUOUS:
            if self.bounds is None:
                raise ValueError(f"Continuous dimension '{self.name}' must have bounds")
            if self.bounds[0] >= self.bounds[1]:
                raise ValueError(f"Invalid bounds for '{self.name}': {self.bounds}")
            if self.log_scale and self.bounds[0] <= 0:
                raise ValueError(f"Log scale requires positive bounds for '{self.name}'")
        elif self.type in [VariableType.CATEGORICAL, VariableType.ORDINAL]:
            if self.choices is None or len(self.choices) == 0:
                raise ValueError(f"{self.type.value.capitalize()} dimension '{self.name}' must have choices")
            if self.bounds is not None:
                raise ValueError(f"{self.type.value.capitalize()} dimension '{self.name}' should not have bounds")


@dataclass(frozen=True)
class SearchSpace:
    """Immutable search space specification for mixed variables."""
    dimensions: Tuple[Dimension, ...]
    
    def __post_init__(self):
        """Cache dimension indices by type."""
        continuous_dims = []
        categorical_dims = []
        ordinal_dims = []
        
        for i, dim in enumerate(self.dimensions):
            if dim.type == VariableType.CONTINUOUS:
                continuous_dims.append(i)
            elif dim.type == VariableType.CATEGORICAL:
                categorical_dims.append(i)
            elif dim.type == VariableType.ORDINAL:
                ordinal_dims.append(i)
        
        # Store as object attributes (frozen dataclass allows this in __post_init__)
        object.__setattr__(self, '_continuous_dims', continuous_dims)
        object.__setattr__(self, '_categorical_dims', categorical_dims)
        object.__setattr__(self, '_ordinal_dims', ordinal_dims)
    
    @property
    def n_dims(self) -> int:
        """Number of dimensions in the search space."""
        return len(self.dimensions)
    
    @property
    def continuous_dims(self) -> List[int]:
        """Indices of continuous dimensions."""
        return self._continuous_dims
    
    @property
    def categorical_dims(self) -> List[int]:
        """Indices of categorical dimensions."""
        return self._categorical_dims
    
    @property
    def ordinal_dims(self) -> List[int]:
        """Indices of ordinal dimensions."""
        return self._ordinal_dims
    
    @property
    def bounds(self) -> torch.Tensor:
        """Get bounds tensor [2, n_dims] for optimization.
        
        For continuous: normalized bounds [0, 1] if normalize=True
        For categorical: [0, n_choices-1] as integer indices
        For ordinal: [0, 1] normalized to preserve ordering if normalize=True
        """
        lower = []
        upper = []
        
        for dim in self.dimensions:
            if dim.type == VariableType.CONTINUOUS:
                if dim.normalize:
                    lower.append(0.0)
                    upper.append(1.0)
                else:
                    lower.append(dim.bounds[0])
                    upper.append(dim.bounds[1])
            elif dim.type == VariableType.CATEGORICAL:
                lower.append(0)
                upper.append(len(dim.choices) - 1)
            elif dim.type == VariableType.ORDINAL:
                if dim.normalize:
                    lower.append(0.0)
                    upper.append(1.0)
                else:
                    lower.append(0)
                    upper.append(len(dim.choices) - 1)
        
        return torch.tensor([lower, upper], dtype=torch.double)
    
    @property
    def original_bounds(self) -> torch.Tensor:
        """Get original (non-normalized) bounds tensor [2, n_dims]."""
        lower = [dim.bounds[0] for dim in self.dimensions]
        upper = [dim.bounds[1] for dim in self.dimensions]
        return torch.tensor([lower, upper], dtype=torch.double)
    
    def normalize(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize parameters from original space to [0, 1].
        
        Args:
            X: Points in original space [..., n_dims]
            
        Returns:
            Normalized points [..., n_dims]
        """
        X_norm = X.clone()
        orig_bounds = self.original_bounds
        
        for i, dim in enumerate(self.dimensions):
            if dim.normalize:
                lower, upper = orig_bounds[0, i], orig_bounds[1, i]
                X_norm[..., i] = (X[..., i] - lower) / (upper - lower)
        
        return X_norm
    
    def denormalize(self, X: torch.Tensor) -> torch.Tensor:
        """Denormalize parameters from [0, 1] to original space.
        
        Args:
            X: Normalized points [..., n_dims]
            
        Returns:
            Points in original space [..., n_dims]
        """
        X_denorm = X.clone()
        orig_bounds = self.original_bounds
        
        for i, dim in enumerate(self.dimensions):
            if dim.normalize:
                lower, upper = orig_bounds[0, i], orig_bounds[1, i]
                X_denorm[..., i] = X[..., i] * (upper - lower) + lower
        
        return X_denorm
    
    def decode_point(self, x: torch.Tensor) -> Dict[str, Any]:
        """Decode a point to parameter dictionary.
        
        Args:
            x: Single point tensor [n_dims] in the search space coordinate system
            
        Returns:
            Dictionary mapping parameter names to actual values
        """
        if x.dim() != 1 or x.shape[0] != self.n_dims:
            raise ValueError(f"Expected 1D tensor of size {self.n_dims}, got {x.shape}")
        
        params = {}
        
        for i, dim in enumerate(self.dimensions):
            if dim.type == VariableType.CONTINUOUS:
                if dim.log_scale:
                    log_min = torch.log(torch.tensor(dim.bounds[0]))
                    log_max = torch.log(torch.tensor(dim.bounds[1]))
                    if dim.normalize:
                        # Denormalize then exp
                        log_val = x[i] * (log_max - log_min) + log_min
                        value = torch.exp(log_val).item()
                    else:
                        # Already in original space
                        value = x[i].item()
                else:
                    if dim.normalize:
                        # Denormalize from [0, 1]
                        value = x[i].item() * (dim.bounds[1] - dim.bounds[0]) + dim.bounds[0]
                    else:
                        value = x[i].item()
                params[dim.name] = value
                
            elif dim.type == VariableType.CATEGORICAL:
                # Index to choice
                idx = int(torch.round(x[i]).item())
                idx = max(0, min(idx, len(dim.choices) - 1))
                params[dim.name] = dim.choices[idx]
                
            elif dim.type == VariableType.ORDINAL:
                if dim.normalize:
                    # Denormalize from [0, 1] to index
                    max_idx = len(dim.choices) - 1
                    idx = int(torch.round(x[i] * max_idx).item())
                else:
                    idx = int(torch.round(x[i]).item())
                idx = max(0, min(idx, len(dim.choices) - 1))
                params[dim.name] = dim.choices[idx]
        
        return params
    
    def encode_point(self, params: Dict[str, Any]) -> torch.Tensor:
        """Encode parameter dictionary to tensor.
        
        Args:
            params: Dictionary mapping parameter names to values
            
        Returns:
            Single point tensor [n_dims] in the search space coordinate system
        """
        if len(params) != self.n_dims:
            raise ValueError(f"Expected {self.n_dims} parameters, got {len(params)}")
        
        x = torch.zeros(self.n_dims, dtype=torch.double)
        
        for i, dim in enumerate(self.dimensions):
            if dim.name not in params:
                raise ValueError(f"Missing parameter '{dim.name}'")
            
            value = params[dim.name]
            
            if dim.type == VariableType.CONTINUOUS:
                if dim.log_scale:
                    # Apply log transform then normalize
                    log_min = torch.log(torch.tensor(dim.bounds[0]))
                    log_max = torch.log(torch.tensor(dim.bounds[1]))
                    log_val = torch.log(torch.tensor(value))
                    if dim.normalize:
                        x[i] = (log_val - log_min) / (log_max - log_min)
                    else:
                        x[i] = torch.exp(log_val).item()  # Keep in original space
                else:
                    if dim.normalize:
                        x[i] = (value - dim.bounds[0]) / (dim.bounds[1] - dim.bounds[0])
                    else:
                        x[i] = value
                        
            elif dim.type == VariableType.CATEGORICAL:
                # Map to index
                try:
                    x[i] = dim.choices.index(value)
                except ValueError:
                    raise ValueError(f"Invalid choice '{value}' for categorical '{dim.name}'")
                    
            elif dim.type == VariableType.ORDINAL:
                # Map to index then optionally normalize
                try:
                    idx = dim.choices.index(value)
                except ValueError:
                    raise ValueError(f"Invalid choice '{value}' for ordinal '{dim.name}'")
                
                if dim.normalize:
                    # Normalize to [0, 1] to preserve ordering
                    max_idx = len(dim.choices) - 1
                    x[i] = idx / max_idx if max_idx > 0 else 0.0
                else:
                    x[i] = idx
        
        return x
    
    def get_fixed_features_list(self) -> List[Dict[int, float]]:
        """Generate all categorical combinations for mixed optimization.
        
        Returns list of dicts mapping dimension index to fixed value.
        Only includes categorical dimensions (not ordinals).
        """
        if not self.categorical_dims:
            return [{}]  # No categoricals to fix
        
        # Generate all combinations
        import itertools
        
        categorical_choices = []
        for dim_idx in self.categorical_dims:
            dim = self.dimensions[dim_idx]
            categorical_choices.append(range(len(dim.choices)))
        
        combinations = []
        for combo in itertools.product(*categorical_choices):
            fixed_features = {}
            for i, dim_idx in enumerate(self.categorical_dims):
                fixed_features[dim_idx] = float(combo[i])
            combinations.append(fixed_features)
        
        return combinations
    
    def get_discrete_combinations(self, include_ordinals: bool = True) -> List[Dict[int, float]]:
        """Get all discrete combinations.
        
        Args:
            include_ordinals: Whether to include ordinal variables
            
        Returns:
            List of dicts mapping dimension index to value
        """
        import itertools
        
        discrete_dims = self.categorical_dims.copy()
        if include_ordinals:
            discrete_dims.extend(self.ordinal_dims)
        
        if not discrete_dims:
            return [{}]
        
        discrete_dims.sort()  # Keep consistent ordering
        
        choices_per_dim = []
        for dim_idx in discrete_dims:
            dim = self.dimensions[dim_idx]
            if dim.type == VariableType.CATEGORICAL:
                choices_per_dim.append([float(i) for i in range(len(dim.choices))])
            elif dim.type == VariableType.ORDINAL:
                if dim.normalize:
                    # Normalized ordinal values
                    max_idx = len(dim.choices) - 1
                    choices_per_dim.append([i / max_idx if max_idx > 0 else 0.0 
                                           for i in range(len(dim.choices))])
                else:
                    choices_per_dim.append([float(i) for i in range(len(dim.choices))])
        
        combinations = []
        for combo in itertools.product(*choices_per_dim):
            fixed_features = {}
            for i, dim_idx in enumerate(discrete_dims):
                fixed_features[dim_idx] = combo[i]
            combinations.append(fixed_features)
        
        return combinations
    
    @staticmethod
    def from_bounds(bounds: torch.Tensor, normalize: bool = False) -> 'SearchSpace':
        """Create a simple continuous search space from bounds tensor.
        
        Args:
            bounds: Bounds tensor [2, n_dims]
            normalize: Whether to normalize dimensions to [0, 1]
            
        Returns:
            SearchSpace instance
        """
        if bounds.dim() != 2 or bounds.shape[0] != 2:
            raise ValueError(f"Expected bounds shape [2, n_dims], got {bounds.shape}")
        
        n_dims = bounds.shape[1]
        dimensions = tuple(
            Dimension(
                name=f'x{i}',
                type='continuous',
                bounds=(float(bounds[0, i]), float(bounds[1, i])),
                normalize=normalize
            )
            for i in range(n_dims)
        )
        return SearchSpace(dimensions)

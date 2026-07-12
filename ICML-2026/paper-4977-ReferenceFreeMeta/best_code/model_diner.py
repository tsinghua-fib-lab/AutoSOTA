"""
DINER Model - Hash Encoding + MLP for MRI Reconstruction
Implements the DINER backbone as described in the paper:
- Multi-resolution hash encoding (Instant-NGP style)
- MLP with 2 hidden layers, 16 neurons, ReLU activation

Reference: DINER (Xie et al., 2023) and Instant-NGP (Müller et al., 2022)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class HashEncoding(nn.Module):
    """
    Multi-resolution Hash Encoding for 2D coordinates.
    Implements the hash encoding from Instant-NGP.
    """
    def __init__(
        self,
        n_input_dims: int = 2,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 12,
        per_level_scale: float = 2.0,
    ):
        super().__init__()
        self.n_input_dims = n_input_dims
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale
        
        # Total output dimension
        self.output_dim = n_levels * n_features_per_level
        
        # Hash table size
        self.hash_table_size = 2 ** log2_hashmap_size
        
        # Create hash table embeddings for each level
        # We use separate embedding tables for each level (simplified vs shared table)
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.hash_table_size, n_features_per_level)
            for _ in range(n_levels)
        ])
        
        # Initialize embeddings uniformly
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)
        
        # Compute resolution for each level
        self.resolutions = []
        for i in range(n_levels):
            resolution = int(base_resolution * (per_level_scale ** i))
            self.resolutions.append(resolution)
        
        # Primes for spatial hashing
        self.primes = torch.tensor([1, 2654435761], dtype=torch.int64)
    
    def _hash_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Hash coordinates to table indices using spatial hashing.
        coords: [N, n_input_dims] integer coordinates
        Returns: [N] hash indices
        """
        result = torch.zeros(coords.shape[0], dtype=torch.int64, device=coords.device)
        for d in range(self.n_input_dims):
            result = torch.bitwise_xor(
                result,
                coords[:, d].long() * self.primes[d].item()
            )
        return result % self.hash_table_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, 2] input coordinates in [0, 1] range
        Returns:
            features: [N, n_levels * n_features_per_level]
        """
        N = x.shape[0]
        all_features = []
        
        for level in range(self.n_levels):
            resolution = self.resolutions[level]
            
            # Scale coordinates to current resolution grid
            # Map from [0,1] to [0, resolution-1]
            scaled = x * (resolution - 1)
            
            # Get grid cell indices (floor)
            grid_cell = scaled.floor().long()
            
            # Compute fractional positions for interpolation
            frac = scaled - scaled.floor()
            
            # Get the 4 corners of the grid cell (for 2D bilinear interpolation)
            # Corner 0: (x0, y0), Corner 1: (x1, y0), Corner 2: (x0, y1), Corner 3: (x1, y1)
            corners = []
            corners.append(grid_cell)                              # (floor_x, floor_y)
            corners.append(grid_cell + torch.tensor([1, 0], device=x.device))  # (ceil_x, floor_y)
            corners.append(grid_cell + torch.tensor([0, 1], device=x.device))  # (floor_x, ceil_y)
            corners.append(grid_cell + torch.tensor([1, 1], device=x.device))  # (ceil_x, ceil_y)
            
            # Clamp to valid range
            for i in range(4):
                corners[i] = corners[i].clamp(0, resolution - 1)
            
            # Hash each corner and get features
            corner_features = []
            for corner in corners:
                hash_idx = self._hash_coords(corner)
                feat = self.embeddings[level](hash_idx)  # [N, n_features_per_level]
                corner_features.append(feat)
            
            # Bilinear interpolation weights
            wx = frac[:, 0:1]  # [N, 1]
            wy = frac[:, 1:2]  # [N, 1]
            
            # Interpolate
            feat_00 = corner_features[0]
            feat_10 = corner_features[1]
            feat_01 = corner_features[2]
            feat_11 = corner_features[3]
            
            # f = (1-wx)*(1-wy)*f00 + wx*(1-wy)*f10 + (1-wx)*wy*f01 + wx*wy*f11
            interp = (
                (1 - wx) * (1 - wy) * feat_00 +
                wx * (1 - wy) * feat_10 +
                (1 - wx) * wy * feat_01 +
                wx * wy * feat_11
            )
            
            all_features.append(interp)
        
        # Concatenate features from all levels
        return torch.cat(all_features, dim=-1)


class DinerMLP(nn.Module):
    """
    Small MLP for DINER: 2 hidden layers, 16 neurons, ReLU activation.
    """
    def __init__(self, input_dim: int, output_dim: int = 1, 
                 n_neurons: int = 16, n_hidden_layers: int = 2,
                 activation: str = "ReLU"):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        # Hidden layers
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(current_dim, n_neurons))
            if activation == "ReLU":
                layers.append(nn.ReLU())
            elif activation == "Sine":
                # SIREN-style initialization for sine
                lin = nn.Linear(current_dim, n_neurons)
                if i == 0:
                    nn.init.uniform_(lin.weight, -1/current_dim, 1/current_dim)
                else:
                    nn.init.uniform_(lin.weight, -np.sqrt(6/current_dim)/30, np.sqrt(6/current_dim)/30)
                layers.append(lin)
                # We add a custom sine activation
            current_dim = n_neurons
        
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "Sine":
            # Apply sine activation manually
            for i, layer in enumerate(self.net):
                x = layer(x)
                if i < len(self.net) - 1 and i % 2 == 0:  # Every other layer (after linear)
                    x = torch.sin(30.0 * x)
            return x
        return self.net(x)


class DinerModel(nn.Module):
    """
    DINER model for MRI reconstruction.
    Uses HashEncoding + MLP for both magnitude and phase branches.
    Architecture: MLP with 2 hidden layers, 16 neurons, ReLU
    """
    def __init__(
        self,
        encoding_config: dict = None,
        network_config: dict = None,
    ):
        super().__init__()
        
        # Default encoding config (matching the training script)
        if encoding_config is None:
            encoding_config = {
                "otype": "Grid",
                "type": "Hash",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 12,
                "per_level_scale": 2,
                "interpolation": "Linear"
            }
        
        # Default network config
        if network_config is None:
            network_config = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 16,
                "n_hidden_layers": 2
            }
        
        self.encoding_config = encoding_config
        self.network_config = network_config
        
        # Create hash encoding (shared between mag and phase branches)
        self.encoding = HashEncoding(
            n_input_dims=2,
            n_levels=encoding_config.get("n_levels", 16),
            n_features_per_level=encoding_config.get("n_features_per_level", 2),
            log2_hashmap_size=encoding_config.get("log2_hashmap_size", 19),
            base_resolution=encoding_config.get("base_resolution", 12),
            per_level_scale=encoding_config.get("per_level_scale", 2.0),
        )
        
        encoding_dim = self.encoding.output_dim
        n_neurons = network_config.get("n_neurons", 16)
        n_hidden_layers = network_config.get("n_hidden_layers", 2)
        activation = network_config.get("activation", "ReLU")
        
        # Magnitude branch MLP
        self.model_mag = DinerMLP(
            input_dim=encoding_dim,
            output_dim=1,
            n_neurons=n_neurons,
            n_hidden_layers=n_hidden_layers,
            activation=activation,
        )
        
        # Phase branch MLP
        self.model_phi = DinerMLP(
            input_dim=encoding_dim,
            output_dim=1,
            n_neurons=n_neurons,
            n_hidden_layers=n_hidden_layers,
            activation=activation,
        )
    
    def forward(self, coordinates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagation
        Args:
            coordinates: Coordinate input [N, 2]
        Returns:
            pre_intensity_mag: Predicted magnitude [N, 1]
            pre_intensity_phi: Predicted phase [N, 1]
        """
        # Apply hash encoding
        encoded = self.encoding(coordinates)  # [N, encoding_dim]
        
        # Pass through magnitude and phase branches
        pre_intensity_mag = self.model_mag(encoded)
        pre_intensity_phi = self.model_phi(encoded)
        
        return pre_intensity_mag.float(), pre_intensity_phi.float()
    
    def get_complex_image(
        self,
        coordinates: torch.Tensor,
        shape: Tuple[int, int]
    ) -> torch.Tensor:
        """Get complex image from coordinates"""
        H, W = shape
        pre_intensity_mag, pre_intensity_phi = self.forward(coordinates)
        
        pre_intensity_mag = pre_intensity_mag.view(H, W, 1)
        pre_intensity_phi = pre_intensity_phi.view(H, W, 1)
        
        pre_intensity = torch.complex(pre_intensity_mag, pre_intensity_phi)
        return pre_intensity

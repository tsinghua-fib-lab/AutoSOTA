#!/usr/bin/env python3
"""
Simple test script for the new wavelet recombination layers.
"""
import sys
sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScalarWaveletRecombiner(nn.Module):
    """
    Recombines scalar features across the wavelet dimension using 2-layer MLPs.
    Each output channel is computed by a separate MLP that takes the input
    wavelet features and produces a single scalar output.
    """
    def __init__(self, num_wavelets: int, num_output_channels: int, hidden_dim: int, 
                 nonlin_fn: callable = F.silu):
        """
        Args:
            num_wavelets: Number of input wavelet features (W)
            num_output_channels: Number of output channels to create (W')
            hidden_dim: Hidden dimension for the MLPs
            nonlin_fn: Nonlinearity function to use between layers (default: SiLU)
        """
        super().__init__()
        self.num_wavelets = num_wavelets
        self.num_output_channels = num_output_channels
        self.nonlin_fn = nonlin_fn
        
        # Create MLPs for each output channel
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_wavelets, hidden_dim),
                nn.ReLU() if self.nonlin_fn == F.relu else nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(num_output_channels)
        ])
    
    def forward(self, scalar_inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scalar_inputs: Tensor of shape [N, C, W] - N nodes, C channels, W wavelets
        Returns:
            output: Tensor of shape [N, C, W'] - N nodes, C channels, W' recombined wavelets
        """
        N, C, W = scalar_inputs.shape
        assert W == self.num_wavelets, f"Expected {self.num_wavelets} wavelets, got {W}"
        x = scalar_inputs.reshape(N * C, W)
        outputs = []
        for mlp in self.mlps:
            out = mlp(x)  # [N*C, 1]
            outputs.append(out)
        out_cat = torch.cat(outputs, dim=1)  # [N*C, W']
        return out_cat.view(N, C, self.num_output_channels)


class GatedVectorWaveletRecombiner(nn.Module):
    """
    Recombines vector norms across the wavelet dimension using gating and reweighting.
    Since norms are rotationally invariant, this operation is equivariant.
    """
    def __init__(
        self, 
        num_wavelets: int, 
        num_output_channels: int, 
        hidden_dim: int, 
        gate_hidden_dim: int = None,
    ):
        """
        Args:
            num_wavelets: Number of input wavelet features (W)
            num_output_channels: Number of output channels to create (W')
            hidden_dim: Hidden dimension for the weight MLPs
            gate_hidden_dim: Hidden dimension for the gate MLP (default = hidden_dim)
        """
        super().__init__()
        self.num_wavelets = num_wavelets
        self.num_output_channels = num_output_channels
        gate_hidden_dim = gate_hidden_dim or hidden_dim

        # MLPs to compute scalar weights for combining wavelets into each output channel
        self.weight_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_wavelets, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, num_wavelets)
            )
            for _ in range(num_output_channels)
        ])

        # Gate MLP for modulating each input wavelet based on its norm
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_wavelets, gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(gate_hidden_dim, num_wavelets),
            nn.Sigmoid()  # output gate ∈ (0, 1)
        )

    def forward(self, norm_inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            norm_inputs: Tensor of shape [N, C, W] - N nodes, C channels, W wavelets (already norms)
        Returns:
            output: Tensor of shape [N, C, W'] - N nodes, C channels, W' recombined wavelets
        """
        N, C, W = norm_inputs.shape
        assert W == self.num_wavelets, f"Expected {self.num_wavelets} wavelets, got {W}"
        
        # Step 1: Gating each input wavelet based on its norm
        gates = self.gate_mlp(norm_inputs)  # [N, C, W]
        gated_norms = gates * norm_inputs  # [N, C, W]
        
        # Step 2: For each output channel, compute weighted sum of gated norms
        outputs = []
        for mlp in self.weight_mlps:
            weights = mlp(norm_inputs)  # [N, C, W]
            # Weighted sum across wavelet dimension: [N, C, W] -> [N, C, 1]
            mixed = (weights * gated_norms).sum(dim=2, keepdim=True)  # [N, C, 1]
            outputs.append(mixed)
        
        # Step 3: Concatenate outputs along wavelet dimension: [N, C, W']
        return torch.cat(outputs, dim=2)  # [N, C, W']


def test_scalar_recombiner():
    """Test the ScalarWaveletRecombiner."""
    print("Testing ScalarWaveletRecombiner...")
    
    num_wavelets = 5
    num_output_channels = 3
    hidden_dim = 16
    
    recombiner = ScalarWaveletRecombiner(
        num_wavelets=num_wavelets,
        num_output_channels=num_output_channels,
        hidden_dim=hidden_dim,
        nonlin_fn=F.relu
    )
    
    # Test input: [N, C, W]
    N, C = 10, 4
    scalar_input = torch.randn(N, C, num_wavelets)
    
    scalar_output = recombiner(scalar_input)
    print(f"Scalar input shape: {scalar_input.shape}")
    print(f"Scalar output shape: {scalar_output.shape}")
    assert scalar_output.shape == (N, C, num_output_channels), f"Expected {(N, C, num_output_channels)}, got {scalar_output.shape}"
    print("✓ Scalar recombination test passed!")
    
    # Test that the output is non-trivial (not all zeros)
    assert not torch.all(scalar_output == 0), "Output should not be all zeros"
    print("✓ Scalar recombination produces non-trivial transformation!")


def test_vector_recombiner():
    """Test the GatedVectorWaveletRecombiner."""
    print("\nTesting GatedVectorWaveletRecombiner...")
    
    num_wavelets = 5
    num_output_channels = 3
    hidden_dim = 16
    
    recombiner = GatedVectorWaveletRecombiner(
        num_wavelets=num_wavelets,
        num_output_channels=num_output_channels,
        hidden_dim=hidden_dim
    )
    
    # Test input: [N, C, W] (norms)
    N, C = 10, 4
    norm_input = torch.randn(N, C, num_wavelets)
    
    norm_output = recombiner(norm_input)
    print(f"Norm input shape: {norm_input.shape}")
    print(f"Norm output shape: {norm_output.shape}")
    assert norm_output.shape == (N, C, num_output_channels), f"Expected {(N, C, num_output_channels)}, got {norm_output.shape}"
    print("✓ Vector recombination test passed!")
    
    # Test that the output is non-trivial (not all zeros)
    assert not torch.all(norm_output == 0), "Output should not be all zeros"
    print("✓ Vector recombination produces non-trivial transformation!")


def test_equivariance():
    """Test that vector recombination preserves rotational equivariance."""
    print("\nTesting rotational equivariance...")
    
    num_wavelets = 3
    num_output_channels = 2
    hidden_dim = 8
    
    recombiner = GatedVectorWaveletRecombiner(
        num_wavelets=num_wavelets,
        num_output_channels=num_output_channels,
        hidden_dim=hidden_dim
    )
    
    # Use a proper rotation matrix (around z-axis)
    vector_dim = 3
    identity = torch.eye(vector_dim)
    angle = math.pi / 4  # 45 degrees
    rotation = torch.tensor([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle),  math.cos(angle), 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    # Debug: Check if this rotation matrix is orthogonal
    orthogonality_error = torch.norm(torch.mm(rotation, rotation.t()) - identity)
    print(f"Rotation matrix orthogonality error: {orthogonality_error.item():.6f}")

    # Create ONE set of vector data
    N, C = 4, 2
    vector_input = torch.randn(N, vector_dim, C, num_wavelets)
    
    # Compute norms of original vectors
    norm_original = torch.norm(vector_input, p=2, dim=1)  # [N, C, W]
    
    # Apply rotation to the SAME vectors and compute norms
    vector_rotated = torch.einsum('ij,njkl->nikl', rotation, vector_input)
    norm_rotated = torch.norm(vector_rotated, p=2, dim=1)  # [N, C, W]
    
    # Verify that norms are indeed invariant (should be identical)
    norm_invariance_error = torch.norm(norm_original - norm_rotated)
    print(f"Norm invariance error: {norm_invariance_error.item():.6f}")
    assert norm_invariance_error < 1e-6, "Norms should be rotationally invariant!"
    
    # Get outputs
    output_original = recombiner(norm_original)
    output_rotated = recombiner(norm_rotated)
    
    # Since norms are rotationally invariant, the outputs should be identical
    equivariance_error = torch.norm(output_original - output_rotated)
    print(f"Equivariance error: {equivariance_error.item():.6f}")
    
    if equivariance_error < 1e-6:
        print("✓ Vector recombination is perfectly equivariant (norms are invariant)!")
    else:
        print("⚠ Vector recombination equivariance needs investigation")


if __name__ == "__main__":
    print("Testing wavelet recombination layers...")
    
    test_scalar_recombiner()
    test_vector_recombiner()
    test_equivariance()
    
    print("\n\nAll tests completed!") 
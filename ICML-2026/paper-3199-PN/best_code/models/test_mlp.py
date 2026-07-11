#
# Software Name : learning-parities-with-product-networks
# SPDX-FileCopyrightText: Copyright (c) 2026 Orange S.A.
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License .,
# see the "LICENSE.md" file for more details or https://opensource.org/licenses/MIT
#
# Author: Guillaume Larue, guillaume.larue@orange.com
# Software description: Source code of the paper "Learning High-Dimensional Parity Functions with Product Networks"
#

"""
PyTorch tests for ParallelMLP models.

These tests verify that the ParallelMLP implementation produces correct outputs
for single-layer and multi-layer configurations.
"""

import pytest
import torch
import torch.nn as nn

from . import ParallelMLP


def test_parallel_mlp_single_layer():
    """Test single-layer ParallelMLP with explicit weights and biases."""
    
    # Create model with no hidden layers (direct input to output)
    model = ParallelMLP(
        n_inputs=3,
        n_outputs=4,
        hidden_sizes=[2],  # No hidden layers
        activation=None,
        output_activation=None,
        use_bias=True,
    )
    
    # Set explicit weights for layer 0
    # Shape: [n_outputs=4, output_features=2, input_features=3]
    weights = torch.Tensor([
        [
            [1, 0, 1],
            [0, 1, 0],
        ],
        [
            [0, 1, 1],
            [0, 0, 0],
        ],
        [
            [1, 1, 1],
            [0, 0, 1],
        ],
        [
            [1, 0, 0],
            [0, 1, 1],
        ],
    ])
    
    # Shape: [n_outputs=4, output_features=2]
    biases = torch.Tensor([
        [0, 2],
        [4, 6],
        [8, 10],
        [12, 14],
    ])
    
    model.set_weights(0, weights, biases)
    
    # Test inputs: [batch_size=5, input_features=3]
    inputs = torch.Tensor([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
        [0, 1, 0],
    ])
    
    # Expected output: [batch_size=5, n_outputs=4, output_features=2]
    expected_output = torch.Tensor([
        [
            [2, 2],
            [5, 6],
            [10, 11],
            [13, 15]
        ],
        [
            [0, 3],
            [5, 6],
            [9, 10],
            [12, 15],
        ],
        [
            [2, 3],
            [6, 6],
            [11, 11],
            [13, 16]
        ],
        [
            [0, 2],
            [4, 6],
            [8, 10],
            [12, 14],
        ],
        [
            [0, 3],
            [5, 6],
            [9, 10],
            [12, 15],
        ],
    ])
    
    # Forward pass
    output = model(inputs)
    
    # Verify output shape
    assert output.shape == expected_output.shape, \
        f"Output shape mismatch: expected {expected_output.shape}, got {output.shape}"
    
    # Verify each element
    batch_size = 5
    parallel_outputs = 4
    for i in range(batch_size):
        for j in range(parallel_outputs):
            assert torch.allclose(output[i, j], expected_output[i, j], atol=1e-5), \
                f"Mismatch at batch {i}, parallel {j}: got {output[i, j]}, expected {expected_output[i, j]}"


def test_parallel_mlp_two_layers():
    """Test two-layer ParallelMLP with explicit weights and biases."""
    
    # Create model with one hidden layer
    model = ParallelMLP(
        n_inputs=3,
        n_outputs=4,
        hidden_sizes=[2,1],  # One hidden layer with 2 features
        use_bias=True,
    )
    
    # LAYER 1: Input -> Hidden
    # Shape: [n_outputs=4, h1_features=2, input_features=3]
    weights_l1 = torch.Tensor([
        [
            [1, 0, 1],
            [0, 1, 0],
        ],
        [
            [0, 1, 1],
            [0, 0, 0],
        ],
        [
            [1, 1, 1],
            [0, 0, 1],
        ],
        [
            [1, 0, 0],
            [0, 1, 1],
        ],
    ])
    
    # Shape: [n_outputs=4, h1_features=2]
    biases_l1 = torch.Tensor([
        [0, 2],
        [4, 6],
        [8, 10],
        [12, 14],
    ])
    
    # LAYER 2: Hidden -> Output
    # Shape: [n_outputs=4, output_features=1, h1_features=2]
    weights_l2 = torch.Tensor([
        [
            [1, 1],
        ],
        [
            [0, 1],
        ],
        [
            [1, 0],
        ],
        [
            [1, 0],
        ],
    ])
    
    # Shape: [n_outputs=4, output_features=1]
    biases_l2 = torch.Tensor([
        [0],
        [6],
        [10],
        [12],
    ])
    
    model.set_weights(0, weights_l1, biases_l1)
    model.set_weights(1, weights_l2, biases_l2)
    
    # Test inputs: [batch_size=5, input_features=3]
    inputs = torch.Tensor([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
        [0, 1, 0],
    ])
    
    # Expected output: [batch_size=5, n_outputs=4, output_features=1]
    expected_output = torch.Tensor([
        [
            [4],
            [12],
            [20],
            [25]
        ],
        [
            [3],
            [12],
            [19],
            [24],
        ],
        [
            [5],
            [12],
            [21],
            [25]
        ],
        [
            [2],
            [12],
            [18],
            [24],
        ],
        [
            [3],
            [12],
            [19],
            [24],
        ],
    ])
    
    # Forward pass
    output = model(inputs)
    
    # Verify output shape
    assert output.shape == expected_output.shape, \
        f"Output shape mismatch: expected {expected_output.shape}, got {output.shape}"
    
    # Verify each element
    batch_size = 5
    parallel_outputs = 4
    for i in range(batch_size):
        for j in range(parallel_outputs):
            assert torch.allclose(output[i, j], expected_output[i, j], atol=1e-5), \
                f"Mismatch at batch {i}, parallel {j}: got {output[i, j]}, expected {expected_output[i, j]}"


def test_parallel_mlp_random():
    """Test ParallelMLP with random inputs and weights."""
    n_inputs = 20
    n_outputs = 30 # Parallel outputs - each output is a separate MLP with its own weights and biases
    n_samples = 100
    hidden_sizes = (64, 512, 32) # Hidden layer sizes for each parallel MLP (same for all)
    
    # Create model
    model = ParallelMLP(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        hidden_sizes=hidden_sizes,
        activation=nn.ReLU(),
        use_bias=True,
    )
    
    # Random inputs
    inputs = torch.randn(n_samples, n_inputs)
    
    # Forward pass
    output = model(inputs)
    
    # Check output shape
    expected_shape = (n_samples, n_outputs, hidden_sizes[-1])  # [100, 30, 32]
    assert output.shape == expected_shape, \
        f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
    
    # Check that output is finite
    assert torch.all(torch.isfinite(output)), "Output contains NaN or Inf values"



def test_parallel_mlp_gradient_flow():
    """Test that gradients flow through ParallelMLP."""
    n_inputs = 10
    n_outputs = 5
    n_samples = 32
    
    # Create model
    model = ParallelMLP(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        hidden_sizes=(16, 8),
        activation=nn.ReLU(),
        use_bias=True,
    )
    
    # Random inputs and targets
    inputs = torch.randn(n_samples, n_inputs)
    targets = torch.randn(n_samples, n_outputs, 1)
    
    # Forward pass
    output = model(inputs)
    
    # Compute loss
    loss = torch.sum((output - targets) ** 2)
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist and are non-zero
    for layer_idx, weight in enumerate(model.weights):
        assert weight.grad is not None, f"No gradient for layer {layer_idx} weights"
        grad_norm = torch.norm(weight.grad)
        assert grad_norm > 0, f"Zero gradient for layer {layer_idx} weights"
    
    if model.use_bias:
        for layer_idx, bias in enumerate(model.biases):
            assert bias.grad is not None, f"No gradient for layer {layer_idx} biases"


def test_parallel_mlp_training():
    """Test that ParallelMLP can be trained."""
    n_inputs = 10
    n_outputs = 5
    n_samples = 128
    
    # Create model
    model = ParallelMLP(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        hidden_sizes=(32, 16),
        activation=nn.ReLU(),
        use_bias=True,
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Random inputs and targets
    inputs = torch.randn(n_samples, n_inputs)
    targets = torch.randn(n_samples, n_outputs, 1)
    
    # Initial loss
    output = model(inputs)
    initial_loss = torch.sum((output - targets) ** 2).item()
    
    # Training steps
    for _ in range(100):
        optimizer.zero_grad()
        output = model(inputs)
        loss = torch.sum((output - targets) ** 2)
        loss.backward()
        optimizer.step()
    
    # Final loss
    output = model(inputs)
    final_loss = torch.sum((output - targets) ** 2).item()
    
    # Loss should decrease
    assert final_loss < initial_loss, \
        f"Loss did not decrease: {initial_loss} -> {final_loss}"



if __name__ == "__main__":
    pytest.main([__file__, "-v"])

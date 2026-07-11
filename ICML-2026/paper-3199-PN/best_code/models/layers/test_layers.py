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
PyTorch tests for product layers.

These tests verify that the PyTorch implementation produces the correct layer XOR logic
"""

import pytest
import torch

from . import BinarySymmetricChannelLayer
from . import BinaryProductLayer, MultiBinaryProductLayer

def test_bsc_no_noise():
    """Test BSC layer with no noise (p_e=0) - output should equal input."""
    bsc = BinarySymmetricChannelLayer(p_e=0.0)
    
    # Generate random binary inputs
    inputs = torch.randint(0, 2, (1000, 1000), dtype=torch.float32)
    
    # Apply BSC
    outputs = bsc(inputs)
    
    # Verify no changes
    assert torch.equal(inputs, outputs), "BSC with p_e=0 should not modify inputs"


def test_bsc_p_e():
    """Test BSC layer with various error probabilities."""
    bsc = BinarySymmetricChannelLayer(p_e=0.0)
    estimated_p_e = []
    target_p_e = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for p_e in target_p_e:
        bsc.set_error_probability(p_e)
        
        # Generate random binary inputs
        inputs = torch.randint(0, 2, (1000, 1000), dtype=torch.float32)
        
        # Apply BSC
        outputs = bsc(inputs)
        
        # Calculate empirical error rate
        errors = torch.abs(outputs - inputs)
        estimated_p_e.append(errors.mean().item())
    
    # Verify error rates match targets (within tolerance)
    for est, target in zip(estimated_p_e, target_p_e):
        assert abs(est - target) < 0.01, f"Estimated p_e={est:.4f} differs from target={target}"



def test_product_layer():
    """Test MultiBinaryProductLayer."""
    
    # Create layer
    layer = MultiBinaryProductLayer(
        n_outputs=10,
        use_gaussian_init=True,
        gaussian_mean=0.5,
        gaussian_std=0.2,
    )
    
    # Test forward pass
    batch_size = 32
    n_inputs = 5
    inputs = torch.zeros(batch_size, n_inputs)
    
    outputs = layer(inputs)
    
    # Check output shape
    assert outputs.shape == (batch_size, 10), f"Expected (32, 10), got {outputs.shape}"
    
    # Check output range [0, 1]
    assert torch.all(outputs >= 0) and torch.all(outputs <= 1), "Outputs not in [0,1]"
    
    # Check weights initialized
    assert layer.product_weights is not None, "Weights not initialized"
    assert layer.product_weights.shape == (n_inputs, 10), f"Wrong weight shape: {layer.product_weights.shape}"


def test_bsc_channel_p_e_change():
    """Test BinarySymmetricChannelLayer."""
    
    # Create channel with p_e=0 (no errors)
    channel = BinarySymmetricChannelLayer(p_e=0.0)
    
    inputs = torch.zeros(100, 10)
    outputs = channel(inputs)
    
    # With p_e=0, outputs should equal inputs
    assert torch.all(outputs == inputs), "BSC with p_e=0 should not flip bits"
    
    # Test with p_e=0.5 (maximum errors)
    channel.set_error_probability(0.5)
    outputs = channel(torch.zeros(10000, 10))
    
    # Approximately 50% should be flipped
    flip_rate = torch.mean((outputs != 0).float()).item()
    assert 0.45 < flip_rate < 0.55, f"Expected ~0.5 flip rate, got {flip_rate}"



def test_binary_product_layer():
    """Test single-output binary product layer for XOR computation."""
    n_inputs = 20
    n_samples = 100
    
    # Generate random binary XOR parameters
    xor_parameters = torch.randint(0, 2, (n_inputs,), dtype=torch.float32)
    
    # Generate random binary inputs
    inputs = torch.randint(0, 2, (n_samples, n_inputs), dtype=torch.float32)
    
    # Create layer (with uniform init to match TF behavior initially)
    bpl = BinaryProductLayer(n_inputs=n_inputs, use_gaussian_init=False)
    
    # Initialize with first forward pass
    _ = bpl(inputs)
    
    # Set XOR parameters
    bpl.set_xor_parameters(xor_parameters)
    
    # Compute XOR using layer
    xor_bpl = bpl(inputs)
    
    # Compute expected XOR (binary dot product mod 2)
    xor_target = (inputs @ xor_parameters.unsqueeze(1)).squeeze(1) % 2
    
    # Verify outputs match expected XOR
    assert torch.allclose(xor_target, xor_bpl, atol=1e-5), \
        "Binary product layer output should match XOR operation"


def test_binary_product_layer_explicit_cases():
    """Test BinaryProductLayer module with explicit XOR truth table cases."""
    
    # Test case 1: XOR parameters [0, 0] -> always 0
    bpl = BinaryProductLayer(n_inputs=2, use_gaussian_init=False)
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    _ = bpl(inputs)  # Initialize
    bpl.set_xor_parameters(torch.tensor([0, 0], dtype=torch.float32))
    output = bpl(inputs)
    expected = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-5), "BinaryProductLayer XOR [0,0] failed"
    
    # Test case 2: XOR parameters [0, 1] -> copy second input
    bpl = BinaryProductLayer(n_inputs=2, use_gaussian_init=False)
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    _ = bpl(inputs)  # Initialize
    bpl.set_xor_parameters(torch.tensor([0, 1], dtype=torch.float32))
    output = bpl(inputs)
    expected = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-5), "BinaryProductLayer XOR [0,1] failed"
    
    # Test case 3: XOR parameters [1, 0] -> copy first input
    bpl = BinaryProductLayer(n_inputs=2, use_gaussian_init=False)
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    _ = bpl(inputs)  # Initialize
    bpl.set_xor_parameters(torch.tensor([1, 0], dtype=torch.float32))
    output = bpl(inputs)
    expected = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-5), "BinaryProductLayer XOR [1,0] failed"
    
    # Test case 4: XOR parameters [1, 1] -> true XOR
    bpl = BinaryProductLayer(n_inputs=2, use_gaussian_init=False)
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    _ = bpl(inputs)  # Initialize
    bpl.set_xor_parameters(torch.tensor([1, 1], dtype=torch.float32))
    output = bpl(inputs)
    expected = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-5), "BinaryProductLayer XOR [1,1] failed"
    
    # Test case 5: 4-bit XOR with pattern [1, 0, 1, 1]
    bpl = BinaryProductLayer(n_inputs=4, use_gaussian_init=False)
    inputs = torch.tensor([[0, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1]], dtype=torch.float32)
    _ = bpl(inputs)  # Initialize
    bpl.set_xor_parameters(torch.tensor([1, 0, 1, 1], dtype=torch.float32))
    output = bpl(inputs)
    expected = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-5), "BinaryProductLayer XOR [1,0,1,1] failed"


def test_multi_binary_product_layer():
    """Test multi-output binary product layer for parallel XOR computation."""
    n_inputs = 30
    n_samples = 100
    n_outputs = 40
    
    # Generate random binary XOR parameters [n_inputs, n_outputs]
    xor_parameters = torch.randint(0, 2, (n_inputs, n_outputs), dtype=torch.float32)
    
    # Generate random binary inputs [n_samples, n_inputs]
    inputs = torch.randint(0, 2, (n_samples, n_inputs), dtype=torch.float32)
    
    # Create layer (with uniform init to match TF behavior initially)
    mbpl = MultiBinaryProductLayer(n_outputs=n_outputs, use_gaussian_init=False)
    
    # Initialize with first forward pass
    _ = mbpl(inputs)
    
    # Set XOR parameters
    mbpl.set_xor_parameters(xor_parameters)
    
    # Compute XOR using layer
    xor_mbpl = mbpl(inputs)
    
    # Compute expected XOR (binary matrix multiplication mod 2)
    # [n_samples, n_inputs] @ [n_inputs, n_outputs] = [n_samples, n_outputs]
    xor_target = (inputs @ xor_parameters) % 2
    
    # Verify outputs match expected XOR
    assert torch.allclose(xor_target, xor_mbpl, atol=1e-5), \
        "Multi binary product layer output should match XOR operation"


def test_multi_binary_product_layer_explicit_cases():
    """Test MultiBinaryProductLayer module with explicit XOR truth table cases."""
    
    # Test case 1: XOR parameters [[0, 0], [0, 0]] -> always [0, 0]
    mbpl = MultiBinaryProductLayer(n_outputs=2, use_gaussian_init=False)
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    _ = mbpl(inputs)  # Initialize
    xor_params = torch.tensor([[0, 0], [0, 0]], dtype=torch.float32)  # [n_inputs=2, n_outputs=2]
    mbpl.set_xor_parameters(xor_params)
    output = mbpl(inputs)
    expected = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-5), "MultiBinaryProductLayer XOR [[0,0],[0,0]] failed"
    
    # Test case 2: XOR parameters [[0, 1], [1, 0]] -> copy and swap
    mbpl = MultiBinaryProductLayer(n_outputs=2, use_gaussian_init=False)
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    _ = mbpl(inputs)  # Initialize
    xor_params = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)  # [n_inputs=2, n_outputs=2]
    mbpl.set_xor_parameters(xor_params)
    output = mbpl(inputs)
    expected = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-5), "MultiBinaryProductLayer XOR [[0,1],[1,0]] failed"
    
    # Test case 3: XOR parameters [[1, 0], [1, 1]] -> first output = input[0], second output = XOR
    mbpl = MultiBinaryProductLayer(n_outputs=2, use_gaussian_init=False)
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    _ = mbpl(inputs)  # Initialize
    xor_params = torch.tensor([[1, 0], [1, 1]], dtype=torch.float32)  # [n_inputs=2, n_outputs=2]
    mbpl.set_xor_parameters(xor_params)
    output = mbpl(inputs)
    expected = torch.tensor([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-5), "MultiBinaryProductLayer XOR [[1,0],[1,1]] failed"
    
    # Test case 4: 4-bit inputs, 2 outputs with patterns [[1, 0, 1, 1], [1, 1, 1, 1]]
    mbpl = MultiBinaryProductLayer(n_outputs=2, use_gaussian_init=False)
    inputs = torch.tensor([[0, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1]], dtype=torch.float32)
    _ = mbpl(inputs)  # Initialize
    xor_params = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float32)  # [n_inputs=4, n_outputs=2]
    mbpl.set_xor_parameters(xor_params)
    output = mbpl(inputs)
    expected = torch.tensor([[1, 1], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-5), "MultiBinaryProductLayer XOR [[1,0,1,1],[1,1,1,1]] failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    
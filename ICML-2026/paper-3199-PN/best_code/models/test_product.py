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
PyTorch tests for product models.

These tests verify that the PyTorch implementation produces the correct model XOR logic
"""

import pytest
import torch

from . import BinaryProductModel, MultiBinaryProductModel

def test_product_model():
    """Test single-output product model for XOR computation."""
    n_inputs = 20
    n_samples = 100

    # Generate random binary XOR parameters
    xor_parameters = torch.randint(0, 2, (n_inputs,), dtype=torch.float32)

    # Generate random binary inputs
    inputs = torch.randint(0, 2, (n_samples, n_inputs), dtype=torch.float32)
    
    # Instantiate and build model
    bpm = BinaryProductModel(n_inputs=n_inputs, use_gaussian_init=False)
    bpm(inputs)

    # Set XOR params
    bpm.product.set_xor_parameters(xor_parameters)

    xor_bpm = bpm(inputs)

    # Compute expected XOR (binary dot product mod 2)
    xor_target = (inputs @ xor_parameters.unsqueeze(1)).squeeze(1) % 2

    # Verify outputs match
    assert torch.allclose(xor_target, xor_bpm, atol=1e-5), \
        "Product model output should match XOR operation"


def test_product_model_explicit_cases():
    """Test BinaryProductModel with explicit XOR truth table cases."""
    
    # Test case 1: XOR parameters [0, 0] -> always 0
    bpm = BinaryProductModel(n_inputs=2, use_gaussian_init=False)
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    _ = bpm(inputs)  # Initialize
    bpm.product.set_xor_parameters(torch.tensor([0, 0], dtype=torch.float32))
    output = bpm(inputs)
    expected = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-5), "BinaryProductModel XOR [0,0] failed"
    
    # Test case 2: XOR parameters [0, 1] -> copy second input
    bpm = BinaryProductModel(n_inputs=2, use_gaussian_init=False)
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    _ = bpm(inputs)  # Initialize
    bpm.product.set_xor_parameters(torch.tensor([0, 1], dtype=torch.float32))
    output = bpm(inputs)
    expected = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-5), "BinaryProductModel XOR [0,1] failed"
    
    # Test case 3: XOR parameters [1, 0] -> copy first input
    bpm = BinaryProductModel(n_inputs=2, use_gaussian_init=False)
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    _ = bpm(inputs)  # Initialize
    bpm.product.set_xor_parameters(torch.tensor([1, 0], dtype=torch.float32))
    output = bpm(inputs)
    expected = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-5), "BinaryProductModel XOR [1,0] failed"
    
    # Test case 4: XOR parameters [1, 1] -> true XOR
    bpm = BinaryProductModel(n_inputs=2, use_gaussian_init=False)
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    _ = bpm(inputs)  # Initialize
    bpm.product.set_xor_parameters(torch.tensor([1, 1], dtype=torch.float32))
    output = bpm(inputs)
    expected = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-5), "BinaryProductModel XOR [1,1] failed"
    
    # Test case 5: 4-bit XOR with pattern [1, 0, 1, 1]
    bpm = BinaryProductModel(n_inputs=4, use_gaussian_init=False)
    inputs = torch.tensor([[0, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1]], dtype=torch.float32)
    _ = bpm(inputs)  # Initialize
    bpm.product.set_xor_parameters(torch.tensor([1, 0, 1, 1], dtype=torch.float32))
    output = bpm(inputs)
    expected = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-5), "BinaryProductModel XOR [1,0,1,1] failed"


def test_multi_product_model():
    """Test multi-output product model for parallel XOR computation."""
    n_inputs = 20
    n_samples = 100
    n_outputs = 30

    # Generate random binary XOR parameters [n_inputs, n_outputs]
    xor_parameters = torch.randint(0, 2, (n_inputs, n_outputs), dtype=torch.float32)

    # Generate random binary inputs [n_samples, n_inputs]
    inputs = torch.randint(0, 2, (n_samples, n_inputs), dtype=torch.float32)

    # Instantiate and build model
    mbpm = MultiBinaryProductModel(n_outputs=n_outputs, use_gaussian_init=False)
    mbpm(inputs)

    # Set XOR params
    mbpm.product.set_xor_parameters(xor_parameters)

    xor_mbpm = mbpm(inputs)

    # Compute expected XOR (binary matrix multiplication mod 2)
    xor_target = (inputs @ xor_parameters) % 2

    # Verify outputs match
    assert torch.allclose(xor_target, xor_mbpm, atol=1e-5), \
        "Multi product model output should match XOR operation"


def test_multi_product_model_explicit_cases():
    """Test MultiBinaryProductModel with explicit XOR truth table cases."""
    
    # Test case 1: XOR parameters [[0, 0], [0, 0]] -> always [0, 0]
    mbpm = MultiBinaryProductModel(n_outputs=2, use_gaussian_init=False)
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    _ = mbpm(inputs)  # Initialize
    xor_params = torch.tensor([[0, 0], [0, 0]], dtype=torch.float32)  # [n_inputs=2, n_outputs=2]
    mbpm.product.set_xor_parameters(xor_params)
    output = mbpm(inputs)
    expected = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-5), "MultiBinaryProductModel XOR [[0,0],[0,0]] failed"
    
    # Test case 2: XOR parameters [[0, 1], [1, 0]] -> copy and swap
    mbpm = MultiBinaryProductModel(n_outputs=2, use_gaussian_init=False)
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    _ = mbpm(inputs)  # Initialize
    xor_params = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)  # [n_inputs=2, n_outputs=2]
    mbpm.product.set_xor_parameters(xor_params)
    output = mbpm(inputs)
    expected = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-5), "MultiBinaryProductModel XOR [[0,1],[1,0]] failed"
    
    # Test case 3: XOR parameters [[1, 0], [1, 1]] -> first output = input[0], second output = XOR
    mbpm = MultiBinaryProductModel(n_outputs=2, use_gaussian_init=False)
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    _ = mbpm(inputs)  # Initialize
    xor_params = torch.tensor([[1, 0], [1, 1]], dtype=torch.float32)  # [n_inputs=2, n_outputs=2]
    mbpm.product.set_xor_parameters(xor_params)
    output = mbpm(inputs)
    expected = torch.tensor([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-5), "MultiBinaryProductModel XOR [[1,0],[1,1]] failed"
    
    # Test case 4: 4-bit inputs, 2 outputs with patterns [[1, 0, 1, 1], [1, 1, 1, 1]]
    mbpm = MultiBinaryProductModel(n_outputs=2, use_gaussian_init=False)
    inputs = torch.tensor([[0, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1]], dtype=torch.float32)
    _ = mbpm(inputs)  # Initialize
    xor_params = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float32)  # [n_inputs=4, n_outputs=2]
    mbpm.product.set_xor_parameters(xor_params)
    output = mbpm(inputs)
    expected = torch.tensor([[1, 1], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1e-5), "MultiBinaryProductLayer XOR [[1,0,1,1],[1,1,1,1]] failed"


def test_model_with_oracle():
    """Test MultiBinaryProductModelWithOracle."""
    from models.product import MultiBinaryProductModelWithOracle
    
    # Create model
    model = MultiBinaryProductModelWithOracle(
        n_outputs=10,
        p_e=0.0,
        use_gaussian_init=True,
        gaussian_mean=0.5,
        gaussian_std=0.2,
    )
    
    # Set oracle weights
    n_inputs = 5
    oracle_weights = torch.randint(0, 2, (n_inputs, 10), dtype=torch.float32)
    model.set_oracle_parameters(oracle_weights)
    
    # Forward pass
    inputs = torch.zeros(32, n_inputs)
    y_oracle, y_model, p_epsilon, p_diff = model(inputs)
    
    # Check shapes
    assert y_oracle.shape == (32, 10), f"Wrong oracle output shape: {y_oracle.shape}"
    assert y_model.shape == (32, 10), f"Wrong model output shape: {y_model.shape}"
    assert p_epsilon.shape == (10,), f"Wrong p_epsilon shape: {p_epsilon.shape}"
    
    # Check ranges
    assert torch.all(y_oracle >= 0) and torch.all(y_oracle <= 1), "Oracle outputs not in [0,1]"
    assert torch.all(y_model >= 0) and torch.all(y_model <= 1), "Model outputs not in [0,1]"
    assert torch.all(p_epsilon >= 0) and torch.all(p_epsilon <= 1), "p_epsilon not in [0,1]"
    



def test_model_with_oracle_explicit_case():
    """Test MultiBinaryProductModelWithOracle with explicit values."""
    from models.product import MultiBinaryProductModelWithOracle
    
    # Create model with p_e = 0 (no channel noise)
    model = MultiBinaryProductModelWithOracle(
        n_outputs=2,
        p_e=0.0,
        use_gaussian_init=False,  # We'll set weights explicitly
    )
    
    # Explicit inputs: [3 samples, 5 inputs]
    inputs = torch.tensor([
        [0, 1, 0, 0, 1],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 1, 1]
    ], dtype=torch.float32)
    
    # Oracle weights: [5 inputs, 2 outputs]
    oracle_weights = torch.tensor([
        [1, 1],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 0]
    ], dtype=torch.float32)
    
    # Model weights: [5 inputs, 2 outputs]
    model_weights = torch.tensor([
        [0, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1]
    ], dtype=torch.float32)
    
    # Set oracle parameters
    model.set_oracle_parameters(oracle_weights)
    
    # Initialize model with a forward pass
    _ = model(inputs)
    
    # Set model weights explicitly
    model.product_model.set_xor_parameters(model_weights)
    
    # Forward pass
    y_oracle, y_model, p_epsilon, p_diff = model(inputs)
    
    # Expected oracle output (computed as inputs @ oracle_weights % 2)
    # Sample 0: [0,1,0,0,1] @ [[1,1],[1,0],[1,0],[0,1],[0,0]] = [1, 0]
    # Sample 1: [1,1,0,1,1] @ [[1,1],[1,0],[1,0],[0,1],[0,0]] = [0, 0]
    # Sample 2: [0,0,0,1,1] @ [[1,1],[1,0],[1,0],[0,1],[0,0]] = [0, 1]
    expected_oracle = torch.tensor([
        [1, 0],
        [0, 0],
        [0, 1]
    ], dtype=torch.float32)
    
    # Expected model output (computed as inputs @ model_weights % 2)
    # Sample 0: [0,1,0,0,1] @ [[0,1],[1,1],[1,1],[0,0],[1,1]] = [0, 0]
    # Sample 1: [1,1,0,1,1] @ [[0,1],[1,1],[1,1],[0,0],[1,1]] = [0, 1]
    # Sample 2: [0,0,0,1,1] @ [[0,1],[1,1],[1,1],[0,0],[1,1]] = [1, 1]
    expected_model = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 1]
    ], dtype=torch.float32)
    

    # Expected p_epsilon (mean binary parameter error per output)
    # p_epsilon is computed as mean(abs(model_weights - oracle_weights), dim=0)
    # After binarization (threshold at 0.5):
    # Model weights:  [[0,1],[1,1],[1,1],[0,0],[1,1]]
    # Oracle weights: [[1,1],[1,0],[1,0],[0,1],[0,0]]
    # Differences:    [[1,0],[0,1],[0,1],[0,1],[1,1]]
    # Mean per output: [(1+0+0+0+1)/5, (0+1+1+1+1)/5] = [2/5, 4/5] = [0.4, 0.8]
    expected_p_epsilon = torch.tensor([0.4, 0.8], dtype=torch.float32)

    # Expected p_diff (mean parameter distance per output)
    # p_diff is computed as mean(abs(model_weights - oracle_weights), dim=0)
    # Without binarization:
    # Model weights:  [[0,1],[1,1],[1,1],[0,0],[1,1]]
    # Oracle weights: [[1,1],[1,0],[1,0],[0,1],[0,0]]
    # Differences:    [[1,0],[0,1],[0,1],[0,1],[1,1]]
    # Mean per output: [(1+0+0+0+1)/5, (0+1+1+1+1)/5] = [2/5, 4/5] = [0.4, 0.8]
    expected_p_diff = torch.tensor([0.4, 0.8], dtype=torch.float32) # Same because weight are already binary here
    
    # Verify oracle output
    assert torch.allclose(y_oracle, expected_oracle, atol=1e-5), \
        f"Oracle output mismatch.\nExpected:\n{expected_oracle}\nGot:\n{y_oracle}"
    
    # Verify model output
    assert torch.allclose(y_model, expected_model, atol=1e-5), \
        f"Model output mismatch.\nExpected:\n{expected_model}\nGot:\n{y_model}"
    
    # Verify p_epsilon
    assert torch.allclose(p_epsilon, expected_p_epsilon, atol=1e-5), \
        f"p_epsilon mismatch.\nExpected: {expected_p_epsilon}\nGot: {p_epsilon}"
    
    # Verify p_diff
    assert torch.allclose(p_diff, expected_p_diff, atol=1e-5), \
        f"p_diff mismatch.\nExpected: {expected_p_diff}\nGot: {p_diff}"
    
def test_model_with_oracle_param_errors():
    """Test MultiBinaryProductModelWithOracle parameters errors with explicit values."""
    from models.product import MultiBinaryProductModelWithOracle
    
    # Create model with p_e = 0 (no channel noise)
    model = MultiBinaryProductModelWithOracle(
        n_outputs=2,
        p_e=0.0,
        use_gaussian_init=False,  # We'll set weights explicitly
    )
    
    # Explicit inputs: [3 samples, 5 inputs]
    inputs = torch.tensor([
        [0, 1, 0, 0, 1],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 1, 1]
    ], dtype=torch.float32)
    
    # Oracle weights: [5 inputs, 2 outputs]
    oracle_weights = torch.tensor([
        [1, 1],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 0]
    ], dtype=torch.float32)
    
    # Model weights: [5 inputs, 2 outputs]
    model_weights = torch.tensor([
        [0.3, 0.9],
        [0.9, 0.8],
        [0.6, 0.9],
        [0.4, 0.1],
        [0.8, 0.7]
    ], dtype=torch.float32)
    
    # Set oracle parameters
    model.set_oracle_parameters(oracle_weights)
    
    # Initialize model with a forward pass
    _ = model(inputs)
    
    # Set model weights explicitly
    model.product_model.set_xor_parameters(model_weights)
    
    # Forward pass
    y_oracle, y_model, p_epsilon, p_diff = model(inputs)    

    # Expected p_epsilon (mean binary parameter error per output)
    # p_epsilon is computed as mean(abs(model_weights - oracle_weights), dim=0)
    # After binarization (threshold at 0.5):
    # Model weights:  [[0,1],[1,1],[1,1],[0,0],[1,1]]
    # Oracle weights: [[1,1],[1,0],[1,0],[0,1],[0,0]]
    # Differences:    [[1,0],[0,1],[0,1],[0,1],[1,1]]
    # Mean per output: [(1+0+0+0+1)/5, (0+1+1+1+1)/5] = [2/5, 4/5] = [0.4, 0.8]
    expected_p_epsilon = torch.tensor([0.4, 0.8], dtype=torch.float32)

    # Expected p_diff (mean parameter distance per output)
    # p_diff is computed as mean(abs(model_weights - oracle_weights), dim=0)
    # Without binarization:
    # Model weights:  [[0.3, 0.9], [0.9, 0.8], [0.6, 0.9], [0.4, 0.1], [0.8, 0.7]
    # Oracle weights: [[1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    # Abs Differences:[[0.7, 0.1], [0.1, 0.8], [0.4, 0.9], [0.4, 0.9], [0.8, 0.7]]
    # Mean per output: [(0.7+0.1+0.4+0.4+0.8)/5, (0.1+0.8+0.9+0.9+0.7)/5] = [2/5, 4/5] = [0.48, 0.68]
    expected_p_diff = torch.tensor([0.48, 0.68], dtype=torch.float32) # Same because weight are already binary here

    
    # Verify p_epsilon
    assert torch.allclose(p_epsilon, expected_p_epsilon, atol=1e-5), \
        f"p_epsilon mismatch.\nExpected: {expected_p_epsilon}\nGot: {p_epsilon}"
    
    # Verify p_diff
    assert torch.allclose(p_diff, expected_p_diff, atol=1e-5), \
        f"p_diff mismatch.\nExpected: {expected_p_diff}\nGot: {p_diff}"

def test_gradient_flow():
    """Test that gradients flow through the model."""
    from models.product import MultiBinaryProductModelWithOracle
    
    # Create model
    model = MultiBinaryProductModelWithOracle(
        n_outputs=5,
        p_e=0.5,
        use_gaussian_init=True,
    )
    
    # Set oracle
    n_inputs = 3
    oracle_weights = torch.randint(0, 2, (n_inputs, 5), dtype=torch.float32)
    model.set_oracle_parameters(oracle_weights)
    
    # Forward pass
    inputs = torch.zeros(10, n_inputs)
    y_oracle, y_model, p_epsilon, p_diff = model(inputs)
    
    # Compute loss
    loss = torch.sum((y_model - y_oracle) ** 2)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    # Oracle gradients should be zero (hard_step uses non-differentiable torch.sign)
    # Note: grad is not None because requires_grad=True, but values should be 0
    assert model.product_model.product_weights.grad is not None, "No gradient on model weights"
    
    oracle_grad = model.product_oracle.product_weights.grad
    if oracle_grad is not None:
        oracle_grad_norm = torch.norm(oracle_grad)
        assert oracle_grad_norm == 0, f"Oracle should have zero gradients (hard_step), got norm={oracle_grad_norm}"
    
    # Check gradient is not zero for model
    grad_norm = torch.norm(model.product_model.product_weights.grad)
    assert grad_norm > 0, "Gradient is zero"


def test_training_steps():
    """Test training steps."""
    from models.product import MultiBinaryProductModelWithOracle
    
    # Create model
    n_samples = 512
    n_inputs = 10
    n_outputs = 100
    model = MultiBinaryProductModelWithOracle(
        n_outputs=n_outputs,
        p_e=0.01,
        use_gaussian_init=True,
    )
    
    # Set oracle
    oracle_weights = torch.randint(0, 2, (n_inputs, n_outputs), dtype=torch.float32)
    
    # Use random binary inputs instead of zeros
    inputs = torch.randint(0, 2, (n_samples, n_inputs), dtype=torch.float32)
    
    # Initialize model with forward pass first
    _ = model(inputs)
    
    # Now set oracle parameters
    model.set_oracle_parameters(oracle_weights)
    
    # Create optimizer AFTER weights are initialized
    optimizer = torch.optim.SGD(model.product_model.parameters(), lr=0.001)  # Lower LR
    
    # Initial loss
    y_oracle, y_model, _, _  = model(inputs)
    initial_loss = torch.sum((y_model - y_oracle) ** 2).item()
    
    # Multiple training steps to ensure convergence
    for _ in range(1000):
        inputs = torch.randint(0, 2, (n_samples, n_inputs), dtype=torch.float32)
        optimizer.zero_grad()
        y_oracle, y_model, _, _ = model(inputs)
        loss = torch.sum((y_model - y_oracle) ** 2)
        loss.backward()
        optimizer.step()
    
    # Loss after steps
    y_oracle, y_model, _, _ = model(inputs)
    final_loss = torch.sum((y_model - y_oracle) ** 2).item()
    
    # Loss should decrease after multiple steps
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"



def test_gaussian_vs_uniform():
    """Test that both initializations work."""
    from models.layers.product import MultiBinaryProductLayer
    
    # Gaussian
    layer_gauss = MultiBinaryProductLayer(
        n_outputs=100,
        use_gaussian_init=True,
        gaussian_mean=0.5,
        gaussian_std=0.2,
    )
    
    # Initialize weights by forward pass
    inputs = torch.zeros(10, 5)
    _ = layer_gauss(inputs)
    
    weights_gauss = layer_gauss.product_weights.data
    
    # Check mean close to 0.5
    mean = torch.mean(weights_gauss).item()
    std = torch.std(weights_gauss).item()
    
    assert 0.4  < mean < 0.6,   f"Gaussian mean not close to 0.5: {mean}"
    assert 0.15 < std  < 0.25,  f"Gaussian std not close to 0.2: {std}"
    

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


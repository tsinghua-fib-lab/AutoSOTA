import torch as t
import pytest
from dictionary_learning.dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    JumpReluAutoEncoder,
)


@pytest.mark.parametrize(
    "sae_cls", [AutoEncoder, GatedAutoEncoder, JumpReluAutoEncoder]
)
def test_forward_equals_decode_encode(sae_cls: type) -> None:
    """Test that forward pass equals decode(encode(x)) for all SAE types"""
    batch_size = 4
    act_dim = 8
    dict_size = 6
    x = t.randn(batch_size, act_dim)

    sae = sae_cls(activation_dim=act_dim, dict_size=dict_size)

    # Test without output_features
    forward_out = sae(x)
    encode_decode = sae.decode(sae.encode(x))
    assert t.allclose(forward_out, encode_decode)

    # Test with output_features
    forward_out, features = sae(x, output_features=True)
    encode_features = sae.encode(x)
    assert t.allclose(features, encode_features)


def test_simple_autoencoder() -> None:
    """Test AutoEncoder with simple weight matrices"""
    sae = AutoEncoder(activation_dim=2, dict_size=2)

    # Set simple weights
    with t.no_grad():
        sae.encoder.weight.data = t.tensor([[1.0, 0.0], [0.0, 1.0]])
        sae.decoder.weight.data = t.tensor([[1.0, 0.0], [0.0, 1.0]])
        sae.encoder.bias.data = t.zeros(2)
        sae.bias.data = t.zeros(2)

    # Test encoding
    x = t.tensor([[2.0, -1.0]])
    encoded = sae.encode(x)
    assert t.allclose(encoded, t.tensor([[2.0, 0.0]]))  # ReLU clips negative value

    # Test decoding
    decoded = sae.decode(encoded)
    assert t.allclose(decoded, t.tensor([[2.0, 0.0]]))


def test_simple_gated_autoencoder() -> None:
    """Test GatedAutoEncoder with simple weight matrices"""
    sae = GatedAutoEncoder(activation_dim=2, dict_size=2)

    # Set simple weights and biases
    with t.no_grad():
        sae.encoder.weight.data = t.tensor([[1.0, 0.0], [0.0, 1.0]])
        sae.decoder.weight.data = t.tensor([[1.0, 0.0], [0.0, 1.0]])
        sae.gate_bias.data = t.zeros(2)
        sae.mag_bias.data = t.zeros(2)
        sae.r_mag.data = t.zeros(2)
        sae.decoder_bias.data = t.zeros(2)

    x = t.tensor([[2.0, -1.0]])
    encoded = sae.encode(x)
    assert t.allclose(
        encoded, t.tensor([[2.0, 0.0]])
    )  # Only positive values pass through


def test_normalize_decoder() -> None:
    """Test that normalize_decoder maintains output while normalizing weights"""
    sae = AutoEncoder(activation_dim=4, dict_size=3)
    x = t.randn(2, 4)

    # Get initial output
    initial_output = sae(x)

    # Normalize decoder
    sae.normalize_decoder()

    # Check decoder weights are normalized
    norms = t.norm(sae.decoder.weight, dim=0)
    assert t.allclose(norms, t.ones_like(norms))

    # Check output is maintained
    new_output = sae(x)
    assert t.allclose(initial_output, new_output, atol=1e-4)


def test_scale_biases() -> None:
    """Test that scale_biases correctly scales all bias terms"""
    sae = AutoEncoder(activation_dim=4, dict_size=3)

    # Record initial biases
    initial_encoder_bias = sae.encoder.bias.data.clone()
    initial_bias = sae.bias.data.clone()

    scale = 2.0
    sae.scale_biases(scale)

    assert t.allclose(sae.encoder.bias.data, initial_encoder_bias * scale)
    assert t.allclose(sae.bias.data, initial_bias * scale)


@pytest.mark.parametrize(
    "sae_cls", [AutoEncoder, GatedAutoEncoder, AutoEncoderNew, JumpReluAutoEncoder]
)
def test_output_shapes(sae_cls: type) -> None:
    """Test that output shapes are correct for all operations"""
    batch_size = 3
    act_dim = 4
    dict_size = 5
    x = t.randn(batch_size, act_dim)

    sae = sae_cls(activation_dim=act_dim, dict_size=dict_size)

    # Test encode shape
    encoded = sae.encode(x)
    assert encoded.shape == (batch_size, dict_size)

    # Test decode shape
    decoded = sae.decode(encoded)
    assert decoded.shape == (batch_size, act_dim)

    # Test forward shape with and without features
    output = sae(x)
    assert output.shape == (batch_size, act_dim)

    output, features = sae(x, output_features=True)
    assert output.shape == (batch_size, act_dim)
    assert features.shape == (batch_size, dict_size)

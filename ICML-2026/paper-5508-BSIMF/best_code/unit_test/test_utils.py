"""Unit tests for core/utils.py."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import pytest
import torch
import torch.nn as nn

from utils import LOG_VAR_MIN, LOG_VAR_MAX, _log_normal_diag, MLP, TransformerEncoderLayer, ViTBackbone, CrossAttention


# ---------------------------------------------------------------------------
# _log_normal_diag
# ---------------------------------------------------------------------------

class TestLogNormalDiag:
    def test_output_shape_1d(self):
        x = torch.zeros(5)
        out = _log_normal_diag(x, x, x)
        assert out.shape == (5,)

    def test_output_shape_2d(self):
        x = torch.zeros(3, 4)
        out = _log_normal_diag(x, x, x)
        assert out.shape == (3, 4)

    def test_standard_normal_at_zero(self):
        # log N(0; 0, 1) = -0.5 * log(2*pi) ≈ -0.9189
        x = torch.zeros(1)
        lp = _log_normal_diag(x, x, x)  # mean=0, log_var=0 → var=1
        expected = -0.5 * math.log(2 * math.pi)
        assert abs(lp.item() - expected) < 1e-5

    def test_mode_is_maximum(self):
        # log p is maximized when x == mean
        mean = torch.tensor([1.0, -2.0, 0.5])
        log_var = torch.zeros(3)
        lp_at_mode = _log_normal_diag(mean, mean, log_var)
        lp_off = _log_normal_diag(mean + 1.0, mean, log_var)
        assert (lp_at_mode > lp_off).all()

    def test_log_var_clamped(self):
        # Extreme log_var should not produce NaN or Inf
        x = torch.zeros(3)
        lp_large = _log_normal_diag(x, x, torch.full((3,), 100.0))
        lp_small = _log_normal_diag(x, x, torch.full((3,), -100.0))
        assert torch.isfinite(lp_large).all()
        assert torch.isfinite(lp_small).all()

    def test_larger_variance_lower_peak(self):
        mean = torch.zeros(1)
        x = torch.zeros(1)
        lp_narrow = _log_normal_diag(x, mean, torch.tensor([0.0]))    # var=1
        lp_wide   = _log_normal_diag(x, mean, torch.tensor([2.0]))    # var≈7.4
        assert lp_narrow > lp_wide

    def test_broadcasting(self):
        x   = torch.zeros(4, 3)
        mu  = torch.zeros(1, 3)
        lv  = torch.zeros(4, 1)
        out = _log_normal_diag(x, mu, lv)
        assert out.shape == (4, 3)

    def test_constants_exported(self):
        assert LOG_VAR_MIN == -20.0
        assert LOG_VAR_MAX == 20.0


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class TestMLP:
    def test_output_shape(self):
        mlp = MLP(in_dim=8, hidden_dims=(16, 16), out_dim=4)
        x = torch.randn(5, 8)
        out = mlp(x)
        assert out.shape == (5, 4)

    def test_no_hidden_layers(self):
        # hidden_dims=() → single linear layer
        mlp = MLP(in_dim=6, hidden_dims=(), out_dim=3)
        x = torch.randn(2, 6)
        out = mlp(x)
        assert out.shape == (2, 3)

    def test_single_hidden_layer(self):
        mlp = MLP(in_dim=4, hidden_dims=(8,), out_dim=2)
        x = torch.randn(7, 4)
        assert mlp(x).shape == (7, 2)

    def test_dropout_zero_deterministic(self):
        mlp = MLP(in_dim=4, hidden_dims=(8,), out_dim=2, dropout=0.0)
        mlp.eval()
        x = torch.randn(3, 4)
        assert torch.allclose(mlp(x), mlp(x))

    def test_with_dropout(self):
        mlp = MLP(in_dim=4, hidden_dims=(8,), out_dim=2, dropout=0.5)
        mlp.train()
        x = torch.randn(10, 4)
        out = mlp(x)
        assert out.shape == (10, 2)


# ---------------------------------------------------------------------------
# TransformerEncoderLayer
# ---------------------------------------------------------------------------

class TestTransformerEncoderLayer:
    def _layer(self):
        return TransformerEncoderLayer(embed_dim=8, num_heads=2)

    def test_output_shape(self):
        layer = self._layer()
        x = torch.randn(2, 5, 8)   # (B, N, D)
        out = layer(x)
        assert out.shape == (2, 5, 8)

    def test_output_differs_from_input(self):
        # Residual connections mean output != input in general (unless all weights zero)
        torch.manual_seed(0)
        layer = self._layer()
        x = torch.randn(2, 5, 8)
        out = layer(x)
        assert not torch.allclose(out, x)

    def test_batch_independence(self):
        # Each sample in the batch is processed independently (no cross-batch attention)
        layer = self._layer()
        layer.eval()
        x = torch.randn(4, 5, 8)
        out_full = layer(x)
        out_first = layer(x[:1])
        assert torch.allclose(out_full[:1], out_first, atol=1e-5)


# ---------------------------------------------------------------------------
# ViTBackbone
# ---------------------------------------------------------------------------

class TestViTBackbone:
    def _vit(self, image_size=8, patch_size=4):
        return ViTBackbone(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=1,
            embed_dim=8,
            depth=1,
            num_heads=2,
        )

    def test_patch_tokens_shape(self):
        vit = self._vit(image_size=8, patch_size=4)
        B = 3
        x = torch.randn(B, 1, 8, 8)
        tokens, cls = vit(x)
        n_patches = (8 // 4) ** 2   # = 4
        assert tokens.shape == (B, n_patches, 8)

    def test_cls_repr_shape(self):
        vit = self._vit(image_size=8, patch_size=4)
        x = torch.randn(2, 1, 8, 8)
        _, cls = vit(x)
        assert cls.shape == (2, 8)

    def test_patch_size_1(self):
        vit = self._vit(image_size=4, patch_size=1)
        x = torch.randn(2, 1, 4, 4)
        tokens, cls = vit(x)
        assert tokens.shape == (2, 16, 8)  # 4*4 = 16 patches

    def test_batch_independence(self):
        vit = self._vit()
        vit.eval()
        x = torch.randn(4, 1, 8, 8)
        tokens_full, cls_full = vit(x)
        tokens_0, cls_0 = vit(x[:1])
        assert torch.allclose(cls_full[:1], cls_0, atol=1e-5)


# ---------------------------------------------------------------------------
# CrossAttention
# ---------------------------------------------------------------------------

class TestCrossAttention:
    def _ca(self, embed_dim=8, latent_dim=4, num_heads=2):
        return CrossAttention(embed_dim=embed_dim, latent_dim=latent_dim, num_heads=num_heads)

    def test_output_shape(self):
        ca = self._ca()
        B, N = 3, 5
        tokens = torch.randn(B, N, 8)
        z      = torch.randn(B, 4)
        out    = ca(tokens, z)
        assert out.shape == (B, N, 8)

    def test_different_z_changes_output(self):
        torch.manual_seed(0)
        ca = self._ca()
        ca.eval()
        tokens = torch.randn(2, 4, 8)
        z1 = torch.randn(2, 4)
        z2 = torch.randn(2, 4)
        assert not torch.allclose(ca(tokens, z1), ca(tokens, z2))

    def test_gradient_flows_through_z(self):
        ca = self._ca()
        tokens = torch.randn(2, 4, 8)
        z = torch.randn(2, 4, requires_grad=True)
        out = ca(tokens, z)
        out.sum().backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

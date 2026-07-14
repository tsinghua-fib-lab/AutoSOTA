"""Unit tests for core/model.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import torch

from model import (
    MaskDecoder,
    ImageDecoder,
    ContinuousLabelDecoder,
    CategoricalLabelDecoder,
    UConditionalPrior,
    ContentUncertaintyDAG,
)


# ---------------------------------------------------------------------------
# MaskDecoder
# ---------------------------------------------------------------------------

class TestMaskDecoder:
    def _decoder(self, image_size=8, patch_size=4, embed_dim=8, z_dim=4):
        return MaskDecoder(
            embed_dim=embed_dim,
            latent_dim=z_dim,
            num_heads=2,
            image_size=image_size,
            patch_size=patch_size,
        )

    def test_output_shape(self):
        dec = self._decoder()
        B, N = 3, 4          # N = (8//4)^2
        tokens = torch.randn(B, N, 8)
        z      = torch.randn(B, 4)
        out    = dec(tokens, z)
        assert out.shape == (B, 8 * 8)

    def test_probabilities_in_0_1(self):
        dec = self._decoder()
        tokens = torch.randn(2, 4, 8)
        z      = torch.randn(2, 4)
        out    = dec(tokens, z)
        assert (out > 0).all() and (out < 1).all()

    def test_gradient_flows(self):
        dec = self._decoder()
        tokens = torch.randn(2, 4, 8)
        z = torch.randn(2, 4, requires_grad=True)
        out = dec(tokens, z)
        out.sum().backward()
        assert z.grad is not None


# ---------------------------------------------------------------------------
# ImageDecoder
# ---------------------------------------------------------------------------

class TestImageDecoder:
    def _decoder(self, z_dim=4, y_dim=16, rank_r=2, **kwargs):
        return ImageDecoder(
            z_dim=z_dim,
            y_dim_flat=y_dim,
            rank_r=rank_r,
            hidden_dims=(16,),
            **kwargs,
        )

    def test_base_params_shape_scalar_mode(self):
        dec = self._decoder(base_mean_mode="learnable_scalar", base_var_mode="learnable_scalar")
        mu, lv = dec.base_params(batch_size=3)
        assert mu.shape == (3, 16)
        assert lv.shape == (3, 16)

    def test_base_params_shape_per_pixel_mode(self):
        dec = self._decoder(base_mean_mode="learnable_per_pixel", base_var_mode="learnable_per_pixel")
        mu, lv = dec.base_params(batch_size=5)
        assert mu.shape == (5, 16)
        assert lv.shape == (5, 16)

    def test_content_factor_shape(self):
        dec = self._decoder(rank_r=3)
        z   = torch.randn(4, 4)
        mu, cov = dec.content_factor(z)
        assert mu.shape  == (4, 16)
        assert cov.shape == (4, 16, 3)

    def test_content_params_shape(self):
        dec = self._decoder()
        z   = torch.randn(2, 4)
        mu, lv = dec.content_params(z)
        assert mu.shape == (2, 16)
        assert lv.shape == (2, 16)

    def test_content_params_finite(self):
        dec = self._decoder()
        z   = torch.randn(5, 4)
        mu, lv = dec.content_params(z)
        assert torch.isfinite(mu).all()
        assert torch.isfinite(lv).all()

    def test_rank_zero_content_factor(self):
        dec = self._decoder(rank_r=0)
        z   = torch.randn(2, 4)
        mu, cov = dec.content_factor(z)
        assert cov.shape == (2, 16, 0)


# ---------------------------------------------------------------------------
# ContinuousLabelDecoder
# ---------------------------------------------------------------------------

class TestContinuousLabelDecoder:
    def test_output_shapes(self):
        dec = ContinuousLabelDecoder(z_dim=4, u_dim=6, x_dim=6)
        z   = torch.randn(5, 4)
        u   = torch.randn(5, 6)
        mean, lv = dec(z, u)
        assert mean.shape == (5, 6)
        assert lv.shape   == (5, 6)

    def test_mean_is_az_plus_u(self):
        dec = ContinuousLabelDecoder(z_dim=2, u_dim=3, x_dim=3)
        dec.eval()
        z = torch.randn(4, 2)
        u = torch.randn(4, 3)
        mean, _ = dec(z, u)
        expected = z.matmul(dec.A.t()) + u
        assert torch.allclose(mean, expected, atol=1e-6)

    def test_log_var_finite(self):
        dec = ContinuousLabelDecoder(z_dim=4, u_dim=6, x_dim=6)
        z   = torch.randn(3, 4)
        u   = torch.randn(3, 6)
        _, lv = dec(z, u)
        assert torch.isfinite(lv).all()


# ---------------------------------------------------------------------------
# CategoricalLabelDecoder
# ---------------------------------------------------------------------------

class TestCategoricalLabelDecoder:
    def test_logits_shape(self):
        dec = CategoricalLabelDecoder(z_dim=4, u_dim=3, x_dim=5, num_classes=7)
        z   = torch.randn(6, 4)
        u   = torch.randn(6, 3)
        out = dec.logits(z, u)
        assert out.shape == (6, 5, 7)

    def test_logits_finite(self):
        dec = CategoricalLabelDecoder(z_dim=4, u_dim=3, x_dim=5, num_classes=7)
        z   = torch.randn(3, 4)
        u   = torch.randn(3, 3)
        assert torch.isfinite(dec.logits(z, u)).all()

    def test_different_z_gives_different_logits(self):
        dec = CategoricalLabelDecoder(z_dim=4, u_dim=3, x_dim=5, num_classes=7)
        dec.eval()
        u  = torch.zeros(2, 3)
        z1 = torch.randn(2, 4)
        z2 = torch.randn(2, 4)
        assert not torch.allclose(dec.logits(z1, u), dec.logits(z2, u))


# ---------------------------------------------------------------------------
# UConditionalPrior
# ---------------------------------------------------------------------------

class TestUConditionalPrior:
    def test_output_shapes(self):
        prior = UConditionalPrior(y_feat_dim=8, u_dim=6, hidden_dims=(16,))
        y_feat = torch.randn(4, 8)
        mu, lv = prior(y_feat)
        assert mu.shape == (4, 6)
        assert lv.shape == (4, 6)

    def test_mu_constant_across_batch(self):
        # mu_psi does not depend on y → same for all items in the batch
        prior = UConditionalPrior(y_feat_dim=8, u_dim=4, hidden_dims=(16,))
        prior.eval()
        y = torch.randn(5, 8)
        mu, _ = prior(y)
        # All rows should be the same (broadcast from a single learnable vector)
        assert torch.allclose(mu[0], mu[1]) and torch.allclose(mu[0], mu[-1])

    def test_log_var_varies_with_y(self):
        prior = UConditionalPrior(y_feat_dim=8, u_dim=4, hidden_dims=(16,))
        prior.eval()
        y1 = torch.randn(1, 8)
        y2 = torch.randn(1, 8)
        _, lv1 = prior(y1)
        _, lv2 = prior(y2)
        assert not torch.allclose(lv1, lv2)


# ---------------------------------------------------------------------------
# ContentUncertaintyDAG — model-level tests
# ---------------------------------------------------------------------------

class TestContentUncertaintyDAG:
    # ------ properties ------

    def test_mixture_weights_sum_to_one(self, tiny_model):
        w = tiny_model.mixture_weights
        assert abs(w.sum().item() - 1.0) < 1e-5

    def test_mask_prior_probs_in_range(self, tiny_model):
        rho = tiny_model.mask_prior_probs
        assert (rho > 0).all() and (rho < 1).all()

    def test_mixture_log_vars_clamped(self, tiny_model):
        lv = tiny_model.mixture_log_vars
        assert (lv >= -20.0).all() and (lv <= 20.0).all()

    # ------ encode ------

    def test_encode_output_keys(self, tiny_model, tiny_batch):
        x, y = tiny_batch
        with torch.no_grad():
            out = tiny_model.encode(x, y)
        expected = {"x_feat", "y_tokens", "y_global", "mu_z", "log_var_z",
                    "mu_u", "log_var_u", "mask_x"}
        assert set(out.keys()) == expected

    def test_encode_output_shapes(self, tiny_model, tiny_batch):
        x, y = tiny_batch
        B = x.shape[0]
        with torch.no_grad():
            out = tiny_model.encode(x, y)
        assert out["mu_z"].shape      == (B, tiny_model.z_dim)
        assert out["log_var_z"].shape == (B, tiny_model.z_dim)
        assert out["mu_u"].shape      == (B, tiny_model.u_dim)
        assert out["log_var_u"].shape == (B, tiny_model.u_dim)
        n_patches = (tiny_model.image_size // tiny_model.vit_patch_size) ** 2
        assert out["y_tokens"].shape == (B, n_patches, tiny_model.vit.embed_dim)

    def test_encode_log_var_clamped(self, tiny_model, tiny_batch):
        x, y = tiny_batch
        with torch.no_grad():
            out = tiny_model.encode(x, y)
        for key in ("log_var_z", "log_var_u"):
            assert (out[key] >= -20.0).all() and (out[key] <= 20.0).all()

    def test_encode_nan_x_does_not_propagate(self, tiny_model, tiny_batch):
        x, y = tiny_batch
        x_nan = x.clone()
        x_nan[:, 0] = float("nan")
        with torch.no_grad():
            out = tiny_model.encode(x_nan, y)
        for v in out.values():
            assert torch.isfinite(v).all(), f"NaN found in encode output"

    def test_encode_all_missing_x_finite(self, tiny_model, tiny_batch):
        x, y = tiny_batch
        x_nan = torch.full_like(x, float("nan"))
        with torch.no_grad():
            out = tiny_model.encode(x_nan, y)
        for v in out.values():
            assert torch.isfinite(v).all()

    # ------ mask_probs ------

    def test_mask_probs_shape(self, tiny_model, encode_out):
        z = encode_out["mu_z"]
        y_tokens = encode_out["y_tokens"]
        with torch.no_grad():
            g = tiny_model.mask_probs(y_tokens, z)
        assert g.shape == (z.shape[0], tiny_model.y_dim_flat)

    def test_mask_probs_in_range(self, tiny_model, encode_out):
        z = encode_out["mu_z"]
        with torch.no_grad():
            g = tiny_model.mask_probs(encode_out["y_tokens"], z)
        assert (g > 0).all() and (g < 1).all()

    # ------ decode_x_continuous ------

    def test_decode_x_continuous_shapes(self, tiny_model):
        B = 4
        z = torch.randn(B, tiny_model.z_dim)
        u = torch.randn(B, tiny_model.u_dim)
        with torch.no_grad():
            mean, lv = tiny_model.decode_x_continuous(z, u)
        assert mean.shape == (B, tiny_model.x_dim)
        assert lv.shape   == (B, tiny_model.x_dim)

    # ------ image_base_params / image_content_params ------

    def test_image_base_params_shape(self, tiny_model):
        with torch.no_grad():
            mu, lv = tiny_model.image_base_params(batch_size=5)
        assert mu.shape == (5, tiny_model.y_dim_flat)
        assert lv.shape == (5, tiny_model.y_dim_flat)

    def test_image_content_params_shape(self, tiny_model):
        z = torch.randn(3, tiny_model.z_dim)
        with torch.no_grad():
            mu, lv = tiny_model.image_content_params(z)
        assert mu.shape == (3, tiny_model.y_dim_flat)
        assert lv.shape == (3, tiny_model.y_dim_flat)

    def test_image_content_factor_shape(self, tiny_model):
        z = torch.randn(3, tiny_model.z_dim)
        with torch.no_grad():
            mu, cov = tiny_model.image_content_factor(z)
        assert mu.shape  == (3, tiny_model.y_dim_flat)
        assert cov.shape == (3, tiny_model.y_dim_flat, 2)  # rank_r=2

    # ------ prior_u ------

    def test_prior_u_shape(self, tiny_model, tiny_batch):
        _, y = tiny_batch
        with torch.no_grad():
            mu, lv = tiny_model.prior_u(y)
        assert mu.shape == (y.shape[0], tiny_model.u_dim)
        assert lv.shape == (y.shape[0], tiny_model.u_dim)

    # ------ forward / full pass ------

    def test_forward_returns_encode_out(self, tiny_model, tiny_batch):
        x, y = tiny_batch
        with torch.no_grad():
            out = tiny_model(x, y)
        assert "mu_z" in out and "mu_u" in out

    def test_categorical_model_encodes(self):
        model = ContentUncertaintyDAG(
            x_dim=4, image_size=8, z_dim=2, u_dim=3, num_components=2,
            x_distribution="categorical", x_num_classes=5,
            vit_embed_dim=8, vit_depth=1, vit_num_heads=2, vit_patch_size=4,
            mlp_hidden_dims=(16,), rank_r=2,
        ).eval()
        x = torch.randint(0, 5, (3, 4))
        y = torch.randn(3, 1, 8, 8)
        with torch.no_grad():
            out = model.encode(x, y)
        assert out["mu_z"].shape == (3, 2)

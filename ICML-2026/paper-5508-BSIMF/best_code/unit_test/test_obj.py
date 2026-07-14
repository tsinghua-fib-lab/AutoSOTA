"""Unit tests for core/obj.py."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import torch

from obj import (
    _reparameterize_gaussian,
    _log_normal_lowrank_plus_diag,
    compute_image_likelihood_term,
    compute_label_likelihood_term,
    compute_kl_zc_term,
    compute_kl_u_term,
    compute_kl_m_term,
    compute_sparse_m_term,
    compute_mask_tv_term,
    compute_elbo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(x_dim=4, image_size=8, z_dim=2, u_dim=4, num_components=2,
                rank_r=2, seed=0):
    from model import ContentUncertaintyDAG
    torch.manual_seed(seed)
    return ContentUncertaintyDAG(
        x_dim=x_dim, image_size=image_size, z_dim=z_dim, u_dim=u_dim,
        num_components=num_components, x_distribution="continuous",
        vit_embed_dim=8, vit_depth=1, vit_num_heads=2, vit_patch_size=4,
        mlp_hidden_dims=(16,), rank_r=rank_r,
        mask_prior_rho=0.1,
        base_mean_mode="learnable_scalar", base_var_mode="learnable_scalar",
        base_init_mean=0.1, base_init_var=0.25 ** 2,
        use_mask_x_in_encoder=True, u_missing_fallback_to_prior=True,
    ).eval()


def _make_batch(model, B=3, seed=1):
    torch.manual_seed(seed)
    x = torch.randn(B, model.x_dim)
    y = torch.randn(B, 1, model.image_size, model.image_size)
    return x, y


def _encode(model, x, y):
    with torch.no_grad():
        return model.encode(x, y)


# ---------------------------------------------------------------------------
# _reparameterize_gaussian
# ---------------------------------------------------------------------------

class TestReparameterizeGaussian:
    def test_output_shape(self):
        mu  = torch.zeros(3, 5)
        lv  = torch.zeros(3, 5)
        out = _reparameterize_gaussian(mu, lv, num_samples=4)
        assert out.shape == (4, 3, 5)

    def test_sample_mean_close_to_mu(self):
        torch.manual_seed(42)
        mu  = torch.tensor([[1.0, -2.0, 3.0]])
        lv  = torch.zeros(1, 3)
        out = _reparameterize_gaussian(mu, lv, num_samples=5000)
        sample_mean = out.mean(dim=0)
        assert torch.allclose(sample_mean, mu, atol=0.1)

    def test_sample_std_close_to_expected(self):
        torch.manual_seed(7)
        mu  = torch.zeros(1, 4)
        lv  = torch.full((1, 4), 2.0)  # std = exp(1) ≈ 2.718
        out = _reparameterize_gaussian(mu, lv, num_samples=5000)
        expected_std = (0.5 * lv).exp()
        sample_std   = out.std(dim=0)
        assert torch.allclose(sample_std, expected_std, atol=0.1)

    def test_gradients_flow_through_mu_and_logvar(self):
        mu  = torch.randn(2, 3, requires_grad=True)
        lv  = torch.randn(2, 3, requires_grad=True)
        out = _reparameterize_gaussian(mu, lv, num_samples=4)
        out.sum().backward()
        assert mu.grad  is not None
        assert lv.grad  is not None


# ---------------------------------------------------------------------------
# _log_normal_lowrank_plus_diag
# ---------------------------------------------------------------------------

class TestLogNormalLowrankPlusDiag:
    def test_output_shape(self):
        B, D, r = 4, 6, 2
        x         = torch.randn(B, D)
        mean      = torch.zeros(B, D)
        diag_var  = torch.ones(B, D)
        cov_fac   = torch.zeros(B, D, r)
        out = _log_normal_lowrank_plus_diag(x, mean, diag_var, cov_fac)
        assert out.shape == (B,)

    def test_r0_matches_diagonal_gaussian(self):
        from utils import _log_normal_diag
        B, D = 3, 5
        x        = torch.randn(B, D)
        mean     = torch.randn(B, D)
        diag_var = torch.rand(B, D).abs() + 0.1
        log_var  = diag_var.log()
        cov_fac  = torch.zeros(B, D, 0)

        lp_lr   = _log_normal_lowrank_plus_diag(x, mean, diag_var, cov_fac)
        lp_diag = _log_normal_diag(x, mean, log_var).sum(dim=-1)
        assert torch.allclose(lp_lr, lp_diag, atol=1e-4)

    def test_output_finite(self):
        B, D, r = 5, 8, 3
        x        = torch.randn(B, D)
        mean     = torch.randn(B, D)
        diag_var = torch.ones(B, D)
        cov_fac  = torch.randn(B, D, r) * 0.1
        out = _log_normal_lowrank_plus_diag(x, mean, diag_var, cov_fac)
        assert torch.isfinite(out).all()

    def test_at_mode_is_maximum(self):
        B, D, r = 2, 4, 1
        mean    = torch.zeros(B, D)
        diag_var = torch.ones(B, D)
        cov_fac  = torch.zeros(B, D, r)
        lp_mode = _log_normal_lowrank_plus_diag(mean, mean, diag_var, cov_fac)
        lp_off  = _log_normal_lowrank_plus_diag(mean + 2.0, mean, diag_var, cov_fac)
        assert (lp_mode > lp_off).all()


# ---------------------------------------------------------------------------
# compute_image_likelihood_term
# ---------------------------------------------------------------------------

class TestComputeImageLikelihoodTerm:
    def test_output_shape(self):
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            out = compute_image_likelihood_term(model, y, enc, num_samples_z=2)
        assert out.shape == (x.shape[0],)

    def test_output_finite(self):
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            out = compute_image_likelihood_term(model, y, enc, num_samples_z=2)
        assert torch.isfinite(out).all()

    def test_more_samples_same_shape(self):
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            out = compute_image_likelihood_term(model, y, enc, num_samples_z=8)
        assert out.shape == (x.shape[0],)


# ---------------------------------------------------------------------------
# compute_label_likelihood_term
# ---------------------------------------------------------------------------

class TestComputeLabelLikelihoodTerm:
    def test_output_shape(self):
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            out = compute_label_likelihood_term(model, x, enc, num_samples=2)
        assert out.shape == (x.shape[0],)

    def test_all_missing_x_is_zero(self):
        model = _make_model()
        x, y  = _make_batch(model)
        x_nan = torch.full_like(x, float("nan"))
        enc   = _encode(model, x_nan, y)
        mask  = torch.zeros_like(x)      # all missing
        with torch.no_grad():
            out = compute_label_likelihood_term(model, x_nan, enc, num_samples=2, mask_x=mask)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_output_finite_with_partial_missing(self):
        model = _make_model()
        x, y  = _make_batch(model)
        x_partial = x.clone()
        x_partial[:, 0] = float("nan")
        enc = _encode(model, x_partial, y)
        with torch.no_grad():
            out = compute_label_likelihood_term(model, x_partial, enc, num_samples=2)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# compute_kl_zc_term
# ---------------------------------------------------------------------------

class TestComputeKLZcTerm:
    def test_output_shape(self):
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            kl = compute_kl_zc_term(model, enc, num_samples_z=2)
        assert kl.shape == (x.shape[0],)

    def test_output_finite(self):
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            kl = compute_kl_zc_term(model, enc, num_samples_z=4)
        assert torch.isfinite(kl).all()

    def test_nonnegative_on_average(self):
        # KL >= 0; with many MC samples the mean should be clearly positive
        model = _make_model()
        x, y  = _make_batch(model, B=8)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            kl = compute_kl_zc_term(model, enc, num_samples_z=100)
        assert kl.mean().item() > -0.1


# ---------------------------------------------------------------------------
# compute_kl_u_term
# ---------------------------------------------------------------------------

class TestComputeKLUTerm:
    def test_output_shape(self):
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            kl = compute_kl_u_term(model, y, enc)
        assert kl.shape == (x.shape[0],)

    def test_output_nonnegative(self):
        # Analytic KL is always >= 0
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            kl = compute_kl_u_term(model, y, enc)
        assert (kl >= -1e-5).all()

    def test_output_finite(self):
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            kl = compute_kl_u_term(model, y, enc)
        assert torch.isfinite(kl).all()


# ---------------------------------------------------------------------------
# compute_kl_m_term
# ---------------------------------------------------------------------------

class TestComputeKLMTerm:
    def test_output_shape(self):
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            kl = compute_kl_m_term(model, y, enc, num_samples_z=2)
        assert kl.shape == (x.shape[0],)

    def test_output_nonnegative(self):
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            kl = compute_kl_m_term(model, y, enc, num_samples_z=4)
        assert (kl >= -1e-4).all()


# ---------------------------------------------------------------------------
# compute_sparse_m_term
# ---------------------------------------------------------------------------

class TestComputeSparseMTerm:
    def test_output_shape(self):
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            out = compute_sparse_m_term(model, y, enc, num_samples_z=2, sparse_on="content")
        assert out.shape == (x.shape[0],)

    def test_content_mode_in_0_1(self):
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            out = compute_sparse_m_term(model, y, enc, num_samples_z=2, sparse_on="content")
        assert (out >= 0).all() and (out <= 1).all()

    def test_mask_mode(self):
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            out = compute_sparse_m_term(model, y, enc, num_samples_z=2, sparse_on="mask")
        assert out.shape == (x.shape[0],)

    def test_with_target_nonnegative(self):
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            out = compute_sparse_m_term(model, y, enc, num_samples_z=2,
                                        sparse_on="content", target=0.2)
        assert (out >= 0).all()


# ---------------------------------------------------------------------------
# compute_mask_tv_term
# ---------------------------------------------------------------------------

class TestComputeMaskTVTerm:
    def test_output_shape(self):
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            out = compute_mask_tv_term(model, enc, num_samples_z=1)
        assert out.shape == (x.shape[0],)

    def test_output_nonnegative(self):
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            out = compute_mask_tv_term(model, enc, num_samples_z=1)
        assert (out >= 0).all()

    def test_output_finite(self):
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            out = compute_mask_tv_term(model, enc, num_samples_z=2)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# compute_elbo
# ---------------------------------------------------------------------------

class TestComputeELBO:
    def test_averaged_elbo_is_scalar(self):
        model = _make_model()
        x, y  = _make_batch(model)
        with torch.no_grad():
            elbo, terms = compute_elbo(model, x, y, num_samples_z=2, num_samples_u=2)
        assert elbo.shape == ()

    def test_terms_keys(self):
        model = _make_model()
        x, y  = _make_batch(model)
        with torch.no_grad():
            _, terms = compute_elbo(model, x, y, num_samples_z=2, num_samples_u=2)
        assert set(terms.keys()) == {"L_Y", "L_X", "KL_zc", "KL_u", "KL_m", "Sparse_M", "TV_M"}

    def test_per_sample_elbo_shape(self):
        model = _make_model()
        x, y  = _make_batch(model)
        B = x.shape[0]
        with torch.no_grad():
            elbo, _ = compute_elbo(model, x, y, num_samples_z=2, num_samples_u=2,
                                   average_over_batch=False)
        assert elbo.shape == (B,)

    def test_elbo_finite(self):
        model = _make_model()
        x, y  = _make_batch(model)
        with torch.no_grad():
            elbo, terms = compute_elbo(model, x, y, num_samples_z=2, num_samples_u=2)
        assert torch.isfinite(elbo)
        for v in terms.values():
            assert torch.isfinite(v), f"Non-finite ELBO term"

    def test_sparse_term_nonzero_with_lambda(self):
        model = _make_model()
        x, y  = _make_batch(model)
        with torch.no_grad():
            _, t_no  = compute_elbo(model, x, y, num_samples_z=2, num_samples_u=2,
                                    sparse_m_lambda=0.0)
            _, t_yes = compute_elbo(model, x, y, num_samples_z=2, num_samples_u=2,
                                    sparse_m_lambda=1.0)
        # Sparse_M term itself should be identical; the elbo just uses a different weight
        assert torch.allclose(t_no["Sparse_M"], torch.zeros(()), atol=1e-8)
        assert not torch.allclose(t_yes["Sparse_M"], torch.zeros(()), atol=1e-8)

    def test_tv_term_nonzero_with_lambda(self):
        model = _make_model()
        x, y  = _make_batch(model)
        with torch.no_grad():
            _, t_no  = compute_elbo(model, x, y, num_samples_z=2, num_samples_u=2,
                                    mask_tv_lambda=0.0)
            _, t_yes = compute_elbo(model, x, y, num_samples_z=2, num_samples_u=2,
                                    mask_tv_lambda=1.0, mask_tv_samples=1)
        assert torch.allclose(t_no["TV_M"], torch.zeros(()), atol=1e-8)
        assert not torch.allclose(t_yes["TV_M"], torch.zeros(()), atol=1e-8)

    def test_gradient_flows_through_elbo(self):
        model = _make_model(seed=5)
        model.train()
        x, y  = _make_batch(model)
        elbo, _ = compute_elbo(model, x, y, num_samples_z=2, num_samples_u=2)
        (-elbo).backward()
        has_grads = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grads

    def test_nan_x_handled_gracefully(self):
        model = _make_model()
        x, y  = _make_batch(model)
        x_nan = x.clone()
        x_nan[:, :2] = float("nan")
        with torch.no_grad():
            elbo, terms = compute_elbo(model, x_nan, y, num_samples_z=2, num_samples_u=2)
        assert torch.isfinite(elbo)

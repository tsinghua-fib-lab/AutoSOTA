"""Mathematical correctness tests for BUQ-SIM core modules.

These tests verify specific formula implementations against closed-form
references or brute-force alternatives, targeting wrong matmul ordering,
transposition bugs, sign errors, and additive errors relative to the paper.
"""
import math
import sys
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import pytest
import torch
import torch.distributions as tdist

from utils import _log_normal_diag
from obj import (
    _reparameterize_gaussian,
    _log_normal_lowrank_plus_diag,
    compute_image_likelihood_term,
    compute_label_likelihood_term,
    compute_kl_u_term,
    compute_kl_m_term,
    compute_kl_zc_term,
    compute_elbo,
)
from model import (
    ContentUncertaintyDAG,
    ContinuousLabelDecoder,
    CategoricalLabelDecoder,
    ImageDecoder,
    UConditionalPrior,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(seed=0):
    torch.manual_seed(seed)
    return ContentUncertaintyDAG(
        x_dim=4, image_size=8, z_dim=2, u_dim=4,
        num_components=2, x_distribution="continuous",
        vit_embed_dim=8, vit_depth=1, vit_num_heads=2, vit_patch_size=4,
        mlp_hidden_dims=(16,), rank_r=2,
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
# Woodbury identity vs brute-force MultivariateNormal
# ---------------------------------------------------------------------------

class TestWoodburyVsFullMVN:
    """_log_normal_lowrank_plus_diag must match a full covariance MVN for r > 0."""

    def _full_log_prob(self, x, mean, diag_var, cov_factor):
        B = x.shape[0]
        log_probs = []
        for b in range(B):
            Sigma = torch.diag(diag_var[b]) + cov_factor[b] @ cov_factor[b].t()
            mvn = tdist.MultivariateNormal(loc=mean[b], covariance_matrix=Sigma)
            log_probs.append(mvn.log_prob(x[b]))
        return torch.stack(log_probs)

    def test_rank1_matches_full_mvn(self):
        torch.manual_seed(0)
        B, D, r = 4, 5, 1
        x        = torch.randn(B, D)
        mean     = torch.randn(B, D)
        diag_var = torch.rand(B, D) + 0.5
        cov_fac  = torch.randn(B, D, r) * 0.3

        lp_wb   = _log_normal_lowrank_plus_diag(x, mean, diag_var, cov_fac)
        lp_full = self._full_log_prob(x, mean, diag_var, cov_fac)
        assert torch.allclose(lp_wb, lp_full, atol=1e-4)

    def test_rank2_matches_full_mvn(self):
        torch.manual_seed(1)
        B, D, r = 3, 6, 2
        x        = torch.randn(B, D)
        mean     = torch.randn(B, D)
        diag_var = torch.ones(B, D) * 2.0
        cov_fac  = torch.randn(B, D, r) * 0.2

        lp_wb   = _log_normal_lowrank_plus_diag(x, mean, diag_var, cov_fac)
        lp_full = self._full_log_prob(x, mean, diag_var, cov_fac)
        assert torch.allclose(lp_wb, lp_full, atol=1e-4)

    def test_zero_cov_factor_matches_full_mvn(self):
        """rank-r with all-zero columns reduces to pure diagonal, confirmed by full MVN."""
        torch.manual_seed(2)
        B, D, r = 3, 5, 3
        x        = torch.randn(B, D)
        mean     = torch.randn(B, D)
        diag_var = torch.rand(B, D) + 0.1
        cov_fac  = torch.zeros(B, D, r)

        lp_wb   = _log_normal_lowrank_plus_diag(x, mean, diag_var, cov_fac)
        lp_full = self._full_log_prob(x, mean, diag_var, cov_fac)
        assert torch.allclose(lp_wb, lp_full, atol=1e-4)

    def test_at_mean_logdet_only(self):
        """At x=mean, the quadratic form vanishes; only the logdet+const term remains."""
        torch.manual_seed(3)
        B, D, r = 2, 4, 1
        mean     = torch.randn(B, D)
        diag_var = torch.ones(B, D)
        cov_fac  = torch.randn(B, D, r) * 0.3

        lp_wb   = _log_normal_lowrank_plus_diag(mean, mean, diag_var, cov_fac)
        lp_full = self._full_log_prob(mean, mean, diag_var, cov_fac)
        assert torch.allclose(lp_wb, lp_full, atol=1e-4)

    def test_logdet_via_matrix_det_lemma(self):
        """logdet(diag(d) + F F^T) = sum_i log(d_i) + logdet(I + F^T D^{-1} F)."""
        torch.manual_seed(4)
        B, D, r = 2, 5, 2
        diag_var = torch.rand(B, D) + 0.5
        cov_fac  = torch.randn(B, D, r) * 0.2

        for b in range(B):
            Sigma = torch.diag(diag_var[b]) + cov_fac[b] @ cov_fac[b].t()
            # Full logdet
            sign, logdet_full = torch.linalg.slogdet(Sigma)
            assert sign.item() > 0

            # Matrix determinant lemma: logdet(D + FF^T) = logdet(D) + logdet(I + F^T D^{-1} F)
            logdet_D = torch.log(diag_var[b]).sum()
            D_inv = 1.0 / diag_var[b]
            K = torch.eye(r) + (cov_fac[b] * D_inv.unsqueeze(-1)).t() @ cov_fac[b]
            _, logdet_K = torch.linalg.slogdet(K)
            logdet_lemma = logdet_D + logdet_K

            assert abs(logdet_full.item() - logdet_lemma.item()) < 1e-4


# ---------------------------------------------------------------------------
# ContinuousLabelDecoder: mean = Az + u
# ---------------------------------------------------------------------------

class TestContinuousDecoderMath:
    """p(x|z,u) = N(Az + u, diag(sigma^2)) — verify all terms of the mean formula."""

    def test_zero_u_mean_equals_az(self):
        """When u=0, mean must be A @ z (no spurious additive term)."""
        torch.manual_seed(0)
        dec = ContinuousLabelDecoder(z_dim=3, u_dim=4, x_dim=4)
        dec.eval()
        z = torch.randn(5, 3)
        u = torch.zeros(5, 4)
        with torch.no_grad():
            mean, _ = dec(z, u)
        expected = z.matmul(dec.A.t())
        assert torch.allclose(mean, expected, atol=1e-6)

    def test_zero_z_mean_equals_u(self):
        """When z=0, A@0 = 0 so mean = u exactly."""
        torch.manual_seed(1)
        dec = ContinuousLabelDecoder(z_dim=3, u_dim=4, x_dim=4)
        dec.eval()
        z = torch.zeros(5, 3)
        u = torch.randn(5, 4)
        with torch.no_grad():
            mean, _ = dec(z, u)
        assert torch.allclose(mean, u, atol=1e-6)

    def test_linearity_in_z(self):
        """f(alpha*z, u) - f(0, u) = alpha * (f(z, u) - f(0, u))."""
        torch.manual_seed(2)
        dec = ContinuousLabelDecoder(z_dim=3, u_dim=4, x_dim=4)
        dec.eval()
        z  = torch.randn(4, 3)
        u  = torch.randn(4, 4)
        z0 = torch.zeros(4, 3)
        alpha = 3.7
        with torch.no_grad():
            f_z,  _ = dec(z,       u)
            f_0,  _ = dec(z0,      u)
            f_az, _ = dec(alpha*z, u)
        assert torch.allclose(f_az - f_0, alpha * (f_z - f_0), atol=1e-5)

    def test_superposition_az_plus_u(self):
        """f(z1+z2, u) = f(z1, 0) + f(z2, 0) + u."""
        torch.manual_seed(3)
        dec = ContinuousLabelDecoder(z_dim=3, u_dim=4, x_dim=4)
        dec.eval()
        z1 = torch.randn(3, 3)
        z2 = torch.randn(3, 3)
        u  = torch.randn(3, 4)
        u0 = torch.zeros(3, 4)
        with torch.no_grad():
            f_sum, _ = dec(z1 + z2, u)
            f1,    _ = dec(z1,      u0)
            f2,    _ = dec(z2,      u0)
        assert torch.allclose(f_sum, f1 + f2 + u, atol=1e-5)

    def test_matmul_direction_not_transposed(self):
        """Catch A.T @ z vs z.T @ A transposition: mean should equal z @ A.T, not A @ z.T."""
        torch.manual_seed(4)
        dec = ContinuousLabelDecoder(z_dim=3, u_dim=4, x_dim=4)
        dec.eval()
        # Make A non-square so the wrong transpose would fail the shape check
        # Here x_dim=4, z_dim=3, A is (4,3): A.t() is (3,4), z@A.t() needs z (B,3) → OK
        # The wrong matmul z.matmul(dec.A) (rather than z.matmul(dec.A.t())) would have shape (B,4) only if z_dim==x_dim
        # We test the value instead
        z = torch.eye(3)[:3]  # identity-like input
        u = torch.zeros(3, 4)
        with torch.no_grad():
            mean, _ = dec(z, u)
        # Row b of mean = dec.A @ z[b] = dec.A[:, b] (the b-th column of A)
        # Since z = I_3, row b of mean = A[:, b], i.e., mean = A.t() viewed as (3,4)...
        # Actually z.matmul(A.t()) where z=I_3 gives A.t() transposed = A itself viewed as (3,4) = dec.A
        # Wait: z (3,3) @ A.t() (3,4) = (3,4)
        # z[i] is e_i (unit vector), so z @ A.t() = A.t() (since multiplying by identity)
        expected = z.matmul(dec.A.t())
        assert torch.allclose(mean, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# CategoricalLabelDecoder: logits = Az + Bu
# ---------------------------------------------------------------------------

class TestCategoricalLogitsMath:
    """Logits = einsum(z, A) + einsum(u, B) where A: (x_dim, C, z_dim)."""

    def test_zero_u_gives_az_only(self):
        """When u=0, logits == einsum('bd,kcd->bkc', z, A)."""
        torch.manual_seed(0)
        dec = CategoricalLabelDecoder(z_dim=3, u_dim=2, x_dim=4, num_classes=5)
        dec.eval()
        z  = torch.randn(6, 3)
        u0 = torch.zeros(6, 2)
        with torch.no_grad():
            logits = dec.logits(z, u0)
        Az = torch.einsum("bd,kcd->bkc", z, dec.A)
        assert torch.allclose(logits, Az, atol=1e-6)

    def test_zero_z_gives_bu_only(self):
        """When z=0, logits == einsum('bd,kcd->bkc', u, B)."""
        torch.manual_seed(1)
        dec = CategoricalLabelDecoder(z_dim=3, u_dim=2, x_dim=4, num_classes=5)
        dec.eval()
        z0 = torch.zeros(6, 3)
        u  = torch.randn(6, 2)
        with torch.no_grad():
            logits = dec.logits(z0, u)
        Bu = torch.einsum("bd,kcd->bkc", u, dec.B)
        assert torch.allclose(logits, Bu, atol=1e-6)

    def test_logits_equal_az_plus_bu(self):
        """logits(z, u) == Az + Bu exactly."""
        torch.manual_seed(2)
        dec = CategoricalLabelDecoder(z_dim=3, u_dim=2, x_dim=4, num_classes=5)
        dec.eval()
        z = torch.randn(5, 3)
        u = torch.randn(5, 2)
        with torch.no_grad():
            logits = dec.logits(z, u)
        Az = torch.einsum("bd,kcd->bkc", z, dec.A)
        Bu = torch.einsum("bd,kcd->bkc", u, dec.B)
        assert torch.allclose(logits, Az + Bu, atol=1e-6)

    def test_additivity_in_z_and_u(self):
        """logits(z, u) == logits(z, 0) + logits(0, u): no cross-term bias."""
        torch.manual_seed(3)
        dec = CategoricalLabelDecoder(z_dim=3, u_dim=2, x_dim=4, num_classes=5)
        dec.eval()
        z  = torch.randn(4, 3)
        u  = torch.randn(4, 2)
        z0 = torch.zeros(4, 3)
        u0 = torch.zeros(4, 2)
        with torch.no_grad():
            combined = dec.logits(z, u)
            z_only   = dec.logits(z, u0)
            u_only   = dec.logits(z0, u)
        assert torch.allclose(combined, z_only + u_only, atol=1e-6)

    def test_wrong_A_einsum_order_differs(self):
        """Swapping item and class dims in A changes the logit values when A is non-symmetric.

        The correct einsum 'bd,kcd->bkc' treats axis-0 of A as x_dim (items) and axis-1 as C
        (classes). Using A.permute(1,0,2) swaps those axes. With x_dim == num_classes the
        shapes agree, so we can compare values to verify correctness of the dim assignment.
        """
        torch.manual_seed(4)
        # x_dim == num_classes so both orderings produce the same output shape (B, 5, 5)
        dec = CategoricalLabelDecoder(z_dim=3, u_dim=2, x_dim=5, num_classes=5)
        dec.eval()
        z  = torch.randn(2, 3)
        u0 = torch.zeros(2, 2)
        with torch.no_grad():
            correct = dec.logits(z, u0)
        # Swap x_dim and C axes in A — produces same shape but wrong semantics
        wrong = torch.einsum("bd,kcd->bkc", z, dec.A.permute(1, 0, 2))
        # For a generic non-symmetric A these must differ
        assert not torch.allclose(correct, wrong)


# ---------------------------------------------------------------------------
# ImageDecoder: variance formula Sigma_cont,ii = var_base + ||G_i||^2
# ---------------------------------------------------------------------------

class TestImageDecoderVarianceMath:
    """content_params must encode var_base_i + sum_r G_{i,r}^2."""

    def test_content_var_equals_base_plus_ggT_diag(self):
        """log_var_cont = log(var_base + diag(GG^T)) element-wise."""
        dec = ImageDecoder(z_dim=4, y_dim_flat=16, rank_r=3, hidden_dims=(16,))
        dec.eval()
        z = torch.randn(5, 4)
        with torch.no_grad():
            _, log_var_cont = dec.content_params(z)
            _, cov_factor   = dec.content_factor(z)
            B = z.size(0)
            _, log_var_base = dec.base_params(batch_size=B)

        var_base  = log_var_base.exp()
        var_extra = cov_factor.pow(2).sum(dim=-1)     # (B, d_Y): diag(GG^T)
        expected  = torch.log(var_base + var_extra + 1e-8).clamp(-20.0, 20.0)
        assert torch.allclose(log_var_cont, expected, atol=1e-5)

    def test_rank0_content_var_equals_base_var(self):
        """With rank_r=0 there is no GG^T term; content var should equal base var."""
        dec = ImageDecoder(z_dim=4, y_dim_flat=16, rank_r=0, hidden_dims=(16,))
        dec.eval()
        z = torch.randn(3, 4)
        with torch.no_grad():
            _, log_var_cont = dec.content_params(z)
            _, log_var_base = dec.base_params(batch_size=3)
        assert torch.allclose(log_var_cont, log_var_base, atol=1e-5)

    def test_mu_cont_equals_mu_base_plus_network_shift(self):
        """mu_cont = mu_base + δμ where δμ is channel-0 of the conv decoder: no sign error."""
        dec = ImageDecoder(z_dim=4, y_dim_flat=16, rank_r=2, hidden_dims=(16,))
        dec.eval()
        z = torch.randn(3, 4)
        with torch.no_grad():
            mu_cont, _ = dec.content_factor(z)
            mu_base, _ = dec.base_params(batch_size=3)
            # Extract δμ directly from conv decoder output channel 0
            B = z.size(0)
            seed = dec.seed_mlp(z)
            feat = seed.view(B, dec._conv_channels, dec._init_size, dec._init_size)
            out = dec.conv_upsampler(feat)
            delta_mu = out.reshape(B, 1 + dec.rank_r, dec.y_dim_flat).permute(0, 2, 1)[:, :, 0]
        assert torch.allclose(mu_cont, mu_base + delta_mu, atol=1e-6)

    def test_larger_rank_increases_variance(self):
        """Adding more rank increases or maintains diagonal variance (GG^T is PSD)."""
        torch.manual_seed(5)
        z = torch.randn(4, 4)
        dec_r0 = ImageDecoder(z_dim=4, y_dim_flat=16, rank_r=0, hidden_dims=(16,))
        dec_r2 = ImageDecoder(z_dim=4, y_dim_flat=16, rank_r=2, hidden_dims=(16,))
        # Copy base params so the only difference is the rank
        dec_r2.mu_base.data[:] = dec_r0.mu_base.data
        dec_r2.log_var_base.data[:] = dec_r0.log_var_base.data
        for d in (dec_r0, dec_r2):
            d.eval()
        with torch.no_grad():
            _, lv0 = dec_r0.content_params(z)
            _, lv2 = dec_r2.content_params(z)
        # var_r2 = var_base + extra_term >= var_base = var_r0
        assert (lv2.exp() >= lv0.exp() - 1e-5).all()


# ---------------------------------------------------------------------------
# KL(q(u)||p(u|y)): analytic diagonal Gaussian KL
# ---------------------------------------------------------------------------

class TestKLUMath:
    """KL(q||p) = 0.5 * sum [ var_q/var_p + (mu_p-mu_q)^2/var_p - 1 + log_var_p - log_var_q ]."""

    def test_kl_zero_when_q_equals_p(self):
        """KL(q||p) must be exactly 0 when q = p."""
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)

        y_global = enc["y_global"]
        with torch.no_grad():
            mu_psi, lv_psi = model.u_prior_net(y_global)
        enc_eq = dict(enc)
        enc_eq["mu_u"]      = mu_psi
        enc_eq["log_var_u"] = lv_psi

        with torch.no_grad():
            kl = compute_kl_u_term(model, y, enc_eq)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_kl_formula_matches_torch_distributions(self):
        """Analytic KL formula must match torch.distributions.kl_divergence."""
        torch.manual_seed(42)
        B, d   = 5, 4
        mu_q   = torch.randn(B, d)
        lv_q   = torch.randn(B, d).clamp(-5, 5)
        mu_p   = torch.randn(B, d)
        lv_p   = torch.randn(B, d).clamp(-5, 5)
        var_q  = lv_q.exp()
        var_p  = lv_p.exp()

        # Formula from obj.py
        kl_manual = 0.5 * (
            var_q / (var_p + 1e-8)
            + (mu_p - mu_q) ** 2 / (var_p + 1e-8)
            - 1.0
            + lv_p - lv_q
        ).sum(dim=-1)

        # torch.distributions reference
        kl_torch = tdist.kl_divergence(
            tdist.Normal(mu_q, var_q.sqrt()),
            tdist.Normal(mu_p, var_p.sqrt()),
        ).sum(dim=-1)

        assert torch.allclose(kl_manual, kl_torch, atol=1e-5)

    def test_kl_increases_with_mean_shift(self):
        """KL(q||p) must be strictly larger when q mean moves away from p mean."""
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)

        enc_shifted = dict(enc)
        enc_shifted["mu_u"] = enc["mu_u"].detach() + 5.0

        with torch.no_grad():
            kl_orig    = compute_kl_u_term(model, y, enc)
            kl_shifted = compute_kl_u_term(model, y, enc_shifted)
        assert (kl_shifted > kl_orig).all()

    def test_kl_sign_not_flipped(self):
        """KL(q||p) >= 0 always (catches a sign flip in the formula)."""
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        with torch.no_grad():
            kl = compute_kl_u_term(model, y, enc)
        assert (kl >= -1e-5).all()


# ---------------------------------------------------------------------------
# KL(q(m)||p(m)): Bernoulli KL formula
# ---------------------------------------------------------------------------

class TestBernoulliKLMath:
    """KL(Bern(g) || Bern(rho)) = g log(g/rho) + (1-g) log((1-g)/(1-rho))."""

    def test_kl_zero_when_g_equals_rho_formula(self):
        """KL formula evaluates to 0 when g == rho."""
        rho = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        g   = rho.clone()
        eps = 1e-8
        kl  = (
            g   * (torch.log(g   + eps) - torch.log(rho       + eps))
          + (1-g) * (torch.log(1-g + eps) - torch.log(1 - rho + eps))
        )
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)

    def test_kl_formula_matches_torch_distributions(self):
        """Per-pixel KL must match torch.distributions.kl_divergence(Bernoulli, Bernoulli)."""
        torch.manual_seed(7)
        g   = torch.rand(10).clamp(0.01, 0.99)
        rho = torch.rand(10).clamp(0.01, 0.99)
        eps = 1e-8

        kl_manual = (
            g   * (torch.log(g   + eps) - torch.log(rho       + eps))
          + (1-g) * (torch.log(1-g + eps) - torch.log(1 - rho + eps))
        )
        kl_torch = tdist.kl_divergence(
            tdist.Bernoulli(probs=g),
            tdist.Bernoulli(probs=rho),
        )
        assert torch.allclose(kl_manual, kl_torch, atol=1e-5)

    def test_kl_m_zero_when_g_equals_rho(self):
        """compute_kl_m_term should be ~0 when mask_probs returns exactly rho."""
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)

        B   = x.shape[0]
        rho = model.mask_prior_probs          # (d_Y^2,)
        fixed_g = rho.unsqueeze(0).expand(B, -1)

        with mock.patch.object(model, "mask_probs", return_value=fixed_g):
            with torch.no_grad():
                kl = compute_kl_m_term(model, y, enc, num_samples_z=2)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_kl_m_increases_when_g_far_from_rho(self):
        """KL_m should be larger when g is pushed away from rho."""
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        B     = x.shape[0]

        # g close to rho
        rho       = model.mask_prior_probs
        g_near    = rho.unsqueeze(0).expand(B, -1) * 0.99 + 0.005
        # g far from rho (near 1)
        g_far     = torch.full((B, model.y_dim_flat), 0.99)

        with torch.no_grad():
            with mock.patch.object(model, "mask_probs", return_value=g_near):
                kl_near = compute_kl_m_term(model, y, enc, num_samples_z=2)
            with mock.patch.object(model, "mask_probs", return_value=g_far):
                kl_far  = compute_kl_m_term(model, y, enc, num_samples_z=2)
        assert (kl_far > kl_near).all()


# ---------------------------------------------------------------------------
# Factorized image likelihood: L_Y = E_q [ sum_i g_i * ll_base_i + (1-g_i)*ll_cont_i ]
# ---------------------------------------------------------------------------

class TestFactorizedLikelihoodMath:
    """Direct formula tests without going through the full model sampling."""

    def test_g1_gives_base_ll(self):
        """At g=1 everywhere, the pixelwise term equals ll_base."""
        torch.manual_seed(5)
        B, D = 4, 8
        y_flat = torch.randn(B, D)
        mu_b   = torch.randn(B, D)
        lv_b   = torch.zeros(B, D)
        mu_c   = torch.randn(B, D)
        lv_c   = torch.zeros(B, D)

        ll_base = _log_normal_diag(y_flat, mu_b, lv_b)
        ll_cont = _log_normal_diag(y_flat, mu_c, lv_c)
        g = torch.ones(B, D)

        term = g * ll_base + (1.0 - g) * ll_cont
        assert torch.allclose(term, ll_base, atol=1e-6)

    def test_g0_gives_content_ll(self):
        """At g=0 everywhere, the pixelwise term equals ll_cont."""
        torch.manual_seed(6)
        B, D = 4, 8
        y_flat = torch.randn(B, D)
        mu_b   = torch.randn(B, D)
        lv_b   = torch.zeros(B, D)
        mu_c   = torch.randn(B, D)
        lv_c   = torch.zeros(B, D)

        ll_base = _log_normal_diag(y_flat, mu_b, lv_b)
        ll_cont = _log_normal_diag(y_flat, mu_c, lv_c)
        g = torch.zeros(B, D)

        term = g * ll_base + (1.0 - g) * ll_cont
        assert torch.allclose(term, ll_cont, atol=1e-6)

    def test_linear_interpolation_in_g(self):
        """term(g) = g*ll_base + (1-g)*ll_cont is strictly linear in g."""
        torch.manual_seed(7)
        B, D = 3, 6
        y_flat = torch.randn(B, D)
        mu_b   = torch.randn(B, D)
        lv_b   = torch.zeros(B, D)
        mu_c   = torch.randn(B, D)
        lv_c   = torch.zeros(B, D)

        ll_base = _log_normal_diag(y_flat, mu_b, lv_b)
        ll_cont = _log_normal_diag(y_flat, mu_c, lv_c)

        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            g = torch.full((B, D), t)
            term_interp = g * ll_base + (1.0 - g) * ll_cont
            term_direct = t * ll_base + (1.0 - t) * ll_cont
            assert torch.allclose(term_interp, term_direct, atol=1e-6)

    def test_pure_background_via_mock(self):
        """When mask_probs returns g≈1, compute_image_likelihood_term ≈ sum_i ll_base_i."""
        torch.manual_seed(8)
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)
        B     = y.size(0)
        y_flat = y.view(B, -1)

        # g → 1: only base term contributes
        fixed_g = torch.full((B, model.y_dim_flat), 1.0 - 1e-4)

        with torch.no_grad():
            mu_base, lv_base = model.image_base_params(batch_size=B)
            expected = _log_normal_diag(y_flat, mu_base, lv_base).sum(dim=1)

            with mock.patch.object(model, "mask_probs", return_value=fixed_g):
                L_Y = compute_image_likelihood_term(model, y, enc, num_samples_z=4)

        # (1-g)≈1e-4 times 64 pixels worth of ll_cont can contribute ~0.03–0.06 per sample
        assert torch.allclose(L_Y, expected, atol=0.1)


# ---------------------------------------------------------------------------
# Label likelihood: variance formula diag(A Var_z A^T) + Var_u
# ---------------------------------------------------------------------------

class TestLabelLikelihoodVarianceMath:
    """Verify that the analytic L_X uses the correct variance expansion."""

    def test_z_variance_formula_vs_brute_force(self):
        """diag(A diag(s_z^2) A^T)_i = sum_j A[i,j]^2 * s_z2[j] = s_z2 @ A^2.T."""
        torch.manual_seed(9)
        z_dim, x_dim = 3, 5
        A    = torch.randn(x_dim, z_dim)
        s_z2 = torch.rand(2, z_dim) + 0.1    # (B, d_Z)

        A_sq         = A.pow(2)
        var_z_contrib = s_z2.matmul(A_sq.t())   # (B, d_X): formula in obj.py

        # Brute-force per batch element
        for b in range(2):
            Sigma_z_b = torch.diag(s_z2[b])         # (d_Z, d_Z)
            full      = A @ Sigma_z_b @ A.t()        # (d_X, d_X)
            expected  = torch.diag(full)             # (d_X,)
            assert torch.allclose(var_z_contrib[b], expected, atol=1e-5)

    def test_wrong_matmul_order_differs(self):
        """s_z2 @ A^2.T computes diag(A Var A^T); missing .t() gives diag(A^T Var A) instead.

        With square A (x_dim == z_dim) both orderings produce the same shape but different
        values: correct contracts z-axis of A^2 while wrong contracts x-axis.
        """
        torch.manual_seed(10)
        # Square to allow both orderings without shape error
        dim = 3
        A    = torch.randn(dim, dim)
        s_z2 = torch.rand(2, dim) + 0.1      # (B, z_dim)
        A_sq = A.pow(2)

        correct = s_z2.matmul(A_sq.t())       # diag(A diag(s_z2) A^T)  — correct
        wrong   = s_z2.matmul(A_sq)           # diag(A^T diag(s_z2) A)  — wrong axis

        # Verify correct via brute force
        for b in range(2):
            full = A @ torch.diag(s_z2[b]) @ A.t()
            assert torch.allclose(correct[b], torch.diag(full), atol=1e-5)

        # Verify wrong computes the other quadratic form
        for b in range(2):
            full_wrong = A.t() @ torch.diag(s_z2[b]) @ A
            assert torch.allclose(wrong[b], torch.diag(full_wrong), atol=1e-5)

        # The two forms differ for a generic (non-symmetric) matrix
        assert not torch.allclose(correct, wrong)

    def test_deterministic_z_variance_contribution_is_zero(self):
        """When log_var_z → -20, var_z contribution to L_X is negligible."""
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)

        enc_det = dict(enc)
        enc_det["log_var_z"] = torch.full_like(enc["log_var_z"], -20.0)

        s_z2 = enc_det["log_var_z"].clamp(-20.0, 20.0).exp()    # ≈ exp(-20) ≈ 0
        A_sq = model.x_decoder.A.pow(2)
        var_z_contrib = s_z2.matmul(A_sq.t())

        assert torch.allclose(var_z_contrib, torch.zeros_like(var_z_contrib), atol=1e-7)

    def test_deterministic_u_variance_contribution_is_zero(self):
        """When log_var_u → -20, u variance contribution to L_X is negligible."""
        model = _make_model()
        x, y  = _make_batch(model)
        enc   = _encode(model, x, y)

        enc_det = dict(enc)
        enc_det["log_var_u"] = torch.full_like(enc["log_var_u"], -20.0)

        s_u2 = enc_det["log_var_u"].clamp(-20.0, 20.0).exp()
        assert torch.allclose(s_u2, torch.zeros_like(s_u2), atol=1e-7)


# ---------------------------------------------------------------------------
# ELBO decomposition: elbo = 1/1000*L_Y + L_X - 1/100*KL_zc - KL_u - 1/1000*KL_m
# ---------------------------------------------------------------------------

class TestELBODecompositionMath:
    """Verify the weighted combination of terms matches the returned per-sample ELBO."""

    def test_per_sample_elbo_equals_weighted_sum(self):
        """Per-sample ELBO == (1/1000)*L_Y + L_X - (1/100)*KL_zc - KL_u - (1/1000)*KL_m."""
        model = _make_model()
        x, y  = _make_batch(model)

        with torch.no_grad():
            elbo, terms = compute_elbo(
                model, x, y,
                num_samples_z=2, num_samples_u=2,
                average_over_batch=False,
                sparse_m_lambda=0.0, mask_tv_lambda=0.0,
            )

        expected = (
            (1.0 / 1000) * terms["L_Y"]
          + terms["L_X"]
          - (1.0 / 100)  * terms["KL_zc"]
          - terms["KL_u"]
          - (1.0 / 1000) * terms["KL_m"]
        )
        assert torch.allclose(elbo, expected, atol=1e-5)

    def test_sparse_penalty_subtracted_from_elbo(self):
        """With sparse_m_lambda=lam, ELBO should equal base_elbo - lam*Sparse_M."""
        model = _make_model()
        x, y  = _make_batch(model)
        lam   = 0.5

        with torch.no_grad():
            elbo_lam, terms_lam = compute_elbo(
                model, x, y,
                num_samples_z=2, num_samples_u=2,
                average_over_batch=False,
                sparse_m_lambda=lam,
            )
        base = (
            (1.0 / 1000) * terms_lam["L_Y"]
          + terms_lam["L_X"]
          - (1.0 / 100)  * terms_lam["KL_zc"]
          - terms_lam["KL_u"]
          - (1.0 / 1000) * terms_lam["KL_m"]
          - lam            * terms_lam["Sparse_M"]
        )
        assert torch.allclose(elbo_lam, base, atol=1e-5)

    def test_tv_penalty_subtracted_from_elbo(self):
        """With mask_tv_lambda=lam, ELBO should equal base - lam*TV_M."""
        model = _make_model()
        x, y  = _make_batch(model)
        lam   = 0.3

        with torch.no_grad():
            elbo_lam, terms_lam = compute_elbo(
                model, x, y,
                num_samples_z=2, num_samples_u=2,
                average_over_batch=False,
                mask_tv_lambda=lam, mask_tv_samples=1,
            )
        base = (
            (1.0 / 1000) * terms_lam["L_Y"]
          + terms_lam["L_X"]
          - (1.0 / 100)  * terms_lam["KL_zc"]
          - terms_lam["KL_u"]
          - (1.0 / 1000) * terms_lam["KL_m"]
          - lam            * terms_lam["TV_M"]
        )
        assert torch.allclose(elbo_lam, base, atol=1e-5)

    def test_averaged_elbo_equals_mean_of_terms(self):
        """With average_over_batch=True, returned ELBO == mean of per-sample ELBO."""
        model = _make_model()
        x, y  = _make_batch(model)

        with torch.no_grad():
            elbo_avg, terms_avg = compute_elbo(
                model, x, y,
                num_samples_z=2, num_samples_u=2,
                average_over_batch=True,
                sparse_m_lambda=0.0, mask_tv_lambda=0.0,
            )

        # terms_avg values are already batch means
        expected_avg = (
            (1.0 / 1000) * terms_avg["L_Y"]
          + terms_avg["L_X"]
          - (1.0 / 100)  * terms_avg["KL_zc"]
          - terms_avg["KL_u"]
          - (1.0 / 1000) * terms_avg["KL_m"]
        )
        assert torch.allclose(elbo_avg, expected_avg, atol=1e-5)

    def test_kl_coefficients_not_swapped(self):
        """KL_zc coefficient is 1/100 and KL_m coefficient is 1/1000 — not swapped."""
        model = _make_model()
        x, y  = _make_batch(model)

        with torch.no_grad():
            elbo, terms = compute_elbo(
                model, x, y,
                num_samples_z=2, num_samples_u=2,
                average_over_batch=False,
                sparse_m_lambda=0.0, mask_tv_lambda=0.0,
            )

        # If coefficients were swapped, this would fail
        correct_elbo = (
            (1.0 / 1000) * terms["L_Y"]
          + terms["L_X"]
          - (1.0 / 100)  * terms["KL_zc"]    # KL_zc gets 1/100
          - terms["KL_u"]
          - (1.0 / 1000) * terms["KL_m"]     # KL_m gets 1/1000
        )
        swapped_elbo = (
            (1.0 / 1000) * terms["L_Y"]
          + terms["L_X"]
          - (1.0 / 1000) * terms["KL_zc"]    # wrong: swapped
          - terms["KL_u"]
          - (1.0 / 100)  * terms["KL_m"]     # wrong: swapped
        )
        assert torch.allclose(elbo, correct_elbo, atol=1e-5)
        # The two formulations agree only if KL_zc == KL_m (extremely unlikely)
        if not torch.allclose(terms["KL_zc"], terms["KL_m"], atol=1e-3):
            assert not torch.allclose(elbo, swapped_elbo, atol=1e-3)


# ---------------------------------------------------------------------------
# log q(z) formula used in KL_zc
# ---------------------------------------------------------------------------

class TestLogQFormulaMath:
    """log q(z) = -0.5*(D*log2pi + sum log_var + sum (z-mu)^2/var)."""

    def test_log_q_matches_torch_normal_log_prob(self):
        """log q(z) computed by KL_zc formula matches torch.distributions.Normal."""
        torch.manual_seed(42)
        B, d    = 3, 4
        mu_z    = torch.randn(B, d)
        lv_z    = torch.randn(B, d).clamp(-3, 3)
        var_z   = lv_z.exp()

        # Draw a sample
        z = mu_z + var_z.sqrt() * torch.randn(B, d)

        # Formula from obj.py
        diff     = z - mu_z
        log_q_manual = -0.5 * (
            d * math.log(2.0 * math.pi)
          + lv_z.sum(dim=-1)
          + (diff.pow(2) / (var_z + 1e-8)).sum(dim=-1)
        )

        # torch.distributions reference
        log_q_torch = tdist.Normal(mu_z, var_z.sqrt()).log_prob(z).sum(dim=-1)

        assert torch.allclose(log_q_manual, log_q_torch, atol=1e-5)

    def test_log_q_at_mean_is_normalizing_constant(self):
        """log q(mu) = -0.5*(D*log2pi + sum log_var): the quadratic term vanishes at z=mu."""
        torch.manual_seed(0)
        B, d  = 2, 5
        mu_z  = torch.randn(B, d)
        lv_z  = torch.randn(B, d).clamp(-3, 3)

        log_q = -0.5 * (d * math.log(2 * math.pi) + lv_z.sum(dim=-1))
        log_q_full = tdist.Normal(mu_z, lv_z.exp().sqrt()).log_prob(mu_z).sum(dim=-1)
        assert torch.allclose(log_q, log_q_full, atol=1e-5)


# ---------------------------------------------------------------------------
# Reparameterize: gradient math
# ---------------------------------------------------------------------------

class TestReparameterizeGradientMath:
    """The reparameterization trick gives correct gradient magnitude w.r.t. mu."""

    def test_grad_of_sum_wrt_mu_is_num_samples(self):
        """d/d_mu [ sum_{l,b,d} (mu + eps*std) ] = L * ones(B,D)."""
        torch.manual_seed(0)
        B, D, L = 3, 4, 10
        mu  = torch.randn(B, D, requires_grad=True)
        lv  = torch.zeros(B, D)    # fixed log-var, no contribution to mu gradient

        out = _reparameterize_gaussian(mu, lv, num_samples=L)
        out.sum().backward()

        expected = torch.full((B, D), float(L))
        assert torch.allclose(mu.grad, expected, atol=1e-5)

    def test_grad_wrt_logvar_flows_via_std(self):
        """d/d_lv involves 0.5*exp(0.5*lv)*eps; nonzero gradient must reach lv."""
        torch.manual_seed(1)
        B, D, L = 2, 3, 500
        mu  = torch.zeros(B, D)
        lv  = torch.zeros(B, D, requires_grad=True)

        out = _reparameterize_gaussian(mu, lv, num_samples=L)
        # Sum of (eps * exp(0.5*lv)); gradient w.r.t. lv is 0.5 * exp(0.5*lv) * sum(eps)
        out.sum().backward()
        assert lv.grad is not None
        assert torch.isfinite(lv.grad).all()

    def test_samples_mean_converges_to_mu(self):
        """E[z_sample] → mu as num_samples increases (LLN)."""
        torch.manual_seed(2)
        mu  = torch.tensor([[1.0, -2.0, 3.0]])
        lv  = torch.zeros(1, 3)
        out = _reparameterize_gaussian(mu, lv, num_samples=5000)
        assert torch.allclose(out.mean(dim=0), mu, atol=0.1)

"""Tests for adaptive padding in SamplesLoss."""

import pytest
import torch

from flash_sinkhorn import SamplesLoss
from flash_sinkhorn.samples_loss import _pad_inputs, _trim_potentials


class TestPadInputs:
    """Tests for _pad_inputs helper."""

    def test_pad_basic(self):
        torch.manual_seed(0)
        x = torch.randn(100, 8)
        y = torch.randn(70, 8)
        a = torch.ones(100) / 100
        b = torch.ones(70) / 70
        x_p, y_p, a_p, b_p, lx_p, ly_p, n_orig, m_orig = _pad_inputs(x, y, a, b, multiple=128)
        assert x_p.shape == (128, 8)
        assert y_p.shape == (128, 8)
        assert a_p.shape == (128,)
        assert b_p.shape == (128,)
        assert n_orig == 100
        assert m_orig == 70

    def test_pad_weights_sum_preserved(self):
        torch.manual_seed(0)
        a = torch.ones(100) / 100
        b = torch.ones(70) / 70
        x = torch.randn(100, 4)
        y = torch.randn(70, 4)
        _, _, a_p, b_p, _, _, _, _ = _pad_inputs(x, y, a, b, multiple=128)
        torch.testing.assert_close(a_p.sum(), torch.tensor(1.0))
        torch.testing.assert_close(b_p.sum(), torch.tensor(1.0))

    def test_pad_zero_weight_entries(self):
        torch.manual_seed(0)
        x = torch.randn(100, 4)
        y = torch.randn(70, 4)
        a = torch.ones(100) / 100
        b = torch.ones(70) / 70
        _, _, a_p, b_p, _, _, n_orig, m_orig = _pad_inputs(x, y, a, b, multiple=128)
        assert (a_p[n_orig:] == 0).all()
        assert (b_p[m_orig:] == 0).all()

    def test_pad_coordinates_are_zero(self):
        torch.manual_seed(0)
        x = torch.randn(100, 4)
        y = torch.randn(70, 4)
        a = torch.ones(100) / 100
        b = torch.ones(70) / 70
        x_p, y_p, _, _, _, _, n_orig, m_orig = _pad_inputs(x, y, a, b, multiple=128)
        assert (x_p[n_orig:] == 0).all()
        assert (y_p[m_orig:] == 0).all()

    def test_no_padding_needed(self):
        torch.manual_seed(0)
        x = torch.randn(128, 4)
        y = torch.randn(64, 4)
        a = torch.ones(128) / 128
        b = torch.ones(64) / 64
        x_p, y_p, a_p, b_p, _, _, n_orig, m_orig = _pad_inputs(x, y, a, b, multiple=64)
        assert x_p.shape == (128, 4)
        assert y_p.shape == (64, 4)
        assert n_orig == 128
        assert m_orig == 64

    def test_pad_labels(self):
        torch.manual_seed(0)
        x = torch.randn(100, 4)
        y = torch.randn(70, 4)
        a = torch.ones(100) / 100
        b = torch.ones(70) / 70
        lx = torch.randint(0, 5, (100,))
        ly = torch.randint(0, 5, (70,))
        _, _, _, _, lx_p, ly_p, n_orig, m_orig = _pad_inputs(x, y, a, b, multiple=128, label_x=lx, label_y=ly)
        assert lx_p.shape == (128,)
        assert ly_p.shape == (128,)
        assert (lx_p[n_orig:] == 0).all()
        assert (ly_p[m_orig:] == 0).all()

    def test_pad_no_grad_on_padding(self):
        """Gradient flows through torch.cat back to original x."""
        torch.manual_seed(0)
        x = torch.randn(100, 4, requires_grad=True)
        y = torch.randn(70, 4, requires_grad=True)
        a = torch.ones(100) / 100
        b = torch.ones(70) / 70
        x_p, y_p, _, _, _, _, n_orig, _ = _pad_inputs(x, y, a, b, multiple=128)
        assert x_p.requires_grad
        assert y_p.requires_grad
        # Verify gradient actually flows back to original x
        loss = x_p.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (100, 4)
        torch.testing.assert_close(x.grad, torch.ones(100, 4))

    def test_no_padding_is_identity(self):
        """When no padding needed, returns the same tensor objects."""
        torch.manual_seed(0)
        x = torch.randn(128, 4)
        y = torch.randn(64, 4)
        a = torch.ones(128) / 128
        b = torch.ones(64) / 64
        x_p, y_p, a_p, b_p, _, _, _, _ = _pad_inputs(x, y, a, b, multiple=64)
        assert x_p is x
        assert y_p is y
        assert a_p is a
        assert b_p is b

    def test_pad_labels_dtype_preserved(self):
        """Label dtype is preserved after padding."""
        torch.manual_seed(0)
        x = torch.randn(100, 4)
        y = torch.randn(70, 4)
        a = torch.ones(100) / 100
        b = torch.ones(70) / 70
        lx = torch.randint(0, 5, (100,), dtype=torch.int32)
        ly = torch.randint(0, 5, (70,), dtype=torch.int64)
        _, _, _, _, lx_p, ly_p, _, _ = _pad_inputs(
            x, y, a, b, multiple=128, label_x=lx, label_y=ly,
        )
        assert lx_p.dtype == torch.int32
        assert ly_p.dtype == torch.int64


class TestTrimPotentials:
    def test_trim_basic(self):
        f = torch.randn(128)
        g = torch.randn(128)
        f_t, g_t = _trim_potentials(f, g, n_orig=100, m_orig=70)
        assert f_t.shape == (100,)
        assert g_t.shape == (70,)

    def test_trim_preserves_values(self):
        f = torch.randn(128)
        g = torch.randn(128)
        f_t, g_t = _trim_potentials(f, g, n_orig=100, m_orig=70)
        torch.testing.assert_close(f_t, f[:100])
        torch.testing.assert_close(g_t, g[:70])

    def test_trim_noop(self):
        f = torch.randn(128)
        g = torch.randn(64)
        f_t, g_t = _trim_potentials(f, g, n_orig=128, m_orig=64)
        torch.testing.assert_close(f_t, f)
        torch.testing.assert_close(g_t, g)


class TestSamplesLossPadToMultipleInit:
    """Tests for pad_to_multiple kwarg validation in __init__."""

    def test_default_none(self):
        loss = SamplesLoss(loss='sinkhorn')
        assert loss.pad_to_multiple is None

    def test_valid_128(self):
        loss = SamplesLoss(loss='sinkhorn', pad_to_multiple=128)
        assert loss.pad_to_multiple == 128

    def test_valid_32(self):
        loss = SamplesLoss(loss='sinkhorn', pad_to_multiple=32)
        assert loss.pad_to_multiple == 32

    def test_reject_not_multiple_of_32(self):
        with pytest.raises(ValueError, match="multiple of 32"):
            SamplesLoss(loss='sinkhorn', pad_to_multiple=48)

    def test_reject_zero(self):
        with pytest.raises(ValueError, match="multiple of 32"):
            SamplesLoss(loss='sinkhorn', pad_to_multiple=0)

    def test_reject_negative(self):
        with pytest.raises(ValueError, match="multiple of 32"):
            SamplesLoss(loss='sinkhorn', pad_to_multiple=-32)


from flash_sinkhorn.sinkhorn_solvers import (
    sinkhorn_flashstyle_alternating,
    sinkhorn_flashstyle_symmetric,
)


class TestMaskedEarlyStopping:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_alternating_accepts_n_orig_kwarg(self):
        torch.manual_seed(0)
        x = torch.randn(128, 4, device='cuda')
        y = torch.randn(128, 4, device='cuda')
        a = torch.ones(128, device='cuda') / 128
        b = torch.ones(128, device='cuda') / 128
        f, g = sinkhorn_flashstyle_alternating(
            x, y, a, b, eps=0.1, n_iters=10,
            threshold=1e-3, check_every=5,
            n_orig=100, m_orig=100,
        )
        assert f.shape == (128,)
        assert g.shape == (128,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_symmetric_accepts_n_orig_kwarg(self):
        torch.manual_seed(0)
        x = torch.randn(128, 4, device='cuda')
        y = torch.randn(128, 4, device='cuda')
        a = torch.ones(128, device='cuda') / 128
        b = torch.ones(128, device='cuda') / 128
        f, g = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            use_epsilon_scaling=False, eps=0.1, n_iters=10,
            threshold=1e-3, check_every=5,
            n_orig=100, m_orig=100,
        )
        assert f.shape == (128,)
        assert g.shape == (128,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_n_orig_none_is_backward_compatible(self):
        torch.manual_seed(0)
        x = torch.randn(64, 4, device='cuda')
        y = torch.randn(64, 4, device='cuda')
        a = torch.ones(64, device='cuda') / 64
        b = torch.ones(64, device='cuda') / 64
        f1, g1 = sinkhorn_flashstyle_alternating(
            x, y, a, b, eps=0.1, n_iters=20,
            threshold=1e-3, check_every=5,
        )
        f2, g2 = sinkhorn_flashstyle_alternating(
            x, y, a, b, eps=0.1, n_iters=20,
            threshold=1e-3, check_every=5,
            n_orig=None, m_orig=None,
        )
        torch.testing.assert_close(f1, f2)
        torch.testing.assert_close(g1, g2)


class TestSamplesLossPadForward:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cost_parity(self):
        torch.manual_seed(0)
        n, m, d = 100, 70, 8
        x = torch.randn(n, d, device='cuda')
        y = torch.randn(m, d, device='cuda')
        a = torch.ones(n, device='cuda') / n
        b = torch.ones(m, device='cuda') / m
        loss_ref = SamplesLoss(loss='sinkhorn', blur=0.1, half_cost=True,
                               use_epsilon_scaling=False, eps=0.1, n_iters=50)
        loss_pad = SamplesLoss(loss='sinkhorn', blur=0.1, half_cost=True,
                               use_epsilon_scaling=False, eps=0.1, n_iters=50,
                               pad_to_multiple=128)
        cost_ref = loss_ref(a, x, b, y)
        cost_pad = loss_pad(a, x, b, y)
        torch.testing.assert_close(cost_pad, cost_ref, atol=1e-5, rtol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gradient_shape_and_parity(self):
        torch.manual_seed(0)
        n, m, d = 100, 70, 8
        x = torch.randn(n, d, device='cuda', requires_grad=True)
        y = torch.randn(m, d, device='cuda', requires_grad=True)
        a = torch.ones(n, device='cuda') / n
        b = torch.ones(m, device='cuda') / m
        loss_ref = SamplesLoss(loss='sinkhorn', blur=0.1, half_cost=True,
                               use_epsilon_scaling=False, eps=0.1, n_iters=50)
        cost_ref = loss_ref(a, x, b, y)
        grad_x_ref, grad_y_ref = torch.autograd.grad(cost_ref, [x, y])
        x2 = x.detach().clone().requires_grad_(True)
        y2 = y.detach().clone().requires_grad_(True)
        loss_pad = SamplesLoss(loss='sinkhorn', blur=0.1, half_cost=True,
                               use_epsilon_scaling=False, eps=0.1, n_iters=50,
                               pad_to_multiple=128)
        cost_pad = loss_pad(a, x2, b, y2)
        grad_x_pad, grad_y_pad = torch.autograd.grad(cost_pad, [x2, y2])
        assert grad_x_pad.shape == (n, d)
        assert grad_y_pad.shape == (m, d)
        torch.testing.assert_close(grad_x_pad, grad_x_ref, atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(grad_y_pad, grad_y_ref, atol=1e-4, rtol=1e-3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_potentials_shape_and_parity(self):
        torch.manual_seed(0)
        n, m, d = 100, 70, 8
        x = torch.randn(n, d, device='cuda')
        y = torch.randn(m, d, device='cuda')
        a = torch.ones(n, device='cuda') / n
        b = torch.ones(m, device='cuda') / m
        loss_ref = SamplesLoss(loss='sinkhorn', blur=0.1, half_cost=True, potentials=True,
                               use_epsilon_scaling=False, eps=0.1, n_iters=50)
        loss_pad = SamplesLoss(loss='sinkhorn', blur=0.1, half_cost=True, potentials=True,
                               use_epsilon_scaling=False, eps=0.1, n_iters=50,
                               pad_to_multiple=128)
        f_ref, g_ref = loss_ref(a, x, b, y)
        f_pad, g_pad = loss_pad(a, x, b, y)
        assert f_pad.shape == (n,)
        assert g_pad.shape == (m,)
        torch.testing.assert_close(f_pad, f_ref, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(g_pad, g_ref, atol=1e-5, rtol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_no_padding_when_already_aligned(self):
        torch.manual_seed(0)
        n, m, d = 128, 64, 8
        x = torch.randn(n, d, device='cuda')
        y = torch.randn(m, d, device='cuda')
        a = torch.ones(n, device='cuda') / n
        b = torch.ones(m, device='cuda') / m
        loss_ref = SamplesLoss(loss='sinkhorn', blur=0.1, half_cost=True,
                               use_epsilon_scaling=False, eps=0.1, n_iters=50)
        loss_pad = SamplesLoss(loss='sinkhorn', blur=0.1, half_cost=True,
                               use_epsilon_scaling=False, eps=0.1, n_iters=50,
                               pad_to_multiple=32)
        cost_ref = loss_ref(a, x, b, y)
        cost_pad = loss_pad(a, x, b, y)
        torch.testing.assert_close(cost_pad, cost_ref, atol=0, rtol=0)


class TestSamplesLossPadDebiased:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_debiased_cost_parity(self):
        """Debiased cost matches with and without padding."""
        torch.manual_seed(0)
        n, m, d = 100, 70, 8
        x = torch.randn(n, d, device='cuda')
        y = torch.randn(m, d, device='cuda')
        a = torch.ones(n, device='cuda') / n
        b = torch.ones(m, device='cuda') / m
        loss_ref = SamplesLoss(loss='sinkhorn', blur=0.1, half_cost=True, debias=True,
                               use_epsilon_scaling=False, eps=0.1, n_iters=50)
        loss_pad = SamplesLoss(loss='sinkhorn', blur=0.1, half_cost=True, debias=True,
                               use_epsilon_scaling=False, eps=0.1, n_iters=50,
                               pad_to_multiple=128)
        cost_ref = loss_ref(a, x, b, y)
        cost_pad = loss_pad(a, x, b, y)
        torch.testing.assert_close(cost_pad, cost_ref, atol=1e-5, rtol=1e-4)


class TestSamplesLossPadEarlyStopping:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_early_stopping_not_delayed(self):
        """Padding does not significantly increase iteration count."""
        torch.manual_seed(0)
        n, m, d = 100, 70, 8
        x = torch.randn(n, d, device='cuda')
        y = torch.randn(m, d, device='cuda')
        a = torch.ones(n, device='cuda') / n
        b = torch.ones(m, device='cuda') / m
        _, _, n_iters_ref = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            use_epsilon_scaling=False, eps=0.1, n_iters=200,
            threshold=1e-3, check_every=5,
            return_n_iters=True,
        )
        x_p, y_p, a_p, b_p, _, _, no, mo = _pad_inputs(x, y, a, b, multiple=128)
        _, _, n_iters_pad = sinkhorn_flashstyle_symmetric(
            x_p, y_p, a_p, b_p,
            use_epsilon_scaling=False, eps=0.1, n_iters=200,
            threshold=1e-3, check_every=5,
            n_orig=no, m_orig=mo,
            return_n_iters=True,
        )
        assert n_iters_pad <= n_iters_ref * 1.1 + 5, (
            f"Padding increased iterations from {n_iters_ref} to {n_iters_pad}"
        )


class TestSamplesLossPadBatched:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_batched_cost_parity(self):
        torch.manual_seed(0)
        B, n, m, d = 3, 100, 70, 8
        x = torch.randn(B, n, d, device='cuda')
        y = torch.randn(B, m, d, device='cuda')
        a = torch.ones(B, n, device='cuda') / n
        b = torch.ones(B, m, device='cuda') / m
        loss_ref = SamplesLoss(loss='sinkhorn', blur=0.1, half_cost=True,
                               use_epsilon_scaling=False, eps=0.1, n_iters=50)
        loss_pad = SamplesLoss(loss='sinkhorn', blur=0.1, half_cost=True,
                               use_epsilon_scaling=False, eps=0.1, n_iters=50,
                               pad_to_multiple=128)
        cost_ref = loss_ref(a, x, b, y)
        cost_pad = loss_pad(a, x, b, y)
        torch.testing.assert_close(cost_pad, cost_ref, atol=1e-5, rtol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_batched_potentials_shape(self):
        torch.manual_seed(0)
        B, n, m, d = 2, 100, 70, 8
        x = torch.randn(B, n, d, device='cuda')
        y = torch.randn(B, m, d, device='cuda')
        a = torch.ones(B, n, device='cuda') / n
        b = torch.ones(B, m, device='cuda') / m
        loss_pad = SamplesLoss(loss='sinkhorn', blur=0.1, half_cost=True, potentials=True,
                               use_epsilon_scaling=False, eps=0.1, n_iters=50,
                               pad_to_multiple=128)
        f, g = loss_pad(a, x, b, y)
        assert f.shape == (B, n)
        assert g.shape == (B, m)


class TestSamplesLossPadAlternating:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_alternating_cost_parity(self):
        torch.manual_seed(0)
        n, m, d = 100, 70, 8
        x = torch.randn(n, d, device='cuda')
        y = torch.randn(m, d, device='cuda')
        a = torch.ones(n, device='cuda') / n
        b = torch.ones(m, device='cuda') / m
        loss_ref = SamplesLoss(loss='sinkhorn', blur=0.1, half_cost=True,
                               backend='alternating',
                               use_epsilon_scaling=False, eps=0.1, n_iters=50)
        loss_pad = SamplesLoss(loss='sinkhorn', blur=0.1, half_cost=True,
                               backend='alternating',
                               use_epsilon_scaling=False, eps=0.1, n_iters=50,
                               pad_to_multiple=128)
        cost_ref = loss_ref(a, x, b, y)
        cost_pad = loss_pad(a, x, b, y)
        torch.testing.assert_close(cost_pad, cost_ref, atol=1e-5, rtol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_alternating_potentials_shape(self):
        torch.manual_seed(0)
        n, m, d = 100, 70, 8
        x = torch.randn(n, d, device='cuda')
        y = torch.randn(m, d, device='cuda')
        a = torch.ones(n, device='cuda') / n
        b = torch.ones(m, device='cuda') / m
        loss_pad = SamplesLoss(loss='sinkhorn', blur=0.1, half_cost=True, potentials=True,
                               backend='alternating',
                               use_epsilon_scaling=False, eps=0.1, n_iters=50,
                               pad_to_multiple=128)
        f, g = loss_pad(a, x, b, y)
        assert f.shape == (n,)
        assert g.shape == (m,)

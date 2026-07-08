"""Tests for the diffusion samplers in ``calibrated_guidance.inference``.

We test the two implemented samplers (``flow_matching`` and ``edm_sde``) for
their structural contract (shapes, trajectories, device, x_t override) plus, for
flow matching, an exact analytic check: with a constant mean prediction the
linear flow-matching update is a per-step convex combination toward that
constant, and the residual is scaled by ``t_final / t_initial``.
"""

import pytest
import torch

from calibrated_guidance.inference import (
    ddim,
    ddpm,
    edm,
    edm_sde,
    flow_matching,
)


# --------------------------------------------------------------------------- #
# flow_matching
# --------------------------------------------------------------------------- #
def test_flow_matching_output_shape():
    """flow_matching returns [n_samples, *shape]."""
    ts = torch.linspace(1.0, 1e-3, 50)
    out = flow_matching(
        lambda xt, t: torch.zeros_like(xt), ts, shape=(3,), n_samples=11,
        use_tqdm=False,
    )
    assert out.shape == (11, 3)


def test_flow_matching_constant_mean_drives_to_constant():
    """A constant predicted mean c drives all samples to c at t -> 0.

    For x̂0 == c the update x <- x + (t - t_next)/t * (c - x) is a convex
    combination, and the residual telescopes to (t_final / t_initial) * (x_1 - c).
    With t_initial = 1 and t_final = 1e-3 the residual is ~1e-3, so the final
    samples sit essentially on c.
    """
    torch.manual_seed(0)
    c = torch.tensor([2.0, -1.0, 0.5])
    ts = torch.linspace(1.0, 1e-3, 200)

    def mean_fn(xt, t):
        return c.expand_as(xt)

    out = flow_matching(mean_fn, ts, shape=(3,), n_samples=64, use_tqdm=False)
    # Every sample is essentially c (residual scaled by t_final/t_initial=1e-3).
    torch.testing.assert_close(out, c.expand(64, 3), rtol=0, atol=2e-2)


def test_flow_matching_return_intermediate_trajectory_length():
    """return_intermediate yields a trajectory of length num_steps + 1.

    There are len(time_steps) - 1 update steps, plus the initial state.
    """
    ts = torch.linspace(1.0, 1e-3, 30)
    out, traj = flow_matching(
        lambda xt, t: torch.zeros_like(xt), ts, shape=(2,), n_samples=5,
        use_tqdm=False, return_intermediate=True,
    )
    assert isinstance(traj, list)
    assert len(traj) == len(ts)  # initial + one per update step
    assert traj[0].shape == (5, 2)
    torch.testing.assert_close(traj[-1], out)


def test_flow_matching_device_defaults_to_time_steps_device():
    """device defaults to the time-step tensor's device (CPU here)."""
    ts = torch.linspace(1.0, 1e-3, 10)
    out = flow_matching(
        lambda xt, t: torch.zeros_like(xt), ts, shape=(1,), n_samples=3,
        use_tqdm=False,
    )
    assert out.device == ts.device


# --------------------------------------------------------------------------- #
# edm_sde
# --------------------------------------------------------------------------- #
def _edm_schedules(num_steps):
    """Small, well-behaved monotone schedule tensors for edm_sde.

    edm_sde indexes sigma_steps / factor_steps / scaling_factor / scaling_steps
    by step i in range(num_steps) (num_steps = len(sigma_steps) - 1).  We give
    each array num_steps + 1 entries to be safe with the [0] start access.
    """
    n = num_steps + 1
    sigma_steps = torch.linspace(1.0, 0.1, n)
    factor_steps = torch.linspace(0.05, 0.01, n)
    scaling_factor = torch.full((n,), 0.99)
    scaling_steps = torch.linspace(1.0, 0.5, n)
    return sigma_steps, factor_steps, scaling_factor, scaling_steps


def test_edm_sde_output_shape():
    """edm_sde returns [n_samples, *shape]."""
    torch.manual_seed(0)
    num_steps = 20
    schedules = _edm_schedules(num_steps)

    def mean_fn(xt, t):
        # Must be detached: edm_sde asserts not guided_mean.requires_grad.
        return torch.zeros_like(xt)

    out = edm_sde(
        mean_fn, *schedules, shape=(3,), n_samples=8, device="cpu",
        use_tqdm=False,
    )
    assert out.shape == (8, 3)
    assert torch.isfinite(out).all()


def test_edm_sde_rejects_grad_attached_mean():
    """edm_sde asserts the guided mean is detached (no grad)."""
    num_steps = 5
    schedules = _edm_schedules(num_steps)

    def bad_mean_fn(xt, t):
        # Return a tensor that requires grad to trip the internal assertion.
        return (xt * torch.ones(1, requires_grad=True)).clone()

    with pytest.raises(AssertionError):
        edm_sde(
            bad_mean_fn, *schedules, shape=(2,), n_samples=4, device="cpu",
            use_tqdm=False,
        )


def test_edm_sde_x_t_start_override_and_intermediate():
    """x_t_start seeds the chain; return_intermediate gives the full trajectory."""
    torch.manual_seed(0)
    num_steps = 12
    schedules = _edm_schedules(num_steps)
    x0 = torch.full((6, 2), 0.3)

    def mean_fn(xt, t):
        return torch.zeros_like(xt)

    out, traj = edm_sde(
        mean_fn, *schedules, shape=(2,), n_samples=6, device="cpu",
        use_tqdm=False, x_t_start=x0, return_intermediate=True,
    )
    assert out.shape == (6, 2)
    assert len(traj) == num_steps + 1
    # The first recorded state is exactly the override.
    torch.testing.assert_close(traj[0], x0.to("cpu"))
    torch.testing.assert_close(traj[-1], out)


def test_edm_sde_deterministic_when_not_sde():
    """With sde=False there is no injected noise, so runs are reproducible."""
    num_steps = 10
    schedules = _edm_schedules(num_steps)
    x0 = torch.full((4, 2), 0.1)

    def mean_fn(xt, t):
        return torch.zeros_like(xt)

    a = edm_sde(
        mean_fn, *schedules, shape=(2,), n_samples=4, device="cpu",
        use_tqdm=False, x_t_start=x0, sde=False,
    )
    b = edm_sde(
        mean_fn, *schedules, shape=(2,), n_samples=4, device="cpu",
        use_tqdm=False, x_t_start=x0, sde=False,
    )
    torch.testing.assert_close(a, b)


# --------------------------------------------------------------------------- #
# Stub samplers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fn", [ddim, edm, ddpm])
def test_stub_samplers_raise_not_implemented(fn):
    """ddim / edm / ddpm are explicit stubs that must raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        fn(lambda *a, **k: None)

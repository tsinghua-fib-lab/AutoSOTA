import numpy as np
import torch
import torch.nn.functional as F

import mech_design.mechanism as mechanism_module
import tools.visualize as viz

COST_PRICE = 0.6

def constant_cost(choice: torch.Tensor) -> torch.Tensor:
    return torch.full(
        (choice.shape[0],),
        COST_PRICE,
        dtype=choice.dtype,
        device=choice.device,
    )


def dot_kernel(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return (X * Y).sum(dim=-1)


def test_stats_work():
    """Uniform types over [0,1] should accept offers priced at 0.5."""
    mech = mechanism_module.Mechanism(
        npoints=1,
        kernel=dot_kernel,
        y_dim=1,
        temp=1.0,
        is_Y_parameter=False,
        is_there_default=True,
        y_min=0.0,
        y_max=1.0,
    )
    with torch.no_grad():
        mech.Y_rest_raw.fill_(1.0)
        mech.intercept_rest.fill_(0.5)

    xs, q, _, _, _ = viz.eval_mech_1d(mech, N=401)
    cutoff = viz.estimate_cutoff(xs, q, level=0.5)
    assert abs(cutoff - 0.5) < 5e-3
    q = np.array(q)
    below_price = q[xs < 0.49]
    above_price = q[xs > 0.51]
    assert np.allclose(below_price, 0.0, atol=5e-3)
    assert np.allclose(above_price, 1.0, atol=5e-3)


def test_default_intercept_offset():
    """Setting `default_intercept` updates the default option's intercept."""
    level = -0.25
    mech = mechanism_module.Mechanism(
        npoints=3,
        kernel=dot_kernel,
        y_dim=1,
        temp=5.0,
        is_Y_parameter=True,
        is_there_default=True,
        default_intercept=level,
    )
    expected = torch.full_like(mech.intercept0, level)
    torch.testing.assert_close(mech.intercept0, expected)


def test_training_mechanism(tmp_path):
    """Short Mechanism training run produces valid mechanism metadata."""
    dim = 1
    npoints = 10
    patience = 50
    nsteps = 400
    sample = torch.rand(400, dim)
    model_kwargs = {
        "npoints": npoints,
        "kernel": dot_kernel,
        "y_dim": dim,
        "temp": 80.0,
        "is_Y_parameter": True,
        "is_there_default": True,
        "y_min": 0.0,
        "y_max": 1.0,
    }
    mechanism_instance = mechanism_module.Mechanism(**model_kwargs)
    mechanism, mechanism_data = mechanism_instance.fit(
        sample,
        modes=["soft"],
        compile=False,
        optimizers_kwargs_dict={"soft": {"lr": 0.05}},
        schedulers_kwargs_dict={
            "soft": {
                "patience": patience,
                "threshold": 1e-3,
                "factor": 0.5,
                "cooldown": 1,
                "eps": 1e-6,
            },
        },
        train_kwargs={
            "nsteps": nsteps,
            "steps_per_snapshot": 20,
            "steps_per_update": 20,
            "window": 10,
            "constraint_fns": [],
            "use_wandb": False,
            "writing_dir": str(tmp_path / "mech1d"),
            "convergence_tolerance": 1e-5,
        },
    )
    assert "profits" in mechanism_data
    assert mechanism_data["profits"].shape[0] == sample.shape[0]
    revenue_mean = float(mechanism_data["revenue"].mean().item())
    print(f"mean revenue = {revenue_mean:.6f}")


def test_sorted_model_parameterization():
    """Sorted models ensure each candidate vector is increasing along dims."""
    mech = mechanism_module.Mechanism(
        npoints=5,
        kernel=dot_kernel,
        y_dim=3,
        temp=5.0,
        is_Y_parameter=True,
        sorted_model=True,
        y_min=0.0,
        y_max=1.0,
    )
    Y = mech.Y_rest
    increments = F.softplus(mech._Y_rest_param)
    assert torch.all(increments >= 0.0)
    assert torch.all(Y >= 0.0)
    assert torch.all(Y <= 1.0)


def test_sorted_model_custom_anchor():
    """Sorted model increments remain strictly positive even without bounds."""
    mech = mechanism_module.Mechanism(
        npoints=4,
        kernel=dot_kernel,
        y_dim=3,
        temp=5.0,
        is_Y_parameter=True,
        sorted_model=True,
    )
    increments = F.softplus(mech._Y_rest_param)
    assert torch.all(increments >= 0.0)
    assert torch.any(increments > 1e-6)


def test_sorted_model_sorts_inputs():
    mech = mechanism_module.Mechanism(
        npoints=6,
        kernel=dot_kernel,
        y_dim=3,
        temp=5.0,
        is_Y_parameter=True,
        sorted_model=True,
    )
    sample = torch.tensor([[0.8, 0.2, 0.5], [0.1, 0.9, 0.4]], dtype=torch.float32)
    choice_unsorted, values_unsorted = mech.forward(sample, selection_mode="soft", already_sorted=False)
    sorted_sample = torch.sort(sample, dim=-1).values
    choice_sorted, values_sorted = mech.forward(sorted_sample, selection_mode="soft", already_sorted=True)
    assert torch.allclose(choice_unsorted, choice_sorted)
    assert torch.allclose(values_unsorted, values_sorted)


def test_training_respects_cost(tmp_path):
    """Training with a constant production cost should converge to that price in 1d."""
    dim = 1
    npoints = 1
    temperature = 40.0

    sample = COST_PRICE + 0.4 * torch.rand(400, dim)
    model_kwargs = {
        "npoints": npoints,
        "kernel": dot_kernel,
        "y_dim": dim,
        "temp": temperature,
        "is_Y_parameter": True,
        "is_there_default": True,
        "y_min": 0.0,
        "y_max": 1.0,
        "cost_fn": constant_cost,
    }
    mechanism_instance = mechanism_module.Mechanism(**model_kwargs)
    mechanism, mechanism_data = mechanism_instance.fit(
        sample,
        modes=["soft"],
        compile=False,
        optimizers_kwargs_dict={"soft": {"lr": 5e-3}},
        schedulers_kwargs_dict={
            "soft": {
                "patience": 50,
                "threshold": 1e-3,
                "factor": 0.5,
                "cooldown": 1,
                "eps": 1e-6,
            },
        },
        train_kwargs={
            "nsteps": 200,
            "steps_per_snapshot": 10000,
            "steps_per_update": 20,
            "window": 200,
            "constraint_fns": [],
            "use_wandb": False,
            "writing_dir": str(tmp_path / "cost"),
            "convergence_tolerance": 1e-4,
            "epsilon": 1e-4,
        },
    )
    intercepts = mechanism_data["intercept"].view(-1)
    final_intercept = intercepts[-1] if mechanism_instance.is_there_default else intercepts[0]
    assert torch.isclose(final_intercept, torch.tensor(COST_PRICE), atol=5e-2), (
        f"expected price ≈{COST_PRICE}, got {final_intercept:.3f}"
    )


def test_unsorted_model_bounds():
    mech = mechanism_module.Mechanism(
        npoints=8,
        kernel=dot_kernel,
        y_dim=1,
        temp=5.0,
        is_Y_parameter=True,
        sorted_model=False,
        y_min=0.2,
        y_max=0.8,
    )
    Y = mech.Y_rest
    assert Y.min() >= 0.2 - 1e-4
    assert Y.max() <= 0.8 + 1e-4


def test_unsorted_model_single_lower_bound():
    mech = mechanism_module.Mechanism(
        npoints=5,
        kernel=dot_kernel,
        y_dim=1,
        temp=5.0,
        is_Y_parameter=True,
        sorted_model=False,
        y_min=0.25,
    )
    Y = mech.Y_rest
    tol = 1e-3
    offset = mech.original_dist_to_bounds
    assert Y.min() >= 0.25 + offset - tol
    assert Y.max() > 0.25


def test_unsorted_model_single_upper_bound():
    mech = mechanism_module.Mechanism(
        npoints=5,
        kernel=dot_kernel,
        y_dim=1,
        temp=5.0,
        is_Y_parameter=True,
        sorted_model=False,
        y_max=0.7,
    )
    Y = mech.Y_rest
    tol = 1e-3
    offset = mech.original_dist_to_bounds
    assert Y.min() <= 0.7 + tol
    assert Y.min() >= 0.7 - offset - tol
    assert Y.max() > 0.7


def test_sorted_chain_revenue_meets_separate():
    dim = 4
    npoints = dim + 1
    mech = mechanism_module.Mechanism(
        npoints=npoints,
        kernel=dot_kernel,
        y_dim=dim,
        temp=10.0,
        is_Y_parameter=True,
        sorted_model=True,
        y_min=0.0,
        y_max=1.0,
    )
    target_increments = torch.zeros(1, npoints, dim)
    for i in range(dim):
        target_increments[0, i + 1, i] = 1.0
    base = torch.full_like(target_increments, -20.0)
    mask = target_increments > 0.0
    base[mask] = torch.log(torch.expm1(target_increments[mask]))
    with torch.no_grad():
        mech._Y_rest_param.copy_(base)
        intercepts = torch.zeros(1, npoints)
        for i in range(1, npoints):
            intercepts[0, i] = 0.5 * i
        mech.intercept_rest.copy_(intercepts)

    N = 200_000
    sample = torch.rand(N, dim)
    sorted_sample, _ = torch.sort(sample, dim=-1, descending=True)
    choice, v = mech.forward(sorted_sample, selection_mode="hard", already_sorted=True)
    ker = mech.kernel_fn(sorted_sample, choice)
    revenue = ker - v
    assert abs(revenue.mean().item() - 0.5) < 1e-2

"""End-to-end CBG-on-SBI: the integration produces valid posterior samples, and
on the linear-Gaussian task (Task 1) the guided samples match the *exact*
analytic posterior — the strongest correctness check, no C2ST needed."""

import math

import pytest
import torch

from experiments.sbi.benchmark import guided_samples, run_benchmark, run_task
from experiments.sbi.tasks import TASKS

UNIFORM_BOUNDS = {"task2": (-1, 1), "task3": (-3, 3), "task4": (-10, 10), "task5": (-1, 1)}


@pytest.mark.parametrize("task_key", list(TASKS))
def test_guided_samples_finite_and_in_bounds(task_key):
    torch.manual_seed(0)
    s = guided_samples(task_key, num_steps=25, num_particles=150, n_samples=600, device="cpu")
    assert s.shape == (600, TASKS[task_key].dim)
    assert torch.isfinite(s).all()
    if task_key in UNIFORM_BOUNDS:
        low, high = UNIFORM_BOUNDS[task_key]
        assert s.min() >= low - 1e-3 and s.max() <= high + 1e-3


def test_task1_mean_recovers_in_ci():
    """Light end-to-end check (runs in CI): the Task-1 posterior mean is exactly
    0.5 y, and the CBG estimator recovers it quickly. The full mean+std recovery
    needs K=1000 to nail the variance -> the `slow` test below."""
    from experiments.sbi.data_io import load_observation

    torch.manual_seed(0)
    y = load_observation("gaussian_linear", 1)
    s = guided_samples("task1", num_steps=80, num_particles=500, n_samples=2000, device="cpu")
    torch.testing.assert_close(s.mean(0).double(), 0.5 * y.double(), atol=0.07, rtol=0)


@pytest.mark.slow
@pytest.mark.parametrize("estimator", ["reinforce", "reparam"])
def test_task1_recovers_exact_gaussian_posterior(estimator):
    """Task 1 posterior is exactly N(0.5 y, 0.05 I). Both CBG estimators must
    converge to it (mean AND per-dim std). Recovering the variance needs K=1000,
    so this is a (local) slow test; CI covers the posterior math via
    test_sbi_posteriors and the mean recovery via the light test above."""
    from experiments.sbi.data_io import load_observation

    torch.manual_seed(0)
    y = load_observation("gaussian_linear", 1)
    s = guided_samples("task1", estimator=estimator, num_steps=100,
                       num_particles=1000, n_samples=4000, device="cpu")
    torch.testing.assert_close(s.mean(0).double(), 0.5 * y.double(), atol=0.05, rtol=0)
    torch.testing.assert_close(s.std(0).mean().double(),
                               torch.tensor(0.05 ** 0.5, dtype=torch.float64), atol=0.03, rtol=0)


def test_task4_guided_mean_near_observation():
    """Gaussian-mixture (Task 4) shares mean=x; the posterior concentrates near y."""
    from experiments.sbi.data_io import load_observation

    torch.manual_seed(0)
    y = load_observation("gaussian_mixture", 1)
    s = guided_samples("task4", num_steps=40, num_particles=300, n_samples=1500, device="cpu")
    assert (s.mean(0) - y).abs().mean() < 1.0


@pytest.mark.slow
def test_run_task_returns_valid_c2st():
    pytest.importorskip("sklearn")
    torch.manual_seed(0)
    c = run_task("task1", num_steps=40, num_particles=300, n_samples=800, device="cpu")
    assert 0.45 <= c <= 1.0


@pytest.mark.slow  # runs C2ST on the full 10^4 reference set
def test_run_benchmark_structure():
    pytest.importorskip("sklearn")
    torch.manual_seed(0)
    res = run_benchmark(num_steps=20, num_particles=100, observations=(1,),
                        device="cpu", tasks=("task1",))
    assert "task1" in res and "average" in res
    assert set(res["task1"]) >= {"c2st_mean", "paper"}
    assert 0.4 <= res["task1"]["c2st_mean"] <= 1.0

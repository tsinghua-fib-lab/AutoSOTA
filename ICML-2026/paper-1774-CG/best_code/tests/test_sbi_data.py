"""SBI benchmark data integrity: every task ships well-formed observations and
reference posterior samples that lie in the prior support."""

import pytest
import torch

from experiments.sbi.data_io import load_observation, load_reference_samples
from experiments.sbi.tasks import TASKS

ALL_TASKS = list(TASKS)
UNIFORM_BOUNDS = {"task2": (-1, 1), "task3": (-3, 3), "task4": (-10, 10), "task5": (-1, 1)}


@pytest.mark.parametrize("task_key", ALL_TASKS)
@pytest.mark.parametrize("num_observation", [1, 5, 10])
def test_reference_samples_well_formed(task_key, num_observation):
    task = TASKS[task_key]
    ref = load_reference_samples(task.name, num_observation)
    assert ref.ndim == 2 and ref.shape[0] > 1000
    assert ref.shape[1] == task.dim, f"{task_key}: ref dim {ref.shape[1]} != {task.dim}"
    assert torch.isfinite(ref).all()


@pytest.mark.parametrize("task_key", ALL_TASKS)
def test_observation_well_formed(task_key):
    task = TASKS[task_key]
    y = load_observation(task.name, 1)
    assert y.ndim == 1 and y.numel() > 0
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("task_key", list(UNIFORM_BOUNDS))
def test_reference_samples_within_uniform_support(task_key):
    """Uniform-prior tasks: reference posterior samples must lie inside the box."""
    low, high = UNIFORM_BOUNDS[task_key]
    ref = load_reference_samples(TASKS[task_key].name, 1)
    assert ref.min() >= low - 1e-4 and ref.max() <= high + 1e-4


def test_gaussian_reference_samples_reasonable():
    """Task 1 posterior is N(0.5 y, 0.05 I); reference samples should match it."""
    from experiments.sbi.data_io import load_observation

    y = load_observation("gaussian_linear", 1)
    ref = load_reference_samples("gaussian_linear", 1)
    torch.testing.assert_close(ref.mean(0), 0.5 * y, atol=0.02, rtol=0)
    torch.testing.assert_close(ref.std(0).mean(), torch.tensor(0.05 ** 0.5), atol=0.02, rtol=0)

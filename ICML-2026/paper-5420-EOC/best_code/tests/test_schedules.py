"""Sanity checks for the dropout schedules."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dropout_mft.schedules import (
    effective_xi,
    schedule_layers,
)

# Schedules that spend exactly the budget h_bar on average.
BUDGET_PRESERVING = [
    "constant",
    "linear",
    "reverse_linear",
    "step",
    "reverse_step",
    "big_step",
]

# Every schedule schedule_layers() knows how to build.
IMPLEMENTED = ["none", "double", "triple", *BUDGET_PRESERVING]


@pytest.mark.parametrize("schedule", BUDGET_PRESERVING)
@pytest.mark.parametrize("depth", [4, 12, 24])
def test_budget_preserving_schedules_keep_mean(schedule, depth):
    h_bar = 0.1
    layers = schedule_layers(schedule, depth, h_bar)
    assert len(layers) == depth
    assert np.mean(layers) == pytest.approx(h_bar)
    assert all(h >= 0 for h in layers)


def test_none_is_all_zero():
    assert schedule_layers("none", 8, 0.1) == [0.0] * 8


@pytest.mark.parametrize("schedule,factor", [("double", 2.0), ("triple", 3.0)])
def test_multiplier_schedules(schedule, factor):
    layers = schedule_layers(schedule, 8, 0.1)
    assert np.mean(layers) == pytest.approx(factor * 0.1)


def test_unknown_schedule_raises():
    with pytest.raises(ValueError):
        schedule_layers("does_not_exist", 8, 0.1)


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_implemented_schedules_build(name):
    layers = schedule_layers(name, 12, 0.1)
    assert len(layers) == 12


def test_effective_xi_infinite_without_dropout():
    assert math.isinf(effective_xi(schedule_layers("none", 8, 0.1)))


def test_effective_xi_finite_with_dropout():
    xi = effective_xi(schedule_layers("constant", 8, 0.1))
    assert math.isfinite(xi)
    assert xi > 0

import pytest
import torch

from verl.trainer.ppo.format_constraint import (
    FormatConstraintConfig,
    FormatConstraintController,
    apply_format_constraint,
)
from verl.utils.dataset.rl_dataset import (
    deterministic_mirror_decision,
    mirror_probability,
)


def test_mirror_ramp_schedule():
    assert mirror_probability(25, 'ramp', 0.5, 50, 100) == 0.0
    assert mirror_probability(50, 'ramp', 0.5, 50, 100) == 0.0
    assert mirror_probability(75, 'ramp', 0.5, 50, 100) == 0.25
    assert mirror_probability(100, 'ramp', 0.5, 50, 100) == 0.5
    assert mirror_probability(125, 'ramp', 0.5, 50, 100) == 0.5


def test_balanced_view_assignment_flips_across_epochs():
    first = [deterministic_mirror_decision(item, 0, 0.5) for item in range(8)]
    second = [deterministic_mirror_decision(item, 1, 0.5) for item in range(8)]
    assert sum(first) == 4
    assert second == [not value for value in first]


def test_format_constraint_only_rewards_formatted_incorrect():
    controller = FormatConstraintController(FormatConstraintConfig(
        target_rate=1.0, dual_lr=1.0, max_bonus=0.2,
        ema_alpha=1.0, warmup_steps=0,
    ))
    scores = torch.zeros(3, 4)
    scores[0, 1] = 1.0
    scores[1, 2] = 0.1
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]])
    updated, metrics = apply_format_constraint(
        token_level_rewards=scores.clone(),
        token_level_scores=scores,
        response_mask=mask,
        controller=controller,
        correctness_score=1.0,
        format_score=0.1,
        global_step=1,
    )
    assert torch.equal(updated[0], scores[0])
    assert torch.equal(updated[2], scores[2])
    assert updated[1, 2] > scores[1, 2]
    assert metrics['format_constraint/eligible_fraction'] == pytest.approx(1.0 / 3.0)


def test_format_constraint_deactivates_after_recovery():
    controller = FormatConstraintController(FormatConstraintConfig(
        target_rate=0.9, dual_lr=1.0, max_bonus=0.2,
        ema_alpha=1.0, warmup_steps=1,
    ))
    controller.observe(0.0, step=1)
    assert controller.coef == 0.0
    controller.observe(0.8, step=2)
    assert abs(controller.coef - 0.1) < 1e-8
    controller.observe(1.0, step=3)
    assert controller.coef == 0.0

import unittest

import numpy as np
import torch

from verl.trainer.ppo.bifrontier import ResponseBudgetConfig, ResponseBudgetController
from verl.trainer.ppo.core_algos import (
    compute_bifrontier_grpo_advantage,
    compute_grpo_outcome_advantage,
    compute_lexiguard_grpo_advantage,
)


def sparse_rewards(values, width=8):
    rewards = torch.zeros(len(values), width, dtype=torch.float32)
    rewards[:, -1] = torch.tensor(values, dtype=torch.float32)
    return rewards


class ResponseBudgetControllerTest(unittest.TestCase):
    def test_warmup_does_not_change_multiplier(self):
        controller = ResponseBudgetController(
            ResponseBudgetConfig(target_tokens=100, dual_lr=0.1, warmup_steps=2)
        )
        controller.observe(200, step=1)
        controller.observe(200, step=2)
        self.assertEqual(controller.coef, 0.0)
        controller.observe(200, step=3)
        self.assertAlmostEqual(controller.coef, 0.1)

    def test_multiplier_is_projected_and_can_recover(self):
        controller = ResponseBudgetController(
            ResponseBudgetConfig(
                target_tokens=100,
                dual_lr=0.2,
                max_coef=0.3,
                warmup_steps=0,
            )
        )
        controller.observe(400, step=1)
        self.assertEqual(controller.coef, 0.3)
        controller.observe(0, step=2)
        self.assertAlmostEqual(controller.coef, 0.1)


class BiFrontierAdvantageTest(unittest.TestCase):
    def setUp(self):
        self.mask = torch.ones(4, 8)
        self.index = np.array(["same"] * 4, dtype=object)

    def test_zero_coefficient_exactly_matches_grpo(self):
        rewards = sparse_rewards([1.0, 1.0, 0.0, 0.0])
        ordinary, _ = compute_grpo_outcome_advantage(rewards, self.mask, self.index)
        candidate, _, _ = compute_bifrontier_grpo_advantage(
            rewards,
            self.mask,
            self.index,
            correctness=torch.tensor([True, True, False, False]),
            response_lengths=torch.tensor([3, 7, 2, 8]),
            length_coef=0.0,
        )
        self.assertTrue(torch.equal(candidate, ordinary))

    def test_all_correct_group_prefers_shorter_correct_response(self):
        rewards = sparse_rewards([1.0, 1.0, 1.0, 1.0])
        advantages, _, metrics = compute_bifrontier_grpo_advantage(
            rewards,
            self.mask,
            self.index,
            correctness=torch.ones(4, dtype=torch.bool),
            response_lengths=torch.tensor([2, 4, 6, 8]),
            length_coef=0.25,
        )
        sequence = advantages[:, 0]
        self.assertGreater(sequence[0], sequence[1])
        self.assertGreater(sequence[1], sequence[2])
        self.assertGreater(sequence[2], sequence[3])
        self.assertAlmostEqual(float(sequence.sum()), 0.0, places=5)
        self.assertEqual(metrics["bifrontier/active_group_fraction"], 1.0)

    def test_wrong_responses_never_get_length_bonus(self):
        rewards = sparse_rewards([1.0, 1.0, 0.0, 0.0])
        ordinary, _ = compute_grpo_outcome_advantage(rewards, self.mask, self.index)
        candidate, _, _ = compute_bifrontier_grpo_advantage(
            rewards,
            self.mask,
            self.index,
            correctness=torch.tensor([True, True, False, False]),
            response_lengths=torch.tensor([3, 7, 1, 8]),
            length_coef=0.35,
        )
        torch.testing.assert_close(candidate[2:], ordinary[2:])
        self.assertGreater(candidate[0, 0], candidate[1, 0])
        self.assertGreater(candidate[1, 0], candidate[2, 0])

    def test_one_correct_response_keeps_ordinary_grpo_signal(self):
        rewards = sparse_rewards([1.0, 0.0, 0.0, 0.0])
        ordinary, _ = compute_grpo_outcome_advantage(rewards, self.mask, self.index)
        candidate, _, metrics = compute_bifrontier_grpo_advantage(
            rewards,
            self.mask,
            self.index,
            correctness=torch.tensor([True, False, False, False]),
            response_lengths=torch.tensor([8, 2, 3, 4]),
            length_coef=0.35,
        )
        torch.testing.assert_close(candidate, ordinary)
        self.assertEqual(metrics["bifrontier/active_group_fraction"], 0.0)


class LexiGuardAdvantageTest(unittest.TestCase):
    def setUp(self):
        self.mask = torch.ones(4, 8)
        self.index = np.array(["same"] * 4, dtype=object)

    def compute(self, correctness, format_valid, lengths, **kwargs):
        return compute_lexiguard_grpo_advantage(
            sparse_rewards([0.0] * 4),
            self.mask,
            self.index,
            correctness=torch.tensor(correctness, dtype=torch.bool),
            format_valid=torch.tensor(format_valid, dtype=torch.bool),
            response_lengths=torch.tensor(lengths),
            **kwargs,
        )

    def test_correctness_remains_above_secondary_format_signal(self):
        advantages, _, _ = self.compute(
            correctness=[True, False, False, False],
            format_valid=[True, True, False, False],
            lengths=[8, 8, 8, 8],
            format_coef=0.5,
        )
        sequence = advantages[:, 0]
        self.assertGreater(sequence[0], torch.max(sequence[1:]))
        self.assertGreater(sequence[1], sequence[2])

    def test_all_wrong_group_learns_format_without_scalar_reward_mixing(self):
        advantages, _, metrics = self.compute(
            correctness=[False, False, False, False],
            format_valid=[True, True, False, False],
            lengths=[8, 8, 8, 8],
            format_coef=0.5,
        )
        sequence = advantages[:, 0]
        self.assertGreater(sequence[0], sequence[2])
        self.assertAlmostEqual(float(sequence.sum()), 0.0, places=5)
        self.assertEqual(metrics["lexiguard/format_active_group_fraction"], 1.0)

    def test_incorrect_tail_penalty_is_suffix_local_and_never_a_bonus(self):
        mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ], dtype=torch.float32)
        advantages, _, metrics = compute_lexiguard_grpo_advantage(
            sparse_rewards([0.0] * 4),
            mask,
            self.index,
            correctness=torch.zeros(4, dtype=torch.bool),
            format_valid=torch.ones(4, dtype=torch.bool),
            response_lengths=mask.sum(-1),
            format_coef=0.0,
            incorrect_tail_tokens=4,
            incorrect_tail_coef=0.05,
        )
        self.assertTrue(torch.equal(advantages[:, :4], torch.zeros(4, 4)))
        self.assertTrue(torch.all(advantages[:3, 4:] == -0.05))
        self.assertTrue(torch.all(advantages[3, 4:] == 0.0))
        self.assertEqual(metrics["lexiguard/incorrect_tail_response_fraction"], 0.75)

    def test_length_tiebreaker_only_changes_correct_responses(self):
        advantages, _, metrics = self.compute(
            correctness=[True, True, False, False],
            format_valid=[True, True, True, True],
            lengths=[2, 8, 1, 8],
            length_coef=0.25,
            format_coef=0.0,
        )
        sequence = advantages[:, 0]
        self.assertGreater(sequence[0], sequence[1])
        self.assertEqual(float(sequence[2]), float(sequence[3]))
        self.assertEqual(metrics["lexiguard/length_active_group_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()

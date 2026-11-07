import unittest

import numpy as np
import torch

from verl.trainer.ppo.core_algos import (
    build_protocol_projection,
    compute_drgrpo_outcome_advantage,
    compute_hcaa_outcome_advantage,
    compute_pda_drgrpo_advantage,
    compute_policy_loss,
)


def sparse_rewards(values, width=3):
    rewards = torch.zeros(len(values), width, dtype=torch.float32)
    rewards[:, -1] = torch.tensor(values, dtype=torch.float32)
    return rewards


class DrGRPOTest(unittest.TestCase):
    def test_advantage_is_group_centered_without_std_normalization(self):
        mask = torch.ones(4, 3)
        advantages, _ = compute_drgrpo_outcome_advantage(
            sparse_rewards([1.0, 0.1, 0.0, 0.0]),
            mask,
            np.array(['same'] * 4, dtype=object),
            advantage_scale=2.0,
        )
        torch.testing.assert_close(
            advantages[:, 0],
            torch.tensor([1.45, -0.35, -0.55, -0.55]),
        )
        self.assertAlmostEqual(float(advantages[:, 0].sum()), 0.0, places=6)

    def test_fixed_policy_denominator_has_expected_scale(self):
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
        zeros = torch.zeros_like(mask)
        advantages = torch.ones_like(mask)
        fixed, clipfrac, _ = compute_policy_loss(
            zeros,
            zeros,
            advantages,
            mask,
            cliprange=0.2,
            loss_normalization='fixed',
            fixed_loss_normalizer=4.0,
        )
        token_mean, _, _ = compute_policy_loss(
            zeros,
            zeros,
            advantages,
            mask,
            cliprange=0.2,
        )
        self.assertAlmostEqual(float(fixed), -3.0 / 8.0)
        self.assertAlmostEqual(float(token_mean), -1.0)
        self.assertEqual(float(clipfrac), 0.0)

    def test_hcaa_amplifies_rare_correct_signal_and_stays_centered(self):
        mask = torch.ones(4, 3)
        advantages, _, metrics = compute_hcaa_outcome_advantage(
            sparse_rewards([1.0, 0.0, 0.0, 0.0]),
            mask,
            np.array(['same'] * 4, dtype=object),
            correctness=torch.tensor([True, False, False, False]),
            advantage_scale=2.0,
            amplification=1.0,
        )
        torch.testing.assert_close(
            advantages[:, 0],
            torch.tensor([2.34375, -0.78125, -0.78125, -0.78125]),
        )
        self.assertAlmostEqual(float(advantages[:, 0].sum()), 0.0, places=6)
        self.assertEqual(metrics['hcaa/active_group_fraction'], 1.0)
        self.assertAlmostEqual(metrics['hcaa/mean_positive_boost'], 1.75)

    def test_hcaa_leaves_uniform_outcome_group_at_zero(self):
        advantages, _, metrics = compute_hcaa_outcome_advantage(
            sparse_rewards([1.0, 1.0, 1.0, 1.0]),
            torch.ones(4, 3),
            np.array(['same'] * 4, dtype=object),
            correctness=torch.ones(4, dtype=torch.bool),
        )
        self.assertTrue(torch.equal(advantages, torch.zeros_like(advantages)))
        self.assertEqual(metrics['hcaa/active_group_fraction'], 0.0)

    def test_clip_higher_preserves_more_positive_advantage_update(self):
        mask = torch.ones(1, 1)
        old_log_prob = torch.zeros(1, 1)
        log_prob = torch.log(torch.full((1, 1), 1.25))
        advantages = torch.ones(1, 1)
        symmetric, _, _ = compute_policy_loss(
            old_log_prob, log_prob, advantages, mask, cliprange=0.2
        )
        clip_higher, _, _ = compute_policy_loss(
            old_log_prob,
            log_prob,
            advantages,
            mask,
            cliprange=0.2,
            cliprange_high=0.28,
        )
        self.assertAlmostEqual(float(symmetric), -1.2, places=6)
        self.assertAlmostEqual(float(clip_higher), -1.25, places=6)

    def test_protocol_projection_marks_tags_and_final_token(self):
        # Tags are [7, 8], [9], and [10, 11]; 0 is padding.
        response_ids = torch.tensor([[1, 2, 7, 8, 3, 9, 4, 10, 11, 99, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
        weights, selected, metrics = build_protocol_projection(
            response_ids=response_ids,
            eos_mask=mask,
            format_valid=torch.tensor([True]),
            protocol_token_sequences=[[7, 8], [9], [10, 11]],
            protocol_mass=12.0,
        )
        expected = torch.tensor(
            [[False, False, True, True, False, True, False, True, True, True, False]]
        )
        self.assertTrue(torch.equal(selected.cpu(), expected))
        self.assertAlmostEqual(float(weights.sum()), 12.0, places=6)
        self.assertAlmostEqual(metrics['pda/protocol_token_fraction'], 0.6)

    def test_protocol_projection_uses_only_invalid_tail(self):
        response_ids = torch.arange(8).unsqueeze(0)
        mask = torch.ones_like(response_ids)
        weights, selected, _ = build_protocol_projection(
            response_ids=response_ids,
            eos_mask=mask,
            format_valid=torch.tensor([False]),
            protocol_token_sequences=[[2, 3]],
            invalid_tail_tokens=3,
            protocol_mass=6.0,
        )
        self.assertTrue(torch.equal(
            selected,
            torch.tensor([[False, False, False, False, False, True, True, True]]),
        ))
        torch.testing.assert_close(weights[0, -3:], torch.full((3,), 2.0))

    def test_pda_matches_decomposed_drgrpo_when_format_is_uniform(self):
        # All responses are formatted, so only correctness residuals remain.
        rewards = sparse_rewards([1.0, 0.1, 0.1, 0.1], width=5)
        advantages, _, metrics = compute_pda_drgrpo_advantage(
            token_level_scores=rewards,
            eos_mask=torch.ones(4, 5),
            index=np.array(['same'] * 4, dtype=object),
            response_ids=torch.arange(20).reshape(4, 5),
            protocol_token_sequences=[[3]],
            correctness_scale=1.8,
            format_scale=0.8,
        )
        torch.testing.assert_close(
            advantages[:, 0], torch.tensor([1.35, -0.45, -0.45, -0.45])
        )
        self.assertEqual(metrics['pda/correctness_active_group_fraction'], 1.0)
        self.assertEqual(metrics['pda/format_active_group_fraction'], 0.0)

    def test_pda_format_credit_is_local_and_group_centered(self):
        rewards = sparse_rewards([0.1, 0.0], width=6)
        advantages, _, metrics = compute_pda_drgrpo_advantage(
            token_level_scores=rewards,
            eos_mask=torch.ones(2, 6),
            index=np.array(['same', 'same'], dtype=object),
            response_ids=torch.tensor([
                [1, 7, 8, 2, 9, 99],
                [1, 2, 3, 4, 5, 99],
            ]),
            protocol_token_sequences=[[7, 8], [9]],
            correctness_scale=1.8,
            format_scale=0.8,
            invalid_tail_tokens=2,
            protocol_mass=4.0,
        )
        # The formatted response receives credit only on tags/final; the
        # malformed response is penalized only on its two-token tail.
        self.assertEqual(float(advantages[0, 0]), 0.0)
        self.assertGreater(float(advantages[0, 1]), 0.0)
        self.assertEqual(float(advantages[1, 3]), 0.0)
        self.assertLess(float(advantages[1, 4]), 0.0)
        self.assertAlmostEqual(float(advantages[0].sum()), 1.6, places=6)
        self.assertAlmostEqual(float(advantages[1].sum()), -1.6, places=6)
        self.assertEqual(metrics['pda/format_active_group_fraction'], 1.0)

    def test_pda_uniform_group_has_zero_advantage(self):
        advantages, _, _ = compute_pda_drgrpo_advantage(
            token_level_scores=sparse_rewards([0.1, 0.1], width=4),
            eos_mask=torch.ones(2, 4),
            index=np.array(['same', 'same'], dtype=object),
            response_ids=torch.arange(8).reshape(2, 4),
            protocol_token_sequences=[[2]],
        )
        self.assertTrue(torch.equal(advantages, torch.zeros_like(advantages)))

    def test_pda_view_conditional_contrast_prioritizes_within_view_signal(self):
        # Both semantic views share a UID. View 0 is mixed while view 1 is
        # uniformly correct, so only view 0 receives the added contrast.
        advantages, _, metrics = compute_pda_drgrpo_advantage(
            token_level_scores=sparse_rewards([1.0, 0.1, 1.0, 1.0], width=3),
            eos_mask=torch.ones(4, 3),
            index=np.array(['same'] * 4, dtype=object),
            response_ids=torch.arange(12).reshape(4, 3),
            protocol_token_sequences=[[2]],
            correctness_scale=1.0,
            format_scale=0.2,
            view_labels=np.array([0, 0, 1, 1], dtype=object),
            view_contrast_scale=1.0,
        )
        torch.testing.assert_close(
            advantages[:, 0], torch.tensor([0.75, -1.25, 0.25, 0.25])
        )
        self.assertAlmostEqual(float(advantages[:, 0].sum()), 0.0, places=6)
        self.assertEqual(
            metrics['pda/view_contrast_active_stratum_fraction'], 0.5
        )
        self.assertGreater(
            metrics['pda/view_contrast_component_abs_token_mean'], 0.0
        )

    def test_pda_view_contrast_requires_view_labels(self):
        with self.assertRaisesRegex(ValueError, 'requires paired view labels'):
            compute_pda_drgrpo_advantage(
                token_level_scores=sparse_rewards([1.0, 0.1], width=3),
                eos_mask=torch.ones(2, 3),
                index=np.array(['same'] * 2, dtype=object),
                response_ids=torch.arange(6).reshape(2, 3),
                protocol_token_sequences=[[2]],
                view_contrast_scale=0.5,
            )

    def test_pda_protocol_guard_is_asymmetric_and_masked(self):
        advantages, _, metrics = compute_pda_drgrpo_advantage(
            token_level_scores=sparse_rewards([0.1, 0.0, 0.0], width=4),
            eos_mask=torch.ones(3, 4),
            index=np.array(['a', 'b', 'c'], dtype=object),
            response_ids=torch.arange(12).reshape(3, 4),
            protocol_token_sequences=[[2]],
            correctness_scale=0.0,
            format_scale=0.0,
            protocol_mass=4.0,
            protocol_guard_coef=0.5,
            protocol_guard_target=0.8,
            protocol_guard_mask=torch.tensor([True, True, False]),
        )
        self.assertAlmostEqual(float(advantages[0].sum()), 0.4, places=6)
        self.assertAlmostEqual(float(advantages[1].sum()), -1.6, places=6)
        self.assertEqual(float(advantages[2].sum()), 0.0)
        self.assertAlmostEqual(
            metrics['pda/protocol_guard_application_fraction'], 2.0 / 3.0
        )
        self.assertGreater(
            metrics['pda/protocol_guard_component_abs_token_mean'], 0.0
        )

    def test_pda_protocol_guard_requires_application_mask(self):
        with self.assertRaisesRegex(ValueError, 'requires an application mask'):
            compute_pda_drgrpo_advantage(
                token_level_scores=sparse_rewards([0.1, 0.0], width=3),
                eos_mask=torch.ones(2, 3),
                index=np.array(['same'] * 2, dtype=object),
                response_ids=torch.arange(6).reshape(2, 3),
                protocol_token_sequences=[[2]],
                protocol_guard_coef=0.1,
            )

    def test_pda_protocol_guard_all_wrong_scope_avoids_mixed_groups(self):
        advantages, _, metrics = compute_pda_drgrpo_advantage(
            token_level_scores=sparse_rewards(
                [1.0, 0.0, 0.1, 0.0], width=4
            ),
            eos_mask=torch.ones(4, 4),
            index=np.array(['mixed', 'mixed', 'null', 'null'], dtype=object),
            response_ids=torch.arange(16).reshape(4, 4),
            protocol_token_sequences=[[2]],
            correctness_scale=0.0,
            format_scale=0.0,
            protocol_mass=4.0,
            protocol_guard_coef=0.5,
            protocol_guard_target=0.8,
            protocol_guard_mask=torch.ones(4, dtype=torch.bool),
            protocol_guard_scope='all_wrong',
        )
        self.assertEqual(float(advantages[0].sum()), 0.0)
        self.assertEqual(float(advantages[1].sum()), 0.0)
        self.assertAlmostEqual(float(advantages[2].sum()), 0.4, places=6)
        self.assertAlmostEqual(float(advantages[3].sum()), -1.6, places=6)
        self.assertEqual(
            metrics['pda/protocol_guard_all_wrong_group_fraction'], 0.5
        )
        self.assertEqual(
            metrics['pda/protocol_guard_scope_response_fraction'], 0.5
        )

    def test_pda_protocol_guard_can_disable_valid_credit(self):
        advantages, _, metrics = compute_pda_drgrpo_advantage(
            token_level_scores=sparse_rewards([0.1, 0.0], width=4),
            eos_mask=torch.ones(2, 4),
            index=np.array(['same', 'same'], dtype=object),
            response_ids=torch.arange(8).reshape(2, 4),
            protocol_token_sequences=[[2]],
            correctness_scale=0.0,
            format_scale=0.0,
            protocol_mass=4.0,
            protocol_guard_coef=0.5,
            protocol_guard_target=0.8,
            protocol_guard_mask=torch.ones(2, dtype=torch.bool),
            protocol_guard_valid_credit=0.0,
        )
        self.assertEqual(float(advantages[0].sum()), 0.0)
        self.assertAlmostEqual(float(advantages[1].sum()), -1.6, places=6)
        self.assertEqual(
            metrics['pda/protocol_guard_positive_response_fraction'], 0.0
        )
        self.assertEqual(
            metrics['pda/protocol_guard_negative_response_fraction'], 0.5
        )

    def test_pda_correct_length_frontier_only_ranks_correct_responses(self):
        advantages, _, metrics = compute_pda_drgrpo_advantage(
            token_level_scores=sparse_rewards([1.0, 1.0, 0.1, 0.0], width=4),
            eos_mask=torch.ones(4, 4),
            index=np.array(['same'] * 4, dtype=object),
            response_ids=torch.arange(16).reshape(4, 4),
            protocol_token_sequences=[[2]],
            correctness_scale=0.0,
            format_scale=0.0,
            correct_length_coef=0.5,
            response_lengths=torch.tensor([10.0, 20.0, 5.0, 1.0]),
        )
        torch.testing.assert_close(
            advantages[:, 0], torch.tensor([0.5, -0.5, 0.0, 0.0])
        )
        self.assertAlmostEqual(float(advantages.sum()), 0.0, places=6)
        self.assertEqual(
            metrics['pda/correct_length_active_group_fraction'], 1.0
        )
        self.assertEqual(
            metrics['pda/correct_length_active_response_fraction'], 0.5
        )

    def test_pda_correct_length_frontier_requires_lengths_when_active(self):
        with self.assertRaisesRegex(ValueError, 'requires response lengths'):
            compute_pda_drgrpo_advantage(
                token_level_scores=sparse_rewards([1.0, 1.0], width=3),
                eos_mask=torch.ones(2, 3),
                index=np.array(['same'] * 2, dtype=object),
                response_ids=torch.arange(6).reshape(2, 3),
                protocol_token_sequences=[[2]],
                correct_length_coef=0.1,
            )


if __name__ == '__main__':
    unittest.main()

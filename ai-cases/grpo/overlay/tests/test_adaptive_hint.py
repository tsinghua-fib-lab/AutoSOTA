import unittest

import numpy as np

from verl.utils.dataset.adaptive_hint import (
    BanditConfig,
    BudgetedHintBandit,
    LeastAssistedMasteryTracker,
    MasteryConfig,
    group_learning_utility,
)


class BudgetedHintBanditTest(unittest.TestCase):
    def test_group_utility_prefers_mixed_rollouts(self):
        self.assertEqual(group_learning_utility(0.0), 0.0)
        self.assertAlmostEqual(group_learning_utility(1.0), 0.2)
        self.assertAlmostEqual(group_learning_utility(0.5), 0.9)
        self.assertGreater(group_learning_utility(0.5), group_learning_utility(1.0))

    def test_policy_matches_budget_for_multiple_targets(self):
        bandit = BudgetedHintBandit(BanditConfig())
        weights = np.array([0.50, 0.35, 0.15])
        for target in (0.05, 0.25, 0.50, 0.90, 0.95):
            policies, _, achieved = bandit.policy(4, target, weights)
            self.assertEqual(policies.shape, (3, 5))
            np.testing.assert_allclose(policies.sum(axis=1), 1.0, atol=1e-10)
            self.assertAlmostEqual(achieved, target, places=8)

    def test_feedback_changes_context_specific_policy(self):
        config = BanditConfig(
            reward_ema_alpha=1.0,
            exploration_coef=0.0,
            temperature=0.10,
        )
        bandit = BudgetedHintBandit(config)
        for arm in range(5):
            bandit.observe(4, context=0, arm=arm, group_utility=1.0 if arm == 4 else 0.0)
            bandit.observe(4, context=2, arm=arm, group_utility=1.0 if arm == 0 else 0.0)
        policies, _, achieved = bandit.policy(4, 0.50, np.array([0.5, 0.0, 0.5]))
        self.assertGreater(policies[0, 4], policies[0, 0])
        self.assertGreater(policies[2, 0], policies[2, 4])
        self.assertAlmostEqual(achieved, 0.50, places=8)

    def test_seeded_sampling_is_reproducible(self):
        bandit = BudgetedHintBandit(BanditConfig())
        weights = np.ones(3)
        first = np.random.RandomState(7)
        second = np.random.RandomState(7)
        draws_a = [bandit.sample(3, 1, 0.4, weights, first)[0] for _ in range(30)]
        draws_b = [bandit.sample(3, 1, 0.4, weights, second)[0] for _ in range(30)]
        self.assertEqual(draws_a, draws_b)


class LeastAssistedMasteryTrackerTest(unittest.TestCase):
    def test_more_assisted_outcome_does_not_contaminate_mastery(self):
        tracker = LeastAssistedMasteryTracker(1, MasteryConfig(ema_alpha=1.0))
        self.assertTrue(tracker.observe(0, success_rate=0.0, hint_fraction=0.25))
        self.assertFalse(tracker.observe(0, success_rate=1.0, hint_fraction=1.0))
        self.assertEqual(tracker.success[0], 0.0)
        self.assertEqual(tracker.hint_floor[0], 0.25)
        self.assertEqual(tracker.context(0), 0)

    def test_less_assisted_outcome_resets_previous_estimate(self):
        tracker = LeastAssistedMasteryTracker(1, MasteryConfig(ema_alpha=0.3))
        tracker.observe(0, success_rate=1.0, hint_fraction=1.0)
        tracker.observe(0, success_rate=0.0, hint_fraction=0.0)
        self.assertEqual(tracker.success[0], 0.0)
        self.assertEqual(tracker.hint_floor[0], 0.0)
        self.assertEqual(tracker.update_counts[0], 1)
        self.assertEqual(tracker.context(0), 0)

    def test_easy_requires_low_assistance_evidence(self):
        tracker = LeastAssistedMasteryTracker(2, MasteryConfig(ema_alpha=1.0))
        tracker.observe(0, success_rate=1.0, hint_fraction=1.0)
        tracker.observe(1, success_rate=1.0, hint_fraction=0.0)
        self.assertEqual(tracker.context(0), 1)
        self.assertEqual(tracker.context(1), 2)
        self.assertEqual(tracker.low_assistance_coverage(), 0.5)

    def test_nearby_lower_assistance_updates_floor_without_reset(self):
        tracker = LeastAssistedMasteryTracker(
            1, MasteryConfig(ema_alpha=0.5, hint_tolerance=0.05)
        )
        tracker.observe(0, success_rate=1.0, hint_fraction=0.25)
        tracker.observe(0, success_rate=0.0, hint_fraction=0.20)
        self.assertAlmostEqual(tracker.success[0], 0.5)
        self.assertAlmostEqual(tracker.hint_floor[0], 0.20)
        self.assertEqual(tracker.update_counts[0], 2)


if __name__ == '__main__':
    unittest.main()

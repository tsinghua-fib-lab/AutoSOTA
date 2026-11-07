import unittest

import torch

from verl.trainer.ppo.deployment_alignment import (
    DeploymentProbeConfig,
    DeploymentProbeController,
)
from verl.trainer.ppo.format_constraint import (
    FormatConstraintConfig,
    FormatConstraintController,
    apply_format_constraint,
    build_format_application_mask,
)


class DeploymentProbeControllerTest(unittest.TestCase):
    def test_repeated_breach_throttles_and_good_probe_recovers(self):
        controller = DeploymentProbeController(
            DeploymentProbeConfig(
                target_format_rate=0.98,
                breach_patience=2,
                decay_factor=0.25,
                recovery_step=0.20,
                warmup_steps=50,
            )
        )
        controller.observe(0.50, step=25)
        self.assertEqual(controller.augmentation_scale, 1.0)
        controller.observe(0.90, step=50)
        self.assertEqual(controller.augmentation_scale, 1.0)
        metrics = controller.observe(0.95, step=75)
        self.assertEqual(controller.augmentation_scale, 0.25)
        self.assertEqual(metrics['deployment_probe/scale_updated'], 1.0)
        controller.observe(0.99, step=100)
        self.assertAlmostEqual(controller.augmentation_scale, 0.45)


class DeploymentFormatConstraintTest(unittest.TestCase):
    def test_anchor_lcb_floor_and_slow_release_keep_guard_active(self):
        controller = FormatConstraintController(
            FormatConstraintConfig(
                target_rate=0.98,
                dual_lr=0.5,
                max_bonus=0.20,
                min_bonus=0.02,
                release_lr_scale=0.25,
                lcb_z=1.0,
                ema_alpha=1.0,
                warmup_steps=0,
                feedback_source='anchor',
                apply_scope='low_hint',
                apply_hint_threshold=1.0 / 3.0,
            )
        )
        breached = controller.observe(0.95, step=1, sample_size=128)
        self.assertGreater(controller.coef, 0.02)
        self.assertLess(
            breached['format_constraint/control_rate'],
            breached['format_constraint/batch_rate'],
        )
        controller.observe(1.0, step=2, sample_size=128)
        self.assertGreaterEqual(controller.coef, 0.02)

    def test_low_hint_scope_expands_beyond_exact_no_hint(self):
        hint_props = torch.tensor([0.0, 0.2, 1.0 / 3.0, 0.5, 1.0])
        selected = build_format_application_mask(
            hint_props, 'low_hint', low_hint_threshold=1.0 / 3.0
        )
        self.assertTrue(torch.equal(
            selected,
            torch.tensor([True, True, True, False, False]),
        ))

    def test_no_hint_scope_controls_observation_and_application(self):
        controller = FormatConstraintController(
            FormatConstraintConfig(
                target_rate=0.8,
                dual_lr=1.0,
                max_bonus=1.0,
                warmup_steps=0,
                feedback_source='no_hint',
                apply_scope='no_hint',
            )
        )
        scores = torch.zeros(4, 2)
        scores[:, -1] = torch.tensor([1.0, 0.1, 0.0, 0.1])
        rewards, metrics = apply_format_constraint(
            token_level_rewards=scores.clone(),
            token_level_scores=scores,
            response_mask=torch.ones_like(scores),
            controller=controller,
            correctness_score=1.0,
            format_score=0.1,
            global_step=1,
            observation_mask=torch.tensor([False, False, True, True]),
            apply_mask=torch.tensor([False, False, True, True]),
        )
        self.assertAlmostEqual(metrics['format_constraint/batch_rate'], 0.5)
        self.assertAlmostEqual(controller.coef, 0.3)
        self.assertAlmostEqual(float(rewards[3, -1]), 0.4)
        self.assertAlmostEqual(float(rewards[1, -1]), 0.1)

    def test_validation_feedback_does_not_update_on_training_batch(self):
        controller = FormatConstraintController(
            FormatConstraintConfig(target_rate=0.9, warmup_steps=0)
        )
        metrics = controller.observe(0.2, step=1, update=False)
        self.assertEqual(controller.coef, 0.0)
        self.assertEqual(metrics['format_constraint/rate_ema'], 0.2)
        controller.observe(0.2, step=1, update=True)
        self.assertGreater(controller.coef, 0.0)


if __name__ == '__main__':
    unittest.main()

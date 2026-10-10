import json
import unittest
from pathlib import Path
import torch
from odelia_autosota import heads
from odelia_autosota.inference import _assert_independent
from odelia_autosota.features import stage_descriptors

torch.set_num_threads(1)


class HeadTests(unittest.TestCase):
    def test_exact_identity_at_initialization(self):
        x = torch.randn(6, 2688)
        z = torch.randn(6, 3)
        for fold in range(7):
            self.assertTrue(torch.equal(heads.build_fold_head(fold)(x, z), z))

    def test_independent_parameters_and_deterministic_initialization(self):
        first = [heads.build_fold_head(f) for f in range(7)]
        _assert_independent(first)
        for f in range(7):
            second = heads.build_fold_head(f)
            for key, value in first[f].state_dict().items():
                self.assertTrue(torch.equal(value, second.state_dict()[key]))
        with self.assertRaises(RuntimeError):
            _assert_independent([first[0]] * 7)

    def test_residual_bound_and_real_gradient_update(self):
        x = torch.randn(9, 2688)
        z = torch.randn(9, 3)
        labels = torch.arange(9) % 3
        head = heads.build_fold_head(0)
        opt = heads.make_optimizer(head)
        loss = heads.training_loss(head, x, z, labels, 0)
        loss.backward()
        opt.step()
        heads.after_optimizer_step(head, 0)
        self.assertFalse(torch.equal(head(x, z), z))
        self.assertLessEqual(float((head(x, z) - z).detach().abs().max()), 0.75 + 1e-6)
        with torch.no_grad():
            head.probe3.up.weight.fill_(1000)
        self.assertLessEqual(
            float(head.probe3(x[:, :768]).detach().abs().max()), 0.25 + 1e-6
        )

    def test_no_cross_fold_dependency(self):
        h0 = heads.build_fold_head(0)
        h1 = heads.build_fold_head(1)
        x = torch.randn(4, 2688)
        z = torch.randn(4, 3)
        before = h0(x, z).clone()
        with torch.no_grad():
            h1.bias.fill_(0.2)
        self.assertTrue(torch.equal(before, h0(x, z)))

    def test_descriptor_order_and_population_std(self):
        stages = [
            torch.arange(c * 2, dtype=torch.float32).reshape(1, c, 1, 1, 2)
            for c in (256, 320, 320)
        ]
        got = stage_descriptors(*stages)
        expected = torch.cat(
            [
                v
                for s in stages
                for v in (
                    s.flatten(2).mean(-1),
                    s.flatten(2).amax(-1),
                    s.flatten(2).std(-1, unbiased=False),
                )
            ],
            dim=1,
        )
        self.assertTrue(torch.equal(got, expected))
        self.assertEqual(got.shape, (1, 2688))

    def test_config_mirror(self):
        path = Path(__file__).resolve().parents[1] / "configs/final_heads.json"
        cfg = json.loads(path.read_text())
        self.assertEqual(
            cfg["fold_specs"],
            {str(k): [v[0], v[1]] for k, v in heads.FOLD_SPECS.items()},
        )

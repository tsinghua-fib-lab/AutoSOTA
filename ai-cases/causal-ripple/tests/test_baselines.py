"""Revision controls preserve schedules and cached-confidence sampling."""

import unittest
from dataclasses import replace

import torch
from test_sampling import ContextModel

from causal_ripple.baselines import confidence_weights, generate_remdm_conf
from causal_ripple.config import get_config
from causal_ripple.run import set_seed


class RevisionBaselineTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_confidence_weights_use_last_decoded_probability(self):
        cache = torch.tensor([0.05, torch.inf, 0.95, 0.5])
        positions = torch.tensor([0, 2, 3])
        weights = confidence_weights(cache, positions)
        self.assertTrue(
            torch.allclose(
                weights, torch.exp(-cache[positions]) / torch.exp(-cache[positions]).sum()
            )
        )
        self.assertGreater(weights[0], weights[2])
        self.assertGreater(weights[2], weights[1])

    def test_large_task_seed_is_valid_for_numpy_and_preserved_for_torch(self):
        seed = 2**62 + 1234
        set_seed(seed)
        self.assertEqual(torch.initial_seed(), seed)

    def test_remdm_finishes_with_compensated_progress_and_fixed_cost(self):
        events = []
        config = replace(get_config("base", "remdm-conf"), revise_every=4)
        model = ContextModel(2)
        prompt = torch.tensor([[2, 3]])
        torch.manual_seed(1234)
        tokens, stats, _ = generate_remdm_conf(
            model,
            prompt,
            steps=32,
            gen_length=64,
            block_length=64,
            mask_id=15,
            config=config,
            trace_callback=events.append,
        )
        self.assertEqual(model.calls, [1] * 32)
        self.assertEqual(stats["revision_events"], 4)
        self.assertTrue(torch.equal(tokens[:, :2], prompt))
        self.assertFalse((tokens[:, 2:] == 15).any())
        self.assertTrue(all(8 <= e["step"] < 24 for e in events))
        self.assertTrue(all(0 <= p < 48 for e in events for p in e["positions"]))
        torch.manual_seed(1234)
        repeated, _, _ = generate_remdm_conf(
            ContextModel(2),
            prompt,
            steps=32,
            gen_length=64,
            block_length=64,
            mask_id=15,
            config=config,
        )
        self.assertTrue(torch.equal(tokens, repeated))


if __name__ == "__main__":
    unittest.main()

import unittest
from dataclasses import replace

import torch

from causal_ripple import get_config
from causal_ripple.revision import select_revision


class ProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.states = []

    def forward(self, x, attention_mask=None):
        self.states.append(x.clone())
        logits = torch.full((x.shape[0], x.shape[1], 16), -10.0)
        for batch in range(len(x)):
            logits[batch, :, 8 if (x[batch] == 9).any() else 4] = 8.0
        return type("Output", (), {"logits": logits})


class RevisionTest(unittest.TestCase):
    def inputs(self):
        x = torch.tensor([[2, 3, 4, 4, 4, 4, 4, 4]])
        scores = torch.full(x.shape, -float("inf"))
        scores[0, 2] = 4.0
        proposed = x.clone()
        proposed[0, 2] = 9
        candidates = torch.ones_like(x, dtype=torch.bool)
        candidates[:, :2] = False
        return x, scores, proposed, candidates, torch.zeros_like(x, dtype=torch.float)

    def test_masks_match_and_contexts_differ_only_at_seed(self):
        model = ProbeModel()
        args = self.inputs()
        result = select_revision(model, *args, mask_id=15, config=get_config("base"))
        old, new = model.states[0]
        self.assertTrue(torch.equal(old == 15, new == 15))
        self.assertEqual(torch.nonzero(old != new).flatten().tolist(), [2])
        self.assertEqual(result.root_position, 2)
        self.assertTrue(result.repair_mask[0, 2])
        self.assertLessEqual(int(result.repair_mask.sum()), 4)
        self.assertFalse(result.repair_mask[0, :2].any())
        self.assertEqual(result.model_calls, 1)

    def test_joint_cap_one_needs_no_probe(self):
        result = select_revision(
            ProbeModel(), *self.inputs(), mask_id=15, config=replace(get_config("base"), max_k=1)
        )
        self.assertEqual(int(result.repair_mask.sum()), 1)
        self.assertEqual(result.model_calls, 0)

    def test_no_actionable_seed_skips_paired_forward(self):
        model = ProbeModel()
        x, scores, proposal, candidates, margin = self.inputs()
        scores.fill_(-float("inf"))
        result = select_revision(
            model, x, scores, proposal, candidates, margin, mask_id=15, config=get_config("base")
        )
        self.assertEqual(model.states, [])
        self.assertFalse(result.repair_mask.any())


if __name__ == "__main__":
    unittest.main()

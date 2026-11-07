import unittest

import torch

from verl.trainer.ppo.repetition_quarantine import (
    RepetitionQuarantineConfig,
    _repetition_onset,
    apply_repetition_quarantine,
)


class RepetitionQuarantineTest(unittest.TestCase):
    def test_detects_third_ngram_and_returns_second_onset(self):
        tokens = [1, 2, 3, 9, 1, 2, 3, 8, 1, 2, 3]
        self.assertEqual(_repetition_onset(tokens, 3, 3), 4)

    def test_never_penalizes_correct_or_formatted_response(self):
        advantages = torch.zeros(2, 12)
        responses = torch.tensor([[1, 2, 1, 2, 1, 2] + [0] * 6] * 2)
        mask = torch.tensor([[1] * 6 + [0] * 6] * 2)
        updated, metrics = apply_repetition_quarantine(
            advantages, responses, mask,
            correctness=torch.tensor([True, False]),
            format_valid=torch.tensor([True, True]),
            config=RepetitionQuarantineConfig(ngram_size=2, min_repeats=3),
        )
        self.assertTrue(torch.equal(updated, advantages))
        self.assertEqual(metrics["repetition_quarantine/active_response_fraction"], 0.0)

    def test_penalty_is_local_to_invalid_suffix(self):
        advantages = torch.zeros(1, 12)
        responses = torch.tensor([[1, 2, 1, 2, 1, 2] + [0] * 6])
        mask = torch.tensor([[1] * 6 + [0] * 6])
        updated, _ = apply_repetition_quarantine(
            advantages, responses, mask,
            correctness=torch.tensor([False]),
            format_valid=torch.tensor([False]),
            config=RepetitionQuarantineConfig(
                coefficient=0.2, ngram_size=2, min_repeats=3, invalid_tail_tokens=20
            ),
        )
        self.assertTrue(torch.equal(updated[0, :2], torch.zeros(2)))
        self.assertTrue(torch.all(updated[0, 2:6] == -0.2))
        self.assertTrue(torch.all(updated[0, 6:] == 0.0))


if __name__ == "__main__":
    unittest.main()

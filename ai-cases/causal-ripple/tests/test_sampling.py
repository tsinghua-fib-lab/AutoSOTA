"""Compatibility fixtures come from the evaluated sampler snapshots."""

import json
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from causal_ripple import generate, get_config


class ContextModel(torch.nn.Module):
    def __init__(self, seed, dtype=torch.bfloat16):
        super().__init__()
        self.seed, self.dtype = seed, dtype
        self.calls = []

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, x, attention_mask=None):
        self.calls.append(int(x.shape[0]))
        positions = torch.arange(x.shape[1], dtype=torch.float32)[None, :, None]
        vocabulary = torch.arange(16, dtype=torch.float32)[None, None, :]
        context = (x.float().sum(-1) % 29)[:, None, None]
        logits = 5 * torch.sin(
            positions * 0.317 + vocabulary * 1.137 + context * 0.227 + self.seed * 0.41
        )
        logits[:, :, 15] = -100
        return type("Output", (), {"logits": logits.to(self.dtype)})


class SamplingCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cases = json.loads(
            (Path(__file__).parent / "fixtures/sampling_compatibility.json").read_text()
        )

    def test_archived_token_and_forward_parity(self):
        for case in self.cases:
            with self.subTest(
                profile=case["profile"], method=case["method"], seed=case["toy_model_seed"]
            ):
                model = ContextModel(case["toy_model_seed"])
                config = replace(
                    get_config(case["profile"], case["method"]), revise_every=case["revise_every"]
                )
                torch.manual_seed(case["seed"])
                actual, stats, _ = generate(
                    model,
                    torch.tensor([case["prompt"]]),
                    steps=case["steps"],
                    gen_length=case["gen_length"],
                    block_length=case["block_length"],
                    remasking=case["method"],
                    mask_id=case["mask_id"],
                    config=config,
                )
                self.assertEqual(actual.tolist(), case["expected_tokens"])
                self.assertEqual(model.calls, case["expected_forward_batches"])
                self.assertEqual(stats["ripple_calls"], case["paired_calls"])

    def test_trace_collection_preserves_generation(self):
        events = []
        config = replace(get_config("bigcodebench"), revise_every=4)
        kwargs = dict(
            steps=32, gen_length=64, block_length=64, remasking="ripple", mask_id=15, config=config
        )
        expected, _, _ = generate(ContextModel(0), torch.tensor([[2, 3]]), **kwargs)
        actual, _, _ = generate(
            ContextModel(0), torch.tensor([[2, 3]]), trace_callback=events.append, **kwargs
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(events)
        self.assertTrue(all(position >= 0 for event in events for position in event["positions"]))

    def test_prompt_and_protected_suffix_are_never_revised(self):
        events = []
        prompt = torch.tensor([[2, 3]])
        config = replace(get_config("ds1000"), revise_every=4)
        output, _, _ = generate(
            ContextModel(1),
            prompt,
            steps=32,
            gen_length=64,
            block_length=64,
            mask_id=15,
            config=config,
            trace_callback=events.append,
        )
        self.assertTrue(torch.equal(output[:, :2], prompt))
        self.assertFalse((output[:, 2:] == 15).any())
        self.assertTrue(all(position < 48 for event in events for position in event["positions"]))


if __name__ == "__main__":
    unittest.main()

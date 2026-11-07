import json
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from causal_ripple.adapters import adapt_ds1000, clean_completion
from causal_ripple.config import PROFILES
from causal_ripple.guards import StaticCertificate
from causal_ripple.refinement import select_by_certificate

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTest(unittest.TestCase):
    def test_profile_snapshots_match_runtime_settings(self):
        for name, config in PROFILES.items():
            with self.subTest(profile=name):
                snapshot = json.loads((ROOT / "configs" / f"{name}.json").read_text())
                self.assertEqual(snapshot, asdict(config))

    def test_reported_counts_match_complete_task_outcomes(self):
        report = json.loads((ROOT / "results/paper_results.json").read_text())
        outcomes = json.loads((ROOT / "results/outcomes.json").read_text())
        for benchmark, expected_n in report["tasks"].items():
            for method, vector in outcomes[benchmark].items():
                with self.subTest(benchmark=benchmark, method=method):
                    self.assertEqual(len(vector), expected_n)
                    self.assertEqual(vector.keys(), outcomes[benchmark]["ripple"].keys())
                    self.assertTrue(all(type(value) is bool for value in vector.values()))
                    if method in report["successful_tasks"]:
                        self.assertEqual(
                            sum(vector.values()), report["successful_tasks"][method][benchmark]
                        )
            cascade = report["cascade"][benchmark]
            base = "cascade_input" if benchmark == "ds1000" else "ripple"
            self.assertEqual(sum(outcomes[benchmark][base].values()), cascade["input_passed"])
            self.assertEqual(
                sum(outcomes[benchmark]["visible_cascade"].values()), cascade["cascade_passed"]
            )

    def test_ds_adapter_preserves_public_variable_contract(self):
        raw = "Problem: double a.\nA:\n<code>\nimport numpy as np\na = np.arange(3)\n</code>\nresult = ... # put solution"
        row = adapt_ds1000(dict(task_id="DS1000/0", prompt=raw, metadata={"library": "Numpy"}))
        self.assertEqual(row["contract"]["output_names"], ["result"])
        self.assertEqual(row["contract"]["provided_names"], ["a", "np"])
        self.assertTrue(row["prompt"].endswith("# SOLUTION START\nresult ="))
        self.assertEqual(
            clean_completion(" a * 2\n# END SOLUTION\nignored", row)[1], "result = a * 2\n"
        )

    def test_ds_adapter_rejects_evaluation_payload(self):
        with self.assertRaises(AssertionError):
            adapt_ds1000(
                dict(task_id="DS1000/0", prompt="", test="private", metadata={"library": "Numpy"})
            )

    def test_certificate_ties_prefer_smaller_edit(self):
        base = torch.tensor([[1, 2, 3, 4]])
        smaller = torch.tensor([[1, 2, 9, 4]])
        larger = torch.tensor([[1, 9, 9, 4]])

        def certificate(tokens, prompt_length, tokenizer):
            return StaticCertificate(1, int((tokens == 9).any()) - 1, 0, 0)

        selected, reason = select_by_certificate(base, [larger, smaller], 1, None, certificate)
        self.assertTrue(torch.equal(selected, smaller))
        self.assertEqual(reason, "certificate")
        selected, reason = select_by_certificate(base, [base.clone()], 1, None, certificate)
        self.assertTrue(torch.equal(selected, base))
        self.assertEqual(reason, "abstain")


if __name__ == "__main__":
    unittest.main()

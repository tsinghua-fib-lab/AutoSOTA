"""BLINK dataset."""

import os
import json
from datetime import datetime
from typing import Dict, List, Callable, Optional

from datasets import load_dataset
from tqdm import tqdm

from .base import BaseDataset


# Default paths (can be overridden via config)
DEFAULT_DATASET_DIR = "./.dataset"
DEFAULT_OUTPUT_DIR = "./output"


class BlinkDataset(BaseDataset):
    """BLINK benchmark dataset."""

    def __init__(self, subset: str = "Counting", dataset_dir: str = None, output_dir: str = None):
        self.subset = subset
        self.data = None
        self.dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR

    def load(self, split: str = "val"):
        ds = load_dataset(
            "BLINK-Benchmark/BLINK", self.subset,
            cache_dir=self.dataset_dir,
        )
        self.data = ds[split]
        return self

    def build_messages(self, sample: Dict) -> List[Dict]:
        """Build messages for BLINK sample."""
        content = []
        for img_key in ["image_1", "image_2", "image_3", "image_4"]:
            if sample.get(img_key) is not None:
                content.append({"type": "image", "image": sample[img_key]})
        content.append({"type": "text", "text": sample["prompt"]})
        return [{"role": "user", "content": content}]

    def get_answer(self, sample: Dict) -> str:
        return sample["answer"]

    def extract_choice(self, sample: Dict, prediction: str) -> Optional[str]:
        """Extract a choice label like '(A)' from free-form prediction."""
        choices = ["(A)", "(B)", "(C)", "(D)", "(E)"][: len(sample.get("choices", []))]
        raw = prediction.strip()

        # Direct forms
        if raw in choices:
            return raw
        if raw in ["A", "B", "C", "D", "E"]:
            return f"({raw})"

        # Check for explicit choice tokens anywhere
        for c in choices:
            if c in prediction:
                return c
        # Check bare letters as tokens
        tokens = set(raw.replace("\n", " ").replace("\t", " ").split())
        letters = {t.strip("()") for t in tokens}
        for idx, c in enumerate(choices):
            letter = chr(ord("A") + idx)
            if letter in letters:
                return c

        # Look at last line separately
        last = raw.splitlines()[-1] if raw else ""
        last_tokens = set(last.replace("\n", " ").replace("\t", " ").split())
        last_letters = {t.strip("()") for t in last_tokens}
        for idx, c in enumerate(choices):
            letter = chr(ord("A") + idx)
            if letter in last_letters:
                return c

        return None

    def check_correct(self, sample: Dict, prediction: str) -> bool:
        extracted = self.extract_choice(sample, prediction)
        return extracted is not None and extracted == sample["answer"]

    def get_result_dict(self, sample: Dict, prediction: str) -> Dict:
        """Build result dict for a sample."""
        parsed_choice = self.extract_choice(sample, prediction)
        return {
            "idx": sample["idx"],
            "question": sample["question"],
            "prompt": sample["prompt"],
            "choices": sample["choices"],
            "answer": sample["answer"],
            "prediction": prediction,
            "prediction_choice": parsed_choice,
        }

    def evaluate(
        self,
        inference_fn: Callable[[Dict], Dict],
        num_samples: int = None,
    ) -> Dict:
        """
        Evaluate on this dataset using provided inference function.

        Args:
            inference_fn: Function that takes a sample dict and returns
                          {"prediction": str, **metrics}
            num_samples: Number of samples to run (None = all)

        Returns:
            Dict with results and summary
        """
        data = self.data
        if num_samples:
            data = data.select(range(min(num_samples, len(data))))

        results = []
        correct = 0

        print(f"\nEvaluating BLINK/{self.subset}")
        print(f"Total samples: {len(data)}")

        for i in tqdm(range(len(data)), desc="Processing"):
            sample = data[i]

            # Call inference function
            output = inference_fn(sample)
            pred = output["prediction"]

            is_correct = self.check_correct(sample, pred)
            if is_correct:
                correct += 1

            results.append({
                **self.get_result_dict(sample, pred),
                **{k: v for k, v in output.items() if k != "prediction"},
            })

            # Print some samples
            if i < 3 or (i + 1) % 50 == 0:
                print(f"\n[{sample['idx']}] Answer: {sample['answer']} | Pred: {pred[:60]}...")

        # Compute summary
        accuracy = correct / len(data) * 100 if len(data) > 0 else 0

        # Compute average metrics from results
        metric_keys = [k for k in results[0].keys() if k.endswith("_ms") or k.endswith("_tokens")]
        avg_metrics = {}
        for key in metric_keys:
            values = [r[key] for r in results if key in r]
            if values:
                avg_metrics[f"avg_{key}"] = sum(values) / len(values)

        summary = {
            "dataset": f"BLINK/{self.subset}",
            "total_samples": len(data),
            "correct": correct,
            "accuracy": accuracy,
            **avg_metrics,
        }

        print(f"\n{'=' * 60}")
        print("RESULTS")
        print("=" * 60)
        print(f"Dataset: BLINK/{self.subset}")
        print(f"Accuracy: {correct}/{len(data)} = {accuracy:.2f}%")
        for key, val in avg_metrics.items():
            print(f"{key}: {val:.2f}")

        return {"results": results, "summary": summary}

    def save_results(self, output: Dict, label: str) -> str:
        """Save results to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = os.path.join(self.output_dir, f"blink_{self.subset.lower()}_{label}", timestamp)
        os.makedirs(output_subdir, exist_ok=True)

        with open(os.path.join(output_subdir, "results.json"), "w") as f:
            json.dump(output["results"], f, indent=2, ensure_ascii=False)

        summary = {**output["summary"], "timestamp": datetime.now().isoformat()}
        with open(os.path.join(output_subdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to {output_subdir}")
        return output_subdir

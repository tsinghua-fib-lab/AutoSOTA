"""MMBench dataset."""

import os
import json
from datetime import datetime
from typing import Dict, List, Callable, Optional

from datasets import load_dataset
from tqdm import tqdm

from .base import BaseDataset


DEFAULT_DATASET_DIR = "./.dataset"
DEFAULT_OUTPUT_DIR = "./output"


class MMBenchDataset(BaseDataset):
    """MMBench benchmark dataset.

    Multiple choice visual question answering.
    Dataset: lmms-lab/MMBench
    """

    def __init__(self, subset: str = "en", dataset_dir: str = None, output_dir: str = None):
        self.subset = subset
        self.data = None
        self.dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR

    def load(self, split: str = "dev"):
        ds = load_dataset(
            "lmms-lab/MMBench",
            self.subset,
            cache_dir=self.dataset_dir,
        )
        self.data = ds[split]
        return self

    def build_messages(self, sample: Dict) -> List[Dict]:
        """Build messages for MMBench sample."""
        content = []
        if sample.get("image") is not None:
            content.append({"type": "image", "image": sample["image"]})

        # Build prompt with question and choices
        question = sample["question"]
        choices = []
        for key in ["A", "B", "C", "D"]:
            if sample.get(key):
                choices.append(f"({key}) {sample[key]}")

        prompt = f"{question}\n" + "\n".join(choices)
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def get_answer(self, sample: Dict) -> str:
        return sample["answer"]

    def extract_choice(self, sample: Dict, prediction: str) -> Optional[str]:
        """Extract choice from prediction."""
        choices = ["A", "B", "C", "D"]
        raw = prediction.strip()

        # Direct letter match
        if raw in choices:
            return raw
        if raw in ["(A)", "(B)", "(C)", "(D)"]:
            return raw[1]

        # Check for choice in prediction
        for c in choices:
            if f"({c})" in prediction:
                return c
            if prediction.startswith(c + ".") or prediction.startswith(c + " ") or prediction.startswith(c + ")"):
                return c

        # Check answer text match
        for c in choices:
            if sample.get(c) and sample[c].lower() in prediction.lower():
                return c

        # Last line check
        last = raw.splitlines()[-1] if raw else ""
        for c in choices:
            if c in last.split():
                return c

        return None

    def check_correct(self, sample: Dict, prediction: str) -> bool:
        extracted = self.extract_choice(sample, prediction)
        return extracted is not None and extracted == sample["answer"]

    def get_result_dict(self, sample: Dict, prediction: str) -> Dict:
        """Build result dict for a sample."""
        parsed_choice = self.extract_choice(sample, prediction)
        return {
            "index": sample.get("index", ""),
            "question": sample["question"],
            "choices": {k: sample.get(k, "") for k in ["A", "B", "C", "D"]},
            "answer": sample["answer"],
            "prediction": prediction,
            "prediction_choice": parsed_choice,
            "category": sample.get("category", ""),
            "l2_category": sample.get("L2-category", ""),
        }

    def evaluate(
        self,
        inference_fn: Callable[[Dict], Dict],
        num_samples: int = None,
    ) -> Dict:
        """Evaluate on this dataset using provided inference function."""
        data = self.data
        if num_samples:
            data = data.select(range(min(num_samples, len(data))))

        results = []
        correct = 0

        print(f"\nEvaluating MMBench/{self.subset}")
        print(f"Total samples: {len(data)}")

        for i in tqdm(range(len(data)), desc="Processing"):
            sample = data[i]
            output = inference_fn(sample)
            pred = output["prediction"]

            is_correct = self.check_correct(sample, pred)
            if is_correct:
                correct += 1

            results.append({
                **self.get_result_dict(sample, pred),
                **{k: v for k, v in output.items() if k != "prediction"},
            })

            if i < 3 or (i + 1) % 50 == 0:
                print(f"\n[{i}] Answer: {sample['answer']} | Pred: {pred[:60]}...")

        accuracy = correct / len(data) * 100 if len(data) > 0 else 0

        metric_keys = [k for k in results[0].keys() if k.endswith("_ms") or k.endswith("_tokens")]
        avg_metrics = {}
        for key in metric_keys:
            values = [r[key] for r in results if key in r]
            if values:
                avg_metrics[f"avg_{key}"] = sum(values) / len(values)

        summary = {
            "dataset": f"MMBench/{self.subset}",
            "total_samples": len(data),
            "correct": correct,
            "accuracy": accuracy,
            **avg_metrics,
        }

        print(f"\n{'=' * 60}")
        print("RESULTS")
        print("=" * 60)
        print(f"Dataset: MMBench/{self.subset}")
        print(f"Accuracy: {correct}/{len(data)} = {accuracy:.2f}%")
        for key, val in avg_metrics.items():
            print(f"{key}: {val:.2f}")

        return {"results": results, "summary": summary}

    def save_results(self, output: Dict, label: str) -> str:
        """Save results to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = os.path.join(self.output_dir, f"mmbench_{self.subset}_{label}", timestamp)
        os.makedirs(output_subdir, exist_ok=True)

        with open(os.path.join(output_subdir, "results.json"), "w") as f:
            json.dump(output["results"], f, indent=2, ensure_ascii=False)

        summary = {**output["summary"], "timestamp": datetime.now().isoformat()}
        with open(os.path.join(output_subdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to {output_subdir}")
        return output_subdir

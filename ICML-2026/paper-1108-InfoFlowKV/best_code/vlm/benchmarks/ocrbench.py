"""OCRBench dataset."""

import os
import re
import json
from datetime import datetime
from typing import Dict, List, Callable, Optional

from datasets import load_dataset
from tqdm import tqdm

from .base import BaseDataset


DEFAULT_DATASET_DIR = "./.dataset"
DEFAULT_OUTPUT_DIR = "./output"


class OCRBenchDataset(BaseDataset):
    """OCRBench benchmark dataset.

    OCR and text-centric VQA evaluation.
    Dataset: echo840/OCRBench
    Mixed evaluation: exact match for text recognition, relaxed for VQA.
    """

    def __init__(self, dataset_dir: str = None, output_dir: str = None):
        self.data = None
        self.dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR

    def load(self, split: str = "test"):
        ds = load_dataset(
            "echo840/OCRBench",
            cache_dir=self.dataset_dir,
        )
        # OCRBench might only have one split
        available_splits = list(ds.keys())
        if split not in available_splits:
            split = available_splits[0]
        self.data = ds[split]
        return self

    def build_messages(self, sample: Dict) -> List[Dict]:
        """Build messages for OCRBench sample."""
        content = []
        if sample.get("image") is not None:
            content.append({"type": "image", "image": sample["image"]})

        question = sample.get("question", sample.get("query", ""))
        content.append({"type": "text", "text": question})
        return [{"role": "user", "content": content}]

    def get_answer(self, sample: Dict) -> str:
        answer = sample.get("answer", sample.get("answers", ""))
        if isinstance(answer, list):
            return answer[0] if answer else ""
        return str(answer)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def _contains_answer(self, prediction: str, answer: str) -> bool:
        """Check if prediction contains the answer."""
        norm_pred = self._normalize_text(prediction)
        norm_ans = self._normalize_text(answer)
        return norm_ans in norm_pred

    def check_correct(self, sample: Dict, prediction: str) -> bool:
        """Check correctness - exact match or containment."""
        answer = self.get_answer(sample)

        # Exact match after normalization
        if self._normalize_text(prediction) == self._normalize_text(answer):
            return True

        # Answer contained in prediction
        if self._contains_answer(prediction, answer):
            return True

        # Handle multiple acceptable answers
        answers = sample.get("answer", sample.get("answers", []))
        if isinstance(answers, list):
            for ans in answers:
                if self._normalize_text(prediction) == self._normalize_text(str(ans)):
                    return True
                if self._contains_answer(prediction, str(ans)):
                    return True

        return False

    def get_result_dict(self, sample: Dict, prediction: str) -> Dict:
        """Build result dict for a sample."""
        return {
            "question": sample.get("question", sample.get("query", "")),
            "answer": self.get_answer(sample),
            "prediction": prediction,
            "dataset_type": sample.get("dataset", ""),
            "type": sample.get("type", ""),
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

        # Track per-type accuracy
        type_correct = {}
        type_total = {}

        print(f"\nEvaluating OCRBench")
        print(f"Total samples: {len(data)}")

        for i in tqdm(range(len(data)), desc="Processing"):
            sample = data[i]
            output = inference_fn(sample)
            pred = output["prediction"]

            is_correct = self.check_correct(sample, pred)
            if is_correct:
                correct += 1

            # Track per-type
            sample_type = sample.get("type", "unknown")
            type_total[sample_type] = type_total.get(sample_type, 0) + 1
            if is_correct:
                type_correct[sample_type] = type_correct.get(sample_type, 0) + 1

            results.append({
                **self.get_result_dict(sample, pred),
                "correct": is_correct,
                **{k: v for k, v in output.items() if k != "prediction"},
            })

            if i < 3 or (i + 1) % 50 == 0:
                print(f"\n[{i}] Answer: {self.get_answer(sample)[:30]} | Pred: {pred[:60]}...")

        accuracy = correct / len(data) * 100 if len(data) > 0 else 0

        # Per-type accuracy
        type_accuracy = {}
        for t in type_total:
            type_accuracy[t] = type_correct.get(t, 0) / type_total[t] * 100

        metric_keys = [k for k in results[0].keys() if k.endswith("_ms") or k.endswith("_tokens")]
        avg_metrics = {}
        for key in metric_keys:
            values = [r[key] for r in results if key in r]
            if values:
                avg_metrics[f"avg_{key}"] = sum(values) / len(values)

        summary = {
            "dataset": "OCRBench",
            "total_samples": len(data),
            "correct": correct,
            "accuracy": accuracy,
            "type_accuracy": type_accuracy,
            **avg_metrics,
        }

        print(f"\n{'=' * 60}")
        print("RESULTS")
        print("=" * 60)
        print(f"Dataset: OCRBench")
        print(f"Overall Accuracy: {correct}/{len(data)} = {accuracy:.2f}%")
        print("\nPer-type accuracy:")
        for t, acc in sorted(type_accuracy.items()):
            print(f"  {t}: {acc:.2f}%")
        for key, val in avg_metrics.items():
            print(f"{key}: {val:.2f}")

        return {"results": results, "summary": summary}

    def save_results(self, output: Dict, label: str) -> str:
        """Save results to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = os.path.join(self.output_dir, f"ocrbench_{label}", timestamp)
        os.makedirs(output_subdir, exist_ok=True)

        with open(os.path.join(output_subdir, "results.json"), "w") as f:
            json.dump(output["results"], f, indent=2, ensure_ascii=False)

        summary = {**output["summary"], "timestamp": datetime.now().isoformat()}
        with open(os.path.join(output_subdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to {output_subdir}")
        return output_subdir
